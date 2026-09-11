"""F15 -- AgentCore built-in evaluations (Builtin.Faithfulness, Builtin.Correctness).

Getting a real evaluation call to succeed required reverse-engineering the
exact span shape `bedrock-agentcore evaluate` accepts -- none of it is
guessable from the CLI help text alone:

1. Strands' internal Tracer is a true process-wide singleton
   (strands/telemetry/tracer.py: `_tracer_instance`), created on first
   `get_tracer()` call from whatever `opentelemetry.trace.get_tracer_provider()`
   returns *at that moment* -- and ignoring any `tracer_provider` passed to
   `StrandsTelemetry(...)` entirely. A custom capturing provider must be
   registered via `opentelemetry.trace.set_tracer_provider(...)` before the
   first Strands Agent anywhere in the process, which is why this module is
   imported and installs itself from backend/__init__.py rather than lazily.
2. Strands' synchronous `Agent.__call__` bridges into its async
   implementation via `loop.run_in_executor(...)`, so spans actually get
   created in a ThreadPoolExecutor worker thread, not the calling thread --
   a `threading.local()` capture buffer keyed on the caller's thread never
   sees them (this was the original, broken design here; verified by
   checking span count with a debug script before switching to a single
   shared list + lock, see _GlobalCaptureExporter).
3. `evaluate` wants the human-readable span shape (flat-dict `attributes`,
   not the OTLP wire array-of-{key,value} format) plus an explicit
   `"scope": {"name": "strands.telemetry.tracer"}` sibling key -- neither is
   present in either the console exporter's default printout or a canonical
   OTLP JSON export; both had to be reproduced by hand.
4. It specifically wants the top-level `"invoke_agent <name>"` span (which
   carries `gen_ai.agent.name` and the full system/user/choice message
   events), not the lower-level `"chat"` span for an individual model call --
   the latter is rejected as lacking "model/tool/agent invocation details"
   even though it has the same message content.
5. Verified live 2026-09-11: a Faithfulness call against a real trace
   returned a real score, label, and LLM-judge explanation -- see
   DECISION_LOG.md / directives/common_issues.md for the full story.

Scope for this build: one report-generation run captured at a time process-
wide (`run_with_trace_capture` holds a lock for its duration) -- matches the
API layer's rate-limited-live-runs design (F13). Not built to concurrently
attribute spans to several truly simultaneous runs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import boto3
import opentelemetry.trace as trace_api
from botocore.exceptions import BotoCoreError, ClientError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from backend.config import AWS_REGION, EVALUATOR_CORRECTNESS, EVALUATOR_FAITHFULNESS

T = TypeVar("T")

_STRANDS_TRACER_SCOPE = "strands.telemetry.tracer"
_provider_lock = threading.Lock()
_provider_installed = False
_capture_lock = threading.Lock()  # serializes run_with_trace_capture calls, see below
_global_spans: list = []


class _GlobalCaptureExporter(SpanExporter):
    """Appends every span from every thread to one shared list.

    Strands' synchronous `Agent.__call__` bridges into its async
    implementation via `loop.run_in_executor(...)`, so the actual span
    creation happens in a ThreadPoolExecutor worker thread, not the calling
    thread -- a `threading.local()` buffer keyed on the *caller's* thread
    never sees those spans (verified: this was the original, broken design).
    A single shared list sidesteps that entirely; `run_with_trace_capture`
    takes `_capture_lock` for its whole run and slices off only what was
    appended during that window, which is correct for one run at a time.
    Concurrent overlapping runs would need real per-run correlation (e.g. by
    trace_id) instead -- out of scope for this build, see module docstring.
    """

    def export(self, spans) -> SpanExportResult:
        _global_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def ensure_trace_capture_installed() -> None:
    """Install a capturing TracerProvider as the process-wide OTel default.

    Idempotent and safe to call multiple times; only the first call has any
    effect (OTel's global provider can only be set once per process). Must
    run before the first Strands Agent is constructed anywhere in the
    process -- see module docstring point 1.
    """
    global _provider_installed
    with _provider_lock:
        if _provider_installed:
            return
        provider = TracerProvider()
        # SimpleSpanProcessor (not BatchSpanProcessor) -- synchronous, no
        # background worker thread of its own, so a span is appended to
        # _global_spans as soon as it ends, from whichever thread that is.
        provider.add_span_processor(SimpleSpanProcessor(_GlobalCaptureExporter()))
        trace_api.set_tracer_provider(provider)
        _provider_installed = True


def run_with_trace_capture(fn: Callable[[], T]) -> tuple[T, list]:
    """Run `fn` and return (result, spans emitted anywhere during that call).

    Holds `_capture_lock` for the duration of `fn()` so only one call runs at
    a time process-wide -- see _GlobalCaptureExporter's docstring for why a
    shared list, sliced by a before/after index, is used instead of a
    thread-local buffer.
    """
    ensure_trace_capture_installed()
    with _capture_lock:
        start_index = len(_global_spans)
        result = fn()
        spans = list(_global_spans[start_index:])
    return result, spans


def _readable_span_to_eval_dict(span) -> Optional[dict]:
    """Convert an OTel ReadableSpan to the flat-dict shape `evaluate` accepts,
    for spans whose name starts with "invoke_agent" (see module docstring
    point 3) -- other span types are not evaluable and return None."""
    if not span.name.startswith("invoke_agent"):
        return None

    def _fmt_id(id_int: int, width: int) -> str:
        return format(id_int, f"0{width}x")

    return {
        "name": span.name,
        "context": {
            "trace_id": f"0x{_fmt_id(span.context.trace_id, 32)}",
            "span_id": f"0x{_fmt_id(span.context.span_id, 16)}",
            "trace_state": "[]",
        },
        "kind": str(span.kind),
        "parent_id": f"0x{_fmt_id(span.parent.span_id, 16)}" if span.parent else None,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status": {"status_code": span.status.status_code.name},
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ],
        "links": [],
        "resource": {"attributes": dict(span.resource.attributes or {})},
        "scope": {"name": _STRANDS_TRACER_SCOPE},
    }


@dataclass
class EvaluationOutcome:
    evaluator_id: str
    status: str  # "live" | "gap"
    value: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


_agentcore_client = None


def _get_agentcore_client():
    global _agentcore_client
    if _agentcore_client is None:
        _agentcore_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    return _agentcore_client


def evaluate_spans(spans: list, evaluator_id: str) -> EvaluationOutcome:
    """Run a built-in AgentCore evaluator against captured spans from one run.

    Calls boto3's bedrock-agentcore data-plane client directly. An earlier
    version of this shelled out to `aws bedrock-agentcore evaluate` instead,
    on the (wrong, unverified) assumption that the pinned boto3 lacked this
    operation -- that broke a real deployment: Vercel's Python runtime has
    no `aws` CLI binary on PATH, so every evaluation failed there with
    "[Errno 2] No such file or directory: 'aws'" even though the pipeline
    otherwise ran correctly end-to-end. boto3 1.43.92 (already pinned) has
    `bedrock-agentcore` client's `evaluate()` natively; verified live.
    """
    eval_spans = [d for d in (_readable_span_to_eval_dict(s) for s in spans) if d is not None]
    if not eval_spans:
        return EvaluationOutcome(evaluator_id, status="gap", error="no evaluable invoke_agent span captured")

    try:
        response = _get_agentcore_client().evaluate(
            evaluatorId=evaluator_id,
            evaluationInput={"sessionSpans": eval_spans},
        )
    except (BotoCoreError, ClientError) as exc:
        return EvaluationOutcome(evaluator_id, status="gap", error=str(exc)[:500])

    try:
        first = response["evaluationResults"][0]
        return EvaluationOutcome(
            evaluator_id=evaluator_id,
            status="live",
            value=first.get("value"),
            label=first.get("label"),
            explanation=first.get("explanation"),
        )
    except (KeyError, IndexError) as exc:
        return EvaluationOutcome(evaluator_id, status="gap", error=f"unexpected response shape: {exc}")


def evaluate_faithfulness(spans: list) -> EvaluationOutcome:
    return evaluate_spans(spans, EVALUATOR_FAITHFULNESS)


def evaluate_correctness(spans: list) -> EvaluationOutcome:
    return evaluate_spans(spans, EVALUATOR_CORRECTNESS)
