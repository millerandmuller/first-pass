"""F10 backend -- one field, one button, one progress indicator, no chat.

Single streaming request per generation (GET /api/generate?target=X), not a
POST-then-poll-a-separate-endpoint design: a serverless deployment (Vercel)
may route each HTTP request to a different function instance with no shared
memory, so coordinating a run_id across three separate requests via an
in-memory dict -- the original design, fine for a long-lived local uvicorn
process -- would silently break in production. One open SSE connection for
the whole run (progress events, then a final event carrying the report HTML
itself) needs no cross-request state at all.

No auth, no persisted user data -- deliberately quick and dirty here; the
retrieval/scoring/citation layers underneath are the carefully-built part.
"""

from __future__ import annotations

import hmac
import json
import os
import queue
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.demo_cache import cache_banner_html, load_cached_report
from backend.pipeline import run_pipeline
from backend.report_renderer import render_html, report_to_dict
from backend.target_resolution import resolve_target

app = FastAPI(title="First Pass API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_HEARTBEAT_INTERVAL_SECONDS = 15

# Best-effort live-run rate limit (F13 rule constraint): a live pipeline run
# touches real Bedrock/AgentCore spend, so cap concurrency and enforce a
# minimum gap between live runs process-wide. This is best-effort, not a
# hard security boundary -- a serverless deployment can have multiple warm
# instances that would each track this independently. The hard stop is the
# LIVE_RUNS_ENABLED kill switch below; the AWS billing alarm only notifies.
_LIVE_RUN_LOCK = threading.Lock()
_MIN_SECONDS_BETWEEN_LIVE_RUNS = 30
_last_live_run_started_at = 0.0

# Kill switch for every Bedrock-costing path. Default on; set the environment
# variable LIVE_RUNS_ENABLED to 0/false/no/off to stop all live runs without a
# code change. The three pre-computed demo targets keep serving from cache
# either way -- they never touch Bedrock. Read per request, not at import,
# so a test (or a future runtime config source) can flip it without a reload.
_LIVE_RUNS_ENABLED_ENV = "LIVE_RUNS_ENABLED"
_FALSY = {"0", "false", "no", "off"}
LIVE_RUNS_PAUSED_MESSAGE = (
    "Live runs are currently paused. The three pre-computed demo targets "
    "(GLP-1R, HER2, KRAS) are still available."
)


def live_runs_enabled() -> bool:
    return os.environ.get(_LIVE_RUNS_ENABLED_ENV, "1").strip().lower() not in _FALSY


# Cost Guard: the public site serves the three cached demo targets plus the
# free "no evidence found" result (target resolution itself never touches
# Bedrock) to everyone. A target that DOES resolve to real evidence is about
# to trigger a real, paid Strands/Bedrock run -- reserved for hackathon
# judges during the judging period (through 2026-10-08), via a code carried
# in the private Devpost testing-instructions field, never the public site.
# Nothing here is mocked for the public: the same real pipeline and the same
# real empty-result path run either way, just gated at the point where a run
# would actually start costing money. Unset JUDGE_ACCESS_CODE means no one
# has judge-level access, independent of the LIVE_RUNS_ENABLED kill switch
# below (which still applies on top of a valid code -- see F13/F11 notes and
# DECISION_LOG.md, 2026-09-13).
_JUDGE_ACCESS_CODE_ENV = "JUDGE_ACCESS_CODE"

# Best-effort daily cap on judge-authorized live runs, process-wide (same
# multi-instance caveat as _LIVE_RUN_LOCK above) -- bounds worst-case spend
# if the code ever leaks past the judges it was given to.
_MAX_JUDGE_LIVE_RUNS_PER_DAY = 40
_JUDGE_LIVE_RUN_TIMESTAMPS: list[float] = []

PUBLIC_DEMO_MODE_MESSAGE = (
    "This target has real evidence to write about, which means a live model run -- "
    "reserved for hackathon judges during the judging period. Try GLP-1R, HER2, or "
    "KRAS for a full example now."
)
JUDGE_DAILY_CAP_MESSAGE = (
    "The judge live-run cap for today has been reached. The three pre-computed demo "
    "targets (GLP-1R, HER2, KRAS) are still available; live runs reopen tomorrow."
)


def _valid_judge_code(code: Optional[str]) -> bool:
    configured = os.environ.get(_JUDGE_ACCESS_CODE_ENV, "").strip()
    if not configured or not code:
        return False
    return hmac.compare_digest(configured, code.strip())


def _judge_daily_cap_reached() -> bool:
    cutoff = time.time() - 86400
    while _JUDGE_LIVE_RUN_TIMESTAMPS and _JUDGE_LIVE_RUN_TIMESTAMPS[0] < cutoff:
        _JUDGE_LIVE_RUN_TIMESTAMPS.pop(0)
    return len(_JUDGE_LIVE_RUN_TIMESTAMPS) >= _MAX_JUDGE_LIVE_RUNS_PER_DAY


def _record_judge_live_run() -> None:
    _JUDGE_LIVE_RUN_TIMESTAMPS.append(time.time())


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _friendly_error_message(exc: Exception) -> str:
    """Bedrock throttling is a real, observed-in-production failure mode (not
    hypothetical -- a live demo rehearsal tripped it under ordinary sequential
    use, no adversarial load). The raw botocore ClientError string ("An error
    occurred (ThrottlingException) when calling the ConverseStream operation
    (reached max retries: 4)...") reads to a non-technical user as a crash
    rather than a wait-and-retry situation, so it gets a plain-English
    substitute here. Anything else surfaces as-is."""
    text = str(exc)
    if "ThrottlingException" in text or "Too many requests" in text:
        return "Bedrock is rate-limiting requests right now. Please wait a minute and try again."
    return text


def _generate_events(target: str, code: Optional[str] = None):
    """The single SSE stream for one generation: progress events, then
    exactly one terminal event carrying either the report HTML or an error."""
    cached = load_cached_report(target)
    if cached is not None:
        yield _sse({"stage": "cache", "status": "done", "detail": f"served from {cached.run_label}"})
        report_event = {"stage": "_report", "html": cache_banner_html(cached) + cached.html}
        if cached.report is not None:
            report_event["report"] = cached.report
        yield _sse(report_event)
        return

    # Free openFDA lookup, never Bedrock -- pipeline.run_pipeline takes the
    # same free early-return path for a target with no resolvable evidence.
    # Whether this run is about to cost money is decided by this result, not
    # by the target string itself, so every cost-guard check below applies
    # only when it does; a target with nothing to write about stays open to
    # everyone even while the kill switch is off or no code is present.
    resolution = resolve_target(target)
    will_cost_money = resolution.found

    if will_cost_money and not _valid_judge_code(code):
        # A distinct terminal stage, not "_report_error": this is not a
        # failure, it is the demo working exactly as designed for the vast
        # majority of visitors. The frontend renders it as a CTA modal
        # (book a demo) rather than the red error banner used for genuine
        # failures, so a public visitor does not read "reserved for judges"
        # as "the app is broken."
        yield _sse({"stage": "_judge_only", "detail": PUBLIC_DEMO_MODE_MESSAGE})
        return

    if will_cost_money and _judge_daily_cap_reached():
        yield _sse({"stage": "_report_error", "detail": JUDGE_DAILY_CAP_MESSAGE})
        return

    if will_cost_money and not live_runs_enabled():
        yield _sse({"stage": "_report_error", "detail": LIVE_RUNS_PAUSED_MESSAGE})
        return

    with _LIVE_RUN_LOCK:
        wait_needed = _MIN_SECONDS_BETWEEN_LIVE_RUNS - (time.time() - _last_live_run_started_at)
        if wait_needed > 0:
            yield _sse(
                {
                    "stage": "rate_limit",
                    "status": "running",
                    "detail": f"waiting {wait_needed:.0f}s before starting a live run",
                }
            )
            time.sleep(wait_needed)
        # The cooldown timestamp is NOT stamped here. A garbage target that
        # resolves to "not found" returns in well under a second with no
        # Bedrock call at all -- stamping "now" before resolution even runs
        # would make that free, instant request consume the same 30s cooldown
        # as a real ~150s Bedrock-costing run. It is stamped from on_progress
        # below instead, the first time target resolution actually succeeds.

    q: queue.Queue = queue.Queue()
    result_holder: dict = {}

    def on_progress(stage: str, status: str, detail: Optional[str]) -> None:
        if stage == "target_resolution" and status == "done":
            global _last_live_run_started_at
            with _LIVE_RUN_LOCK:
                _last_live_run_started_at = time.time()
            _record_judge_live_run()
        q.put({"stage": stage, "status": status, "detail": detail})

    def worker() -> None:
        try:
            result_holder["result"] = run_pipeline(target, on_progress=on_progress)
        except Exception as exc:  # a crashed run must surface, never hang the UI forever
            result_holder["error"] = _friendly_error_message(exc)
        q.put({"stage": "_end", "status": "error" if "error" in result_holder else "done"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            # A stage like section_graph runs for 60-90s with no progress
            # event in between (it's one call into the multi-agent graph);
            # a silent SSE connection that long risks an idle-connection
            # timeout at an intermediate proxy/CDN hop. A `: comment\n\n`
            # line is a no-op per the SSE spec -- it resets any such idle
            # timer without the frontend needing to handle a new event type.
            event = q.get(timeout=_HEARTBEAT_INTERVAL_SECONDS)
        except queue.Empty:
            yield ": heartbeat\n\n"
            continue
        if event["stage"] == "_end":
            break
        yield _sse(event)

    if "error" in result_holder:
        yield _sse({"stage": "_report_error", "detail": result_holder["error"]})
    else:
        result = result_holder["result"]
        report_event = {"stage": "_report", "html": render_html(result.report)}
        # result.report is None only in tests that stub run_pipeline with a
        # bare PipelineResult -- a real run always produces a Report.
        if result.report is not None:
            report_event["report"] = report_to_dict(
                result.report,
                served="live",
                nonclinical_label_count=result.nonclinical_label_count,
            )
        yield _sse(report_event)


@app.get("/api/generate")
def generate(target: str, code: Optional[str] = None) -> StreamingResponse:
    target = (target or "").strip()
    if not target:
        raise HTTPException(400, "target is required")
    return StreamingResponse(_generate_events(target, code), media_type="text/event-stream")


@app.get("/api/access")
def access(code: str = "") -> dict:
    """Lets the frontend confirm a code is valid before it claims judge
    access on screen -- this app never silently presents a state as true
    that isn't (see demo_cache.cache_banner_html's own rule)."""
    return {"valid": _valid_judge_code(code)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
