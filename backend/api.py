"""F10 backend -- one field, one button, one progress indicator, no chat.

Single streaming request per generation (GET /api/generate?target=X), not a
POST-then-poll-a-separate-endpoint design: a serverless deployment (Vercel)
may route each HTTP request to a different function instance with no shared
memory, so coordinating a run_id across three separate requests via an
in-memory dict -- the original design, fine for a long-lived local uvicorn
process -- would silently break in production. One open SSE connection for
the whole run (progress events, then a final event carrying the report HTML
itself) needs no cross-request state at all.

No auth, no persisted user data (brief Section 6/8) -- deliberately quick
and dirty here; the retrieval/scoring/citation layers underneath are the
carefully-built part.
"""

from __future__ import annotations

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
from backend.report_renderer import render_html

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


def _generate_events(target: str):
    """The single SSE stream for one generation: progress events, then
    exactly one terminal event carrying either the report HTML or an error."""
    cached = load_cached_report(target)
    if cached is not None:
        yield _sse({"stage": "cache", "status": "done", "detail": f"served from {cached.run_label}"})
        yield _sse({"stage": "_report", "html": cache_banner_html(cached) + cached.html})
        return

    if not live_runs_enabled():
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
        yield _sse({"stage": "_report", "html": render_html(result_holder["result"].report)})


@app.get("/api/generate")
def generate(target: str) -> StreamingResponse:
    target = (target or "").strip()
    if not target:
        raise HTTPException(400, "target is required")
    return StreamingResponse(_generate_events(target), media_type="text/event-stream")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
