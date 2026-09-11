"""F10 backend -- one field, one button, one progress indicator, no chat.

In-memory run store only (brief Section 6: "file cache instead of DB, no
auth" -- deliberately quick and dirty here; the retrieval/scoring/citation
layers underneath are the carefully-built part). Runs are not persisted
across process restarts, which is fine for a single-session demo.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline import PipelineResult, run_pipeline
from backend.report_renderer import render_html

app = FastAPI(title="First Pass API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # no auth, no user data at rest -- see brief Section 6/8
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@dataclass
class RunState:
    run_id: str
    target: str
    status: str = "running"  # "running" | "done" | "error"
    events: list[dict] = field(default_factory=list)
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    _subscribers: list[queue.Queue] = field(default_factory=list)

    def publish(self, event: dict) -> None:
        self.events.append(event)
        for sub in self._subscribers:
            sub.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        for event in self.events:  # replay history for a late subscriber
            q.put(event)
        if self.status != "running":
            q.put({"stage": "_end", "status": self.status})
        self._subscribers.append(q)
        return q


_RUNS: dict[str, RunState] = {}


def _execute(run: RunState) -> None:
    def on_progress(stage: str, status: str, detail: Optional[str]) -> None:
        run.publish({"stage": stage, "status": status, "detail": detail})

    try:
        result = run_pipeline(run.target, on_progress=on_progress)
        run.result = result
        run.status = "done"
        run.publish({"stage": "_end", "status": "done"})
    except Exception as exc:  # a crashed run must surface, never hang the UI forever
        run.status = "error"
        run.error = str(exc)
        run.publish({"stage": "_end", "status": "error", "detail": str(exc)})


@app.post("/api/reports")
def start_report(payload: dict) -> dict:
    target = (payload.get("target") or "").strip()
    if not target:
        raise HTTPException(400, "target is required")

    run_id = uuid.uuid4().hex[:12]
    run = RunState(run_id=run_id, target=target)
    _RUNS[run_id] = run

    thread = threading.Thread(target=_execute, args=(run,), daemon=True)
    thread.start()
    return {"run_id": run_id}


@app.get("/api/reports/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")

    def event_stream():
        q = run.subscribe()
        while True:
            event = q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("stage") == "_end":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/reports/{run_id}/report", response_class=HTMLResponse)
def get_report(run_id: str) -> str:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run_id")
    if run.status == "running":
        raise HTTPException(425, "report not ready yet")
    if run.status == "error":
        raise HTTPException(500, run.error or "pipeline failed")
    return render_html(run.result.report)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
