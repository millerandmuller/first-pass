"""Tests for F10's API layer -- a single streaming /api/generate request.

One open connection per generation: progress events, then a terminal event
carrying the report HTML (or an error) -- no separate start/poll/fetch
requests and so no cross-request state, which is what makes this safe to
run on a serverless deployment (see backend/api.py's module docstring).

Live-run tests use a non-demo target (PD-L1) deliberately: GLP-1R/HER2/KRAS
are the three F11 demo targets and get served from the F13 cache once
pre-warmed (see test_demo_cache.py and backend/demo_cache.py).
"""

import json

from fastapi.testclient import TestClient

from backend import api as api_module
from backend.api import app

client = TestClient(app)


def _read_sse_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_requires_target():
    resp = client.get("/api/generate", params={"target": ""})
    assert resp.status_code == 400


def test_full_run_streams_progress_then_a_terminal_report_event():
    with client.stream("GET", "/api/generate", params={"target": "PD-L1"}) as resp:
        events = _read_sse_events(resp)

    stages = [e["stage"] for e in events]
    assert "target_resolution" in stages
    assert "evidence_retrieval" in stages
    assert "section_graph" in stages
    assert events[-1]["stage"] == "_report"
    assert "PD-L1" in events[-1]["html"]


def test_demo_target_is_served_from_cache_not_a_live_run(tmp_path, monkeypatch):
    from backend import demo_cache

    monkeypatch.setattr(demo_cache, "_CACHE_DIR", str(tmp_path))
    demo_cache.save_cached_report("HER2", "<p>cached HER2 report</p>", 99.0, "dry-run-1")

    with client.stream("GET", "/api/generate", params={"target": "HER2"}) as resp:
        events = _read_sse_events(resp)

    stages = [e["stage"] for e in events]
    assert stages == ["cache", "_report"]  # no live pipeline stages at all
    assert "cached HER2 report" in events[-1]["html"]
    assert "pre-computed" in events[-1]["html"].lower()  # the honest cache banner


def test_live_run_rate_limit_is_enforced_between_consecutive_calls(monkeypatch):
    # Force the rate-limit branch deterministically rather than relying on
    # real elapsed time between two live pipeline calls in the test suite,
    # and stub out run_pipeline entirely: closing the client stream early
    # does NOT stop the background daemon thread that runs it (Python
    # cannot forcibly interrupt a running thread), so without this stub a
    # real live pipeline would keep running to completion regardless of
    # what the test itself observes -- silently wasting real AWS time/cost.
    from backend.pipeline import PipelineResult

    def fake_run_pipeline(target, on_progress):
        on_progress("target_resolution", "done", "stubbed, no live call")
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub report</p>")
    monkeypatch.setattr(api_module, "_last_live_run_started_at", __import__("time").time())
    monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_LIVE_RUNS", 0.2)

    with client.stream("GET", "/api/generate", params={"target": "PD-L1"}) as resp:
        events = _read_sse_events(resp)

    assert events[0]["stage"] == "rate_limit"
    assert events[-1]["stage"] == "_report"
