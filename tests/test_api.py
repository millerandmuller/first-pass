"""Tests for F10's API layer -- start a run, stream progress, fetch the report.

Live-run tests use a non-demo target (PD-L1) deliberately: GLP-1R/HER2/KRAS
are the three F11 demo targets and get served from the F13 cache once
pre-warmed (see test_demo_cache.py and backend/demo_cache.py), which would
make a live-pipeline assertion here flaky depending on cache state.
"""

import json

from fastapi.testclient import TestClient

from backend import demo_cache
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


def test_start_report_requires_target():
    resp = client.post("/api/reports", json={"target": ""})
    assert resp.status_code == 400


def test_report_not_ready_returns_425():
    # Deliberately a target that fails target resolution (no Bedrock call,
    # no CitationLedger evidence path) rather than a real target: this test
    # only cares about the immediate-GET race, and section_graph's global
    # trace-capture lock (backend/evaluations.py) serializes any real
    # pipeline run behind this one if it used Bedrock, doubling wall time
    # and AWS cost in this file for a case that does not need it.
    start = client.post("/api/reports", json={"target": "ZZZ-NOT-A-REAL-TARGET-88888"})
    run_id = start.json()["run_id"]
    resp = client.get(f"/api/reports/{run_id}/report")
    assert resp.status_code == 425


def test_full_run_streams_events_and_serves_report():
    start = client.post("/api/reports", json={"target": "PD-L1"})
    assert start.status_code == 200
    run_id = start.json()["run_id"]

    with client.stream("GET", f"/api/reports/{run_id}/events") as resp:
        events = _read_sse_events(resp)

    stages = [e["stage"] for e in events]
    assert "target_resolution" in stages
    assert "evidence_retrieval" in stages
    assert "section_graph" in stages
    assert events[-1]["stage"] == "_end"
    assert events[-1]["status"] == "done"

    report_resp = client.get(f"/api/reports/{run_id}/report")
    assert report_resp.status_code == 200
    assert "PD-L1" in report_resp.text


def test_unknown_run_id_returns_404():
    assert client.get("/api/reports/does-not-exist/report").status_code == 404
    assert client.get("/api/reports/does-not-exist/events").status_code == 404


def test_demo_target_is_served_from_cache_not_a_live_run(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_cache, "_CACHE_DIR", str(tmp_path))
    demo_cache.save_cached_report("HER2", "<p>cached HER2 report</p>", 99.0, "dry-run-1")

    start = client.post("/api/reports", json={"target": "HER2"})
    run_id = start.json()["run_id"]

    with client.stream("GET", f"/api/reports/{run_id}/events") as resp:
        events = _read_sse_events(resp)

    stages = [e["stage"] for e in events]
    assert stages == ["cache", "_end"]  # no live pipeline stages at all

    report_resp = client.get(f"/api/reports/{run_id}/report")
    assert report_resp.status_code == 200
    assert "cached HER2 report" in report_resp.text
    assert "pre-computed" in report_resp.text.lower()  # the honest cache banner
