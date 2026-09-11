"""Tests for F10's API layer -- start a run, stream progress, fetch the report.

Runs the real pipeline against real openFDA data (no mocks), same as the
manual browser verification. Confirms the SSE contract the frontend depends
on: an event per stage, a terminal "_end" event, and a 425 for a report
fetched before the run finishes.
"""

import json

from fastapi.testclient import TestClient

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
    start = client.post("/api/reports", json={"target": "GLP-1R"})
    run_id = start.json()["run_id"]
    resp = client.get(f"/api/reports/{run_id}/report")
    assert resp.status_code == 425


def test_full_run_streams_events_and_serves_partial_report():
    start = client.post("/api/reports", json={"target": "GLP-1R"})
    assert start.status_code == 200
    run_id = start.json()["run_id"]

    with client.stream("GET", f"/api/reports/{run_id}/events") as resp:
        events = _read_sse_events(resp)

    stages = [e["stage"] for e in events]
    assert "target_resolution" in stages
    assert "evidence_retrieval" in stages
    assert "interpretation_swarm" in stages
    assert events[-1]["stage"] == "_end"
    assert events[-1]["status"] == "done"

    report_resp = client.get(f"/api/reports/{run_id}/report")
    assert report_resp.status_code == 200
    assert "GLP-1R" in report_resp.text
    assert "PARTIAL RUN" in report_resp.text


def test_unknown_run_id_returns_404():
    assert client.get("/api/reports/does-not-exist/report").status_code == 404
    assert client.get("/api/reports/does-not-exist/events").status_code == 404
