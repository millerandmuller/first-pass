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
import time

from fastapi.testclient import TestClient

from backend import api as api_module
from backend.api import app

client = TestClient(app)


def _read_sse_events(response) -> list[dict]:
    return _read_sse_events_from_text("\n".join(response.iter_lines()))


def _read_sse_events_from_text(raw: str) -> list[dict]:
    events = []
    for line in raw.splitlines():
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


def test_full_run_streams_progress_then_a_terminal_report_event(monkeypatch):
    # A valid judge code is required for a real (non-demo, non-empty) target
    # since the Cost Guard landed -- see the judge-access tests below. This
    # is a genuine end-to-end smoke test (real openFDA + real Bedrock, no
    # mocks) and is NOT run as part of routine verification for that reason;
    # run it deliberately when you want to confirm the live path still works.
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
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


def test_heartbeat_comments_keep_a_slow_stage_alive_without_becoming_events(monkeypatch):
    # A stage that runs long with no progress event (e.g. section_graph's
    # 60-90s single call) must not leave the SSE connection silent long
    # enough to trip an idle-connection timeout at an intermediate proxy.
    import time as time_module

    from backend.pipeline import PipelineResult

    monkeypatch.setattr(api_module, "_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [])

    def slow_fake_pipeline(target, on_progress):
        on_progress("target_resolution", "done", None)
        time_module.sleep(0.3)  # several heartbeat intervals with no event
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", slow_fake_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub</p>")

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        raw = resp.read().decode()

    assert raw.count(": heartbeat\n\n") >= 2
    # heartbeats are SSE comments (no "data:" prefix) -- confirm EventSource
    # would never surface them as parsed events by checking none of the
    # actual data-events accidentally look like a heartbeat.
    data_events = _read_sse_events_from_text(raw)
    assert all(e.get("stage") != "heartbeat" for e in data_events)


def test_live_run_rate_limit_is_enforced_between_consecutive_calls(monkeypatch):
    # Force the rate-limit branch deterministically rather than relying on
    # real elapsed time between two live pipeline calls in the test suite,
    # and stub out run_pipeline entirely: closing the client stream early
    # does NOT stop the background daemon thread that runs it (Python
    # cannot forcibly interrupt a running thread), so without this stub a
    # real live pipeline would keep running to completion regardless of
    # what the test itself observes -- silently wasting real AWS time/cost.
    # resolve_target is stubbed too, purely so the real openFDA round-trip's
    # variable latency can't eat into the tight 0.2s window below.
    from types import SimpleNamespace

    from backend.pipeline import PipelineResult

    def fake_run_pipeline(target, on_progress):
        on_progress("target_resolution", "done", "stubbed, no live call")
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "resolve_target", lambda target: SimpleNamespace(found=True))
    monkeypatch.setattr(api_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub report</p>")
    monkeypatch.setattr(api_module, "_last_live_run_started_at", __import__("time").time())
    monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_LIVE_RUNS", 0.2)
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [])

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert events[0]["stage"] == "rate_limit"
    assert events[-1]["stage"] == "_report"


def test_a_not_found_target_does_not_consume_the_live_run_cooldown(monkeypatch):
    # The cooldown exists to space out real Bedrock-costing runs. A target
    # that resolves to "not found" returns in well under a second with no
    # Bedrock call at all -- it must not burn the same 30s cooldown slot a
    # real live run would, or a garbage/mistyped target starves the next
    # genuine visitor's live run of its rate-limit window for no reason.
    from backend.pipeline import PipelineResult

    def fake_not_found_pipeline(target, on_progress):
        on_progress("target_resolution", "empty", f"no drug names resolved for '{target}'")
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", fake_not_found_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub</p>")
    monkeypatch.setattr(api_module, "_last_live_run_started_at", 0.0)
    monkeypatch.setattr(api_module, "_MIN_SECONDS_BETWEEN_LIVE_RUNS", 10_000)

    with client.stream("GET", "/api/generate", params={"target": "ZZZ-GARBAGE"}) as resp:
        events = _read_sse_events(resp)

    # _last_live_run_started_at is 0.0 (year 1970) so no wait is ever
    # triggered for THIS request either way -- the real assertion is that
    # a not-found run does not bump the timestamp forward for the NEXT one.
    assert events[0]["stage"] != "rate_limit"
    assert api_module._last_live_run_started_at == 0.0


def test_kill_switch_blocks_live_runs_but_not_cached_demo_targets(tmp_path, monkeypatch):
    # LIVE_RUNS_ENABLED is the one hard stop on Bedrock spend (the per-process
    # rate limit is best-effort, the billing alarm only emails). Off must mean
    # zero pipeline calls for a live target, while the pre-computed demo
    # targets keep serving -- they never cost anything.
    from backend import demo_cache

    calls = []

    def must_not_run(target, on_progress):
        calls.append(target)
        raise AssertionError("pipeline ran while live runs were disabled")

    monkeypatch.setattr(api_module, "run_pipeline", must_not_run)
    monkeypatch.setattr(demo_cache, "_CACHE_DIR", str(tmp_path))
    demo_cache.save_cached_report("HER2", "<p>cached HER2 report</p>", 99.0, "dry-run-1")
    # A valid judge code is required to even reach the kill switch (see
    # test_public_request_without_a_code_is_blocked_before_the_kill_switch
    # below) -- this test is specifically about the kill switch, so it
    # supplies one and stays scoped to that one check.
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")

    for value in ("0", "false", "No", " OFF "):
        monkeypatch.setenv("LIVE_RUNS_ENABLED", value)
        with client.stream(
            "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
        ) as resp:
            events = _read_sse_events(resp)
        assert events == [{"stage": "_report_error", "detail": api_module.LIVE_RUNS_PAUSED_MESSAGE}]

    with client.stream("GET", "/api/generate", params={"target": "HER2"}) as resp:
        events = _read_sse_events(resp)
    assert [e["stage"] for e in events] == ["cache", "_report"]
    assert calls == []


def test_kill_switch_defaults_to_on(monkeypatch):
    monkeypatch.delenv("LIVE_RUNS_ENABLED", raising=False)
    assert api_module.live_runs_enabled() is True
    monkeypatch.setenv("LIVE_RUNS_ENABLED", "1")
    assert api_module.live_runs_enabled() is True
    monkeypatch.setenv("LIVE_RUNS_ENABLED", "anything-else")
    assert api_module.live_runs_enabled() is True


def test_bedrock_throttling_surfaces_a_plain_english_message_not_the_raw_exception(monkeypatch):
    # Observed live in production (2026-09-11 demo rehearsal): an ordinary,
    # non-adversarial second live run tripped Bedrock's ConverseStream
    # throttling and the raw botocore ClientError string reached the user
    # via the error box, reading as a crash rather than "try again shortly."
    def fake_throttled_pipeline(target, on_progress):
        raise Exception(
            "An error occurred (ThrottlingException) when calling the ConverseStream "
            "operation (reached max retries: 4): Too many requests, please wait before "
            "trying again."
        )

    monkeypatch.setattr(api_module, "run_pipeline", fake_throttled_pipeline)
    monkeypatch.setattr(api_module, "_last_live_run_started_at", 0.0)
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [])

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert events[-1]["stage"] == "_report_error"
    assert "ThrottlingException" not in events[-1]["detail"]
    assert "rate-limiting" in events[-1]["detail"].lower()


# Cost Guard (2026-09-14): the public site serves the three cached demo
# targets and the free "no evidence found" result to everyone; a target that
# resolves to real evidence needs a live model call, reserved for hackathon
# judges via JUDGE_ACCESS_CODE. See DECISION_LOG.md, 2026-09-13, for the
# design this implements.


def test_public_request_without_a_code_is_blocked_before_the_kill_switch(monkeypatch):
    # This is the free-tier boundary itself: no JUDGE_ACCESS_CODE configured
    # at all, a real target, no code on the request -- must never reach the
    # pipeline (real Bedrock spend), regardless of the kill switch's state.
    monkeypatch.delenv("JUDGE_ACCESS_CODE", raising=False)

    def must_not_run(target, on_progress):
        raise AssertionError("pipeline ran for an unauthorized public request")

    monkeypatch.setattr(api_module, "run_pipeline", must_not_run)

    with client.stream("GET", "/api/generate", params={"target": "PD-L1"}) as resp:
        events = _read_sse_events(resp)

    assert events == [{"stage": "_report_error", "detail": api_module.PUBLIC_DEMO_MODE_MESSAGE}]


def test_wrong_code_is_rejected_same_as_no_code(monkeypatch):
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "correct-code")

    def must_not_run(target, on_progress):
        raise AssertionError("pipeline ran with a wrong code")

    monkeypatch.setattr(api_module, "run_pipeline", must_not_run)

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "wrong-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert events == [{"stage": "_report_error", "detail": api_module.PUBLIC_DEMO_MODE_MESSAGE}]


def test_not_found_target_stays_free_even_with_no_code_and_kill_switch_off(monkeypatch):
    # The exact regression this feature had to avoid (DECISION_LOG.md,
    # 2026-09-13): a target with nothing to write about costs nothing, so it
    # must stay open to the public even while the kill switch is off and no
    # judge code is present -- a real Bedrock-costing gate must never block
    # a request that was never going to touch Bedrock in the first place.
    monkeypatch.delenv("JUDGE_ACCESS_CODE", raising=False)
    monkeypatch.setenv("LIVE_RUNS_ENABLED", "0")

    from backend.pipeline import PipelineResult

    calls = []

    def fake_not_found_pipeline(target, on_progress):
        calls.append(target)
        on_progress("target_resolution", "empty", f"no drug names resolved for '{target}'")
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", fake_not_found_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub</p>")

    with client.stream("GET", "/api/generate", params={"target": "ZZZ-GARBAGE"}) as resp:
        events = _read_sse_events(resp)

    assert calls == ["ZZZ-GARBAGE"]
    assert events[-1]["stage"] == "_report"
    assert events[-1] != {"stage": "_report_error", "detail": api_module.LIVE_RUNS_PAUSED_MESSAGE}


def test_valid_code_allows_a_real_target_through_to_the_pipeline(monkeypatch):
    from backend.pipeline import PipelineResult

    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [])
    monkeypatch.setattr(api_module, "_last_live_run_started_at", 0.0)

    calls = []

    def fake_pipeline(target, on_progress):
        calls.append(target)
        on_progress("target_resolution", "done", None)
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub</p>")

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert calls == ["PD-L1"]
    assert events[-1]["stage"] == "_report"
    assert len(api_module._JUDGE_LIVE_RUN_TIMESTAMPS) == 1  # this run was counted toward the daily cap


def test_daily_judge_cap_blocks_further_live_runs_once_reached(monkeypatch):
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_MAX_JUDGE_LIVE_RUNS_PER_DAY", 1)
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [time.time()])  # cap already reached

    def must_not_run(target, on_progress):
        raise AssertionError("pipeline ran past the daily judge cap")

    monkeypatch.setattr(api_module, "run_pipeline", must_not_run)

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert events == [{"stage": "_report_error", "detail": api_module.JUDGE_DAILY_CAP_MESSAGE}]


def test_judge_cap_window_is_rolling_24h_not_a_hard_reset(monkeypatch):
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "test-judge-code")
    monkeypatch.setattr(api_module, "_MAX_JUDGE_LIVE_RUNS_PER_DAY", 1)
    stale_timestamp = time.time() - 90_000  # more than 24h ago
    monkeypatch.setattr(api_module, "_JUDGE_LIVE_RUN_TIMESTAMPS", [stale_timestamp])
    monkeypatch.setattr(api_module, "_last_live_run_started_at", 0.0)

    from backend.pipeline import PipelineResult

    def fake_pipeline(target, on_progress):
        on_progress("target_resolution", "done", None)
        return PipelineResult(report=None, target_resolution=None, is_partial=False)

    monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(api_module, "render_html", lambda report: "<p>stub</p>")

    with client.stream(
        "GET", "/api/generate", params={"target": "PD-L1", "code": "test-judge-code"}
    ) as resp:
        events = _read_sse_events(resp)

    assert events[-1]["stage"] == "_report"


def test_access_endpoint_confirms_a_valid_code_and_rejects_everything_else(monkeypatch):
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "correct-code")
    assert client.get("/api/access", params={"code": "correct-code"}).json() == {"valid": True}
    assert client.get("/api/access", params={"code": "wrong-code"}).json() == {"valid": False}
    assert client.get("/api/access").json() == {"valid": False}

    monkeypatch.delenv("JUDGE_ACCESS_CODE", raising=False)
    assert client.get("/api/access", params={"code": "correct-code"}).json() == {"valid": False}
