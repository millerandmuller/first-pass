"""Tests for the pipeline orchestrator against real upstream data.

Runs end-to-end for a real demo target using every non-LLM stage (F6, F1,
F8, F9). The LLM-dependent stages are expected to report "pending" while
Bedrock Anthropic access is unavailable -- these assertions describe that
honest partial state, not a fake completed one.
"""

from backend.pipeline import run_pipeline


def test_glp1r_end_to_end_produces_a_partial_report_with_pending_llm_stages():
    events = []
    result = run_pipeline("GLP-1R", on_progress=lambda stage, status, detail: events.append((stage, status)))

    assert result.is_partial is True
    assert ("target_resolution", "done") in events
    assert ("evidence_retrieval", "done") in events
    assert ("interpretation_swarm", "pending") in events
    assert ("section_graph", "pending") in events

    report = result.report
    assert len(report.sections) == 7
    assert len(report.ledger) > 0
    # Section 2 (genetics) must use the real curated GLP-1R precedent, not a placeholder
    assert "glucose" in report.sections[1].html_body.lower()


def test_unknown_target_produces_honest_empty_report_not_a_crash():
    result = run_pipeline("ZZZ-NOT-A-REAL-TARGET-99999")
    assert result.target_resolution.found is False
    assert "No evidence found" in result.report.sections[0].html_body


def test_her2_resolves_and_has_no_genetics_gap():
    result = run_pipeline("HER2")
    assert result.target_resolution.found is True
    assert "cardiac" in result.report.sections[1].html_body.lower()


def test_kras_resolves_via_fallback_and_still_has_genetics_precedent():
    result = run_pipeline("KRAS")
    assert result.target_resolution.found is True
    assert result.target_resolution.resolution_path == "full_text"
    # KRAS genetics precedent exists (embryonic lethal) even though FDA label data is sparse
    assert "lethal" in result.report.sections[1].html_body.lower()
