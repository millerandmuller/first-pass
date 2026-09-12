"""Tests for the full pipeline orchestrator (F6 through F15), live end-to-end.

One real target run touches: target resolution (F6), two-hop retrieval (F1),
curated genetics (F8), the 7-node section graph with nested 4-agent swarm
(F5+F3), and AgentCore evaluations (F15). No mocks. Each real run costs
~90-120s, so this file keeps the number of full runs small.
"""

import pytest

from backend.pipeline import _join_numbers, _report_notes, run_pipeline
from backend.regulatory_reasoning import bedrock_claude_is_reachable


def test_join_numbers_renders_one_two_or_more_section_numbers_as_words():
    assert _join_numbers([3]) == "3"
    assert _join_numbers([5, 3]) == "3 and 5"
    assert _join_numbers([5, 2, 3]) == "2, 3, and 5"


def test_report_notes_is_empty_when_no_flags_were_raised():
    assert _report_notes([], {}, {}) == []


def test_report_notes_renders_content_filtered_sections_as_words_not_a_python_list():
    assert _report_notes([2], {}, {}) == [
        "Section 2 was blocked by the model provider's content filter."
    ]
    assert _report_notes([3, 5], {}, {}) == [
        "Sections 3 and 5 were blocked by the model provider's content filter."
    ]


def test_report_notes_renders_hallucination_flags_as_words_not_a_python_list():
    assert _report_notes([], {3: ["R-9"]}, {}) == [
        "Unregistered reference IDs were caught and suppressed in section 3."
    ]
    assert _report_notes([], {3: ["R-9"], 5: ["R-12"]}, {}) == [
        "Unregistered reference IDs were caught and suppressed in sections 3 and 5."
    ]


def test_report_notes_produces_the_exact_dealbreaker_note_with_the_matched_phrase_quoted():
    assert _report_notes([], {}, {6: ["is acceptable"]}) == [
        "Section 6 contains wording that reads as setting a NOAEL or judging acceptability "
        "('is acceptable'). This report does not make that call; treat that sentence as "
        "unreviewed model output."
    ]


def test_report_notes_quotes_multiple_dealbreaker_phrases_comma_separated():
    assert _report_notes([], {}, {6: ["is acceptable", "the NOAEL is 50"]}) == [
        "Section 6 contains wording that reads as setting a NOAEL or judging acceptability "
        "('is acceptable', 'the NOAEL is 50'). This report does not make that call; treat "
        "that sentence as unreviewed model output."
    ]


def test_report_notes_combines_all_three_note_kinds_in_order():
    notes = _report_notes([2], {3: ["R-9"]}, {6: ["is acceptable"]})
    assert len(notes) == 3
    assert notes[0].startswith("Section 2 was blocked")
    assert notes[1].startswith("Unregistered reference IDs")
    assert notes[2].startswith("Section 6 contains wording")


def test_unknown_target_produces_honest_empty_report_not_a_crash():
    # No LLM involved -- target resolution fails fast, no graph is built.
    result = run_pipeline("ZZZ-NOT-A-REAL-TARGET-99999")
    assert result.target_resolution.found is False
    assert result.is_partial is False
    assert "No evidence found" in result.report.sections[0].html_body


@pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable in this environment",
)
def test_glp1r_full_pipeline_produces_a_complete_grounded_report():
    events = []
    result = run_pipeline(
        "GLP-1R", on_progress=lambda stage, status, detail: events.append((stage, status))
    )

    assert ("target_resolution", "done") in events
    assert ("evidence_retrieval", "done") in events
    assert ("section_graph", "done") in events
    assert any(stage == "evaluations" for stage, _ in events)

    report = result.report
    assert len(report.sections) == 7
    assert len(report.ledger) > 0
    assert len(report.interpretation_cards) == 4  # the four Section 5 stances
    assert {c.stance for c in report.interpretation_cards} == {
        "adverse",
        "non_adverse",
        "adaptive",
        "artifact",
    }
    # Section 2 (genetics) must use the real curated GLP-1R precedent
    assert "glucose" in report.sections[1].html_body.lower()
    # every reported evaluation score is either a real value or an honest gap
    for ev in report.evaluation_scores:
        assert ev.status in ("live", "gap")
