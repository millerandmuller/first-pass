"""Tests for F5 -- the seven-node section graph with nested interpretation swarm."""

import time

import pytest

from backend.citation_ledger import CitationLedger
from backend.genetics_precedent import get_precedent
from backend.openfda_client import gather_evidence_for_drugs
from backend.regulatory_reasoning import bedrock_claude_is_reachable
from backend.section_graph import (
    SECTION_PROMPTS,
    _check_dealbreakers,
    _process_node_result,
    _text_to_safe_html,
)
from backend.target_resolution import resolve_target


def test_text_to_safe_html_links_a_valid_citation():
    ledger = CitationLedger()
    rid = ledger.register(source_type="label", title="X", url="https://x/1")
    html_body, hallucinated = _text_to_safe_html(f"A claim [{rid}].", ledger)
    assert f'<a class="ref-link" href="#ref-{rid}">[{rid}]</a>' in html_body
    assert hallucinated == []


def test_text_to_safe_html_flags_hallucinated_reference_id():
    ledger = CitationLedger()
    html_body, hallucinated = _text_to_safe_html("A claim [R-99].", ledger)
    assert hallucinated == ["R-99"]
    assert '<a class="ref-link"' not in html_body  # never rendered as a real link
    assert "[R-99]" in html_body  # visible, not silently dropped


def test_text_to_safe_html_strips_leading_markdown_heading():
    # Regression: a section-writer model included its own markdown heading
    # despite the prompt saying not to (observed live, GLP-1R Section 4 run).
    ledger = CitationLedger()
    text = "# 4. Regulatory & Clinical Precedents\n\nLiraglutide is an approved drug."
    html_body, _ = _text_to_safe_html(text, ledger)
    assert "Regulatory & Clinical Precedents" not in html_body
    assert "Liraglutide is an approved drug." in html_body


def test_text_to_safe_html_escapes_raw_html_in_model_output():
    ledger = CitationLedger()
    html_body, _ = _text_to_safe_html("Contains <script>alert(1)</script> text.", ledger)
    assert "<script>" not in html_body
    assert "&lt;script&gt;" in html_body


def test_check_dealbreakers_flags_acceptance_language_only_in_section_6():
    flagged = _check_dealbreakers(6, "This target is acceptable for further development.")
    assert len(flagged) == 1
    not_checked = _check_dealbreakers(3, "This target is acceptable for further development.")
    assert not_checked == []


def test_check_dealbreakers_flags_explicit_noael_setting():
    flagged = _check_dealbreakers(6, "The NOAEL is 50 mg/kg/day based on this data.")
    assert len(flagged) == 1


def test_check_dealbreakers_silent_on_clean_text():
    assert _check_dealbreakers(6, "Multiple interpretations exist; the debate is summarized.") == []


def test_section_5_result_is_present_with_bids():
    from backend.interpretation_swarm import InterpretationBid
    from backend.section_graph import _section_5_result

    bids = [InterpretationBid(stance="adverse", interpretation_text="x", reference_ids=["R-1"], confidence_note="y")]
    result = _section_5_result(bids)
    assert result.section.number == 5
    assert "Four independent interpretations" in result.section.html_body


def test_section_5_result_is_honest_when_no_bids_were_produced():
    from backend.section_graph import _section_5_result

    result = _section_5_result([])
    assert "No interpretations were produced" in result.section.html_body


def test_score_bids_fills_grounding_scores_from_ledger_excerpt_text():
    from backend.interpretation_swarm import InterpretationBid
    from backend.section_graph import _score_bids

    ledger = CitationLedger()
    rid = ledger.register(
        source_type="label",
        title="X",
        url="https://x/1",
        excerpt_text="Rats given 100 mg/kg showed no adverse effects on fertility.",
    )
    bid = InterpretationBid(
        stance="adverse",
        interpretation_text="Rats given 100 mg/kg showed no adverse effects on fertility.",
        reference_ids=[rid],
        confidence_note="note",
    )
    _score_bids([bid], ledger)
    assert bid.grounding_scores[rid] == 1.0
    # F4: the claim text comes back with grounded/ungrounded spans,
    # not just a bare score -- this was built (grounding_score.highlight_html)
    # but never wired into the real pipeline until this function called it.
    assert '<span class="grounded">Rats</span>' in bid.highlighted_html
    assert "ungrounded" not in bid.highlighted_html  # every word here is grounded


def test_score_bids_skips_sources_with_no_excerpt_text():
    from backend.interpretation_swarm import InterpretationBid
    from backend.section_graph import _score_bids

    ledger = CitationLedger()
    rid = ledger.register(source_type="label", title="X", url="https://x/1")  # no excerpt_text
    bid = InterpretationBid(stance="adverse", interpretation_text="claim", reference_ids=[rid], confidence_note="n")
    _score_bids([bid], ledger)
    assert bid.grounding_scores == {}
    # No citable source text at all still produces highlighted_html -- every
    # word renders "ungrounded", an honest signal rather than a blank field.
    assert '<span class="ungrounded">claim</span>' in bid.highlighted_html


def test_reentry_plan_starts_at_the_first_missing_stance():
    # A fake result missing stances 3 and 4 (adaptive, artifact) -- the
    # swarm's fixed handoff order -- must trigger re-entry starting at
    # stance 3, covering the fixed order through the end from there.
    from backend.interpretation_swarm import InterpretationBid, _reentry_plan

    bids = [
        InterpretationBid(stance="adverse", interpretation_text="x", reference_ids=[], confidence_note="n"),
        InterpretationBid(stance="non_adverse", interpretation_text="y", reference_ids=[], confidence_note="n"),
    ]
    assert _reentry_plan(bids) == ["adaptive", "artifact"]


def test_reentry_plan_is_none_when_all_four_stances_bid():
    from backend.interpretation_swarm import INTERPRETATION_STANCES, InterpretationBid, _reentry_plan

    bids = [
        InterpretationBid(stance=s, interpretation_text="x", reference_ids=[], confidence_note="n")
        for s in INTERPRETATION_STANCES
    ]
    assert _reentry_plan(bids) is None


def test_interpretation_entries_renders_a_gap_card_for_each_missing_stance():
    # Rendering 2 of 4 bids must produce 2 gap cards, never a silently
    # shortened grid and never a fabricated third/fourth bid.
    from backend.report_renderer import InterpretationCard, _interpretation_entries

    cards = [
        InterpretationCard(
            stance="adverse", interpretation_text="x", reference_ids=[], grounding_scores={}, confidence_note="n"
        ),
        InterpretationCard(
            stance="non_adverse", interpretation_text="y", reference_ids=[], grounding_scores={}, confidence_note="n"
        ),
    ]
    entries = _interpretation_entries(cards, missing_stances=["adaptive", "artifact"])

    assert len(entries) == 4
    live = [e for e in entries if e["status"] == "live"]
    gaps = [e for e in entries if e["status"] == "gap"]
    assert {e["stance"] for e in live} == {"adverse", "non_adverse"}
    assert len(gaps) == 2
    assert {e["stance"] for e in gaps} == {"adaptive", "artifact"}
    # A gap entry carries no score/text -- nothing for the frontend to
    # mistake for a real bid.
    assert all("text" not in g and "score" not in g for g in gaps)


def test_process_node_result_marks_content_filtered_without_fabricating_text():
    class FakeFilteredResult:
        stop_reason = "content_filtered"

    ledger = CitationLedger()
    result = _process_node_result(3, "Class Effects & Known Target Toxicities", FakeFilteredResult(), ledger)
    assert result.content_filtered is True
    assert "CONTENT FILTER" in result.section.html_body
    assert result.hallucinated_reference_ids == []


@pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable in this environment",
)
def test_process_node_result_handles_a_real_content_filtered_agent_call():
    # Verified 2026-09-11: this exact phrasing pattern deterministically
    # trips Bedrock's content filter with an empty response. Confirms the
    # real AgentResult shape (not just a fake stand-in) is handled correctly.
    from strands import Agent
    from strands.models import BedrockModel

    from backend.config import AWS_REGION, BEDROCK_MODEL_ID

    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
    agent = Agent(model=model, system_prompt="You are a regulatory toxicology reviewer.")
    agent_result = agent(
        "The antibody binds the HER2 epitope with high affinity in cynomolgus "
        "monkey but not in rat tissue."
    )
    assert agent_result.stop_reason == "content_filtered", (
        "this test's premise (a known content-filter trigger) no longer holds -- "
        "Bedrock's filter behavior may have changed; re-verify before trusting this test"
    )

    ledger = CitationLedger()
    result = _process_node_result(1, "Target Profile & Biological Function", agent_result, ledger)
    assert result.content_filtered is True
    assert "CONTENT FILTER" in result.section.html_body


@pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable in this environment",
)
def test_full_graph_runs_end_to_end_with_no_hallucinations_or_dealbreaker_flags():
    from backend.section_graph import run_section_graph

    t0 = time.time()
    ledger = CitationLedger()
    resolution = resolve_target("GLP-1R")
    evidence = gather_evidence_for_drugs(resolution.drug_names[:2])

    excerpt_lines = []
    for item in evidence:
        if item.label.nonclinical_toxicology:
            excerpt_text = " ".join(item.label.nonclinical_toxicology)[:1200]
            rid = ledger.register(
                source_type="label",
                title=f"{item.label.drug_name} label -- Nonclinical Toxicology",
                url=item.label.source_url,
                application_number=item.label.application_number,
                excerpt_text=excerpt_text,
            )
            excerpt_lines.append(f"[{rid}] ({item.label.drug_name}): " + excerpt_text)

    precedent = get_precedent("GLP-1R")
    genetics_excerpt = precedent.phenotype_summary
    genetics_ref = ledger.register(
        source_type="curated",
        title="Curated knockout-precedent dataset",
        url="internal://genetics-precedent/GLP-1R",
        curated=True,
        excerpt_text=genetics_excerpt,
    )
    excerpt_lines.append(f"[{genetics_ref}] (curated, GEMOCKT/KURATIERT): {genetics_excerpt}")

    task_context = (
        f"Target: GLP-1R. Resolved drugs: {', '.join(resolution.drug_names[:2])}. "
        f"Cites {len(ledger)} sources.\n\nEvidence excerpts:\n\n" + "\n\n".join(excerpt_lines)
    )

    results, bids = run_section_graph(ledger, task_context=task_context)

    # The full 1-7 report section set -- SECTION_PROMPTS alone is {1,2,3,4,6,7}
    # (section 5 is the nested swarm, not its own prompt entry), so checking
    # against SECTION_PROMPTS.keys() would be blind to section 5 going missing
    # entirely, which is exactly the bug this assertion is written to catch.
    assert set(results.keys()) == {1, 2, 3, 4, 5, 6, 7}
    for number, result in results.items():
        assert result.hallucinated_reference_ids == [], f"section {number} hallucinated a reference"
        assert result.dealbreaker_flags == [], f"section {number} tripped a dealbreaker"
        assert len(result.section.html_body) > 20

    assert len(bids) == 4
    assert all(bid.grounding_scores for bid in bids), "every bid should have at least one scored citation"
    print(f"\nFull graph run took {time.time() - t0:.1f}s")
