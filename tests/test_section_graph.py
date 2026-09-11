"""Tests for F5 -- the seven-node section graph with nested interpretation swarm."""

import time

import pytest

from backend.citation_ledger import CitationLedger
from backend.genetics_precedent import get_precedent
from backend.openfda_client import gather_evidence_for_drugs
from backend.regulatory_reasoning import bedrock_claude_is_reachable
from backend.section_graph import SECTION_PROMPTS, _check_dealbreakers, _text_to_safe_html
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


@pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable -- see DECISION_LOG.md 2026-09-11",
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
            rid = ledger.register(
                source_type="label",
                title=f"{item.label.drug_name} label -- Nonclinical Toxicology",
                url=item.label.source_url,
                application_number=item.label.application_number,
            )
            excerpt_lines.append(
                f"[{rid}] ({item.label.drug_name}): "
                + " ".join(item.label.nonclinical_toxicology)[:1200]
            )

    precedent = get_precedent("GLP-1R")
    genetics_ref = ledger.register(
        source_type="curated",
        title="Curated knockout-precedent dataset",
        url="internal://genetics-precedent/GLP-1R",
        curated=True,
    )
    excerpt_lines.append(f"[{genetics_ref}] (curated, GEMOCKT/KURATIERT): {precedent.phenotype_summary}")

    task_context = (
        f"Target: GLP-1R. Resolved drugs: {', '.join(resolution.drug_names[:2])}. "
        f"Cites {len(ledger)} sources.\n\nEvidence excerpts:\n\n" + "\n\n".join(excerpt_lines)
    )

    results, bids = run_section_graph(ledger, task_context=task_context)

    assert set(results.keys()) == set(SECTION_PROMPTS.keys())
    for number, result in results.items():
        assert result.hallucinated_reference_ids == [], f"section {number} hallucinated a reference"
        assert result.dealbreaker_flags == [], f"section {number} tripped a dealbreaker"
        assert len(result.section.html_body) > 20

    assert len(bids) == 4
    print(f"\nFull graph run took {time.time() - t0:.1f}s")
