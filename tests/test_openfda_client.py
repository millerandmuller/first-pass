"""Live tests for F1 -- the two-hop openFDA retrieval layer."""

from backend.openfda_client import (
    fetch_review_document,
    gather_evidence_for_drugs,
    get_approval_history,
    get_label,
)


def test_get_label_returns_nonclinical_toxicology_for_known_drug():
    label = get_label("LIRAGLUTIDE")
    assert label.source_status in ("live", "cached")
    assert label.application_number is not None
    assert len(label.nonclinical_toxicology) > 0


def test_get_label_handles_unknown_drug_without_crashing():
    label = get_label("NOT-A-REAL-DRUG-XYZ")
    assert label.source_status == "zero_matches"
    assert label.nonclinical_toxicology == []


def test_get_approval_history_returns_review_docs():
    approval = get_approval_history("FUTIBATINIB")
    assert approval.application_number is not None
    assert len(approval.review_docs) >= 1
    assert approval.review_docs[0].url.endswith("TOC.html")


def test_hop_two_resolves_toc_to_real_review_pdf_text():
    # NDA 214801 -- futibatinib (Lytgobi) -- verified 2026-09-11 to have a
    # MultidisciplineR.pdf, not a PharmR.pdf.
    toc_url = "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2022/214801Orig1s000TOC.html"
    doc = fetch_review_document(toc_url, application_number="NDA214801")
    assert doc.fetch_status in ("live", "cached")
    assert doc.suffix_used == "MultidisciplineR.pdf"
    assert doc.page_count > 100
    assert any(len(page) > 0 for page in doc.text_by_page)


def test_gather_evidence_caps_deep_review_at_configured_limit():
    evidence = gather_evidence_for_drugs(["LIRAGLUTIDE", "SEMAGLUTIDE"])
    assert len(evidence) == 2
    for item in evidence:
        assert item.label.source_status in ("live", "cached", "zero_matches")
