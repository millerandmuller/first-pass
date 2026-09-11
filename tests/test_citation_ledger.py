"""Tests for F2 -- citation ledger and bibliography."""

import pytest

from backend.citation_ledger import CitationLedger


def test_register_returns_sequential_ids():
    ledger = CitationLedger()
    r1 = ledger.register(source_type="label", title="LIRAGLUTIDE Label", url="https://x/1")
    r2 = ledger.register(source_type="label", title="SEMAGLUTIDE Label", url="https://x/2")
    assert r1 == "R-1"
    assert r2 == "R-2"


def test_registering_the_same_source_twice_returns_the_same_id():
    ledger = CitationLedger()
    r1 = ledger.register(source_type="label", title="LIRAGLUTIDE Label", url="https://x/1")
    r2 = ledger.register(source_type="label", title="LIRAGLUTIDE Label", url="https://x/1")
    assert r1 == r2
    assert len(ledger) == 1


def test_different_source_types_same_url_are_distinct():
    ledger = CitationLedger()
    r1 = ledger.register(source_type="label", title="X", url="https://x/1")
    r2 = ledger.register(source_type="review_pdf", title="X review", url="https://x/1")
    assert r1 != r2


def test_cite_requires_registered_id():
    ledger = CitationLedger()
    with pytest.raises(KeyError):
        ledger.cite("R-99")


def test_bibliography_preserves_registration_order():
    ledger = CitationLedger()
    ledger.register(source_type="label", title="B Drug", url="https://x/b")
    ledger.register(source_type="label", title="A Drug", url="https://x/a")
    titles = [s.title for s in ledger.bibliography()]
    assert titles == ["B Drug", "A Drug"]


def test_curated_source_is_marked_in_markdown_and_html():
    ledger = CitationLedger()
    ledger.register(
        source_type="curated",
        title="Curated Knockout Precedent Dataset",
        url="internal://genetics-precedent",
        curated=True,
    )
    markdown = ledger.to_markdown()
    html = ledger.to_html()
    assert "GEMOCKT/KURATIERT" in markdown
    assert "GEMOCKT/KURATIERT" in html


def test_application_number_and_date_appear_in_bibliography_entry():
    ledger = CitationLedger()
    ledger.register(
        source_type="review_pdf",
        title="Multi-disciplinary Review",
        url="https://x/review.pdf",
        application_number="NDA214801",
        date="2022-11-08",
    )
    markdown = ledger.to_markdown()
    assert "NDA214801" in markdown
    assert "2022-11-08" in markdown
