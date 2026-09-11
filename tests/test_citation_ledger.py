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


def test_to_markdown_can_omit_heading_for_embedding_in_a_numbered_section():
    ledger = CitationLedger()
    ledger.register(source_type="label", title="X", url="https://x/1")
    with_heading = ledger.to_markdown()
    without_heading = ledger.to_markdown(include_heading=False)
    assert with_heading.startswith("## References & Regulatory Sources")
    assert not without_heading.startswith("##")
    assert "References & Regulatory Sources" not in without_heading


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


def test_to_html_escapes_untrusted_source_fields():
    # title/url/application_number/date ultimately trace back to openFDA
    # label text, not developer-controlled strings, and to_html() renders
    # via `{{ bibliography_html | safe }}` in the report template -- if any
    # of these fields ever contained a stray "<script>" or a quote breaking
    # out of the href attribute, it would render unescaped in the browser.
    ledger = CitationLedger()
    ledger.register(
        source_type="label",
        title='<script>alert(1)</script> "Drug" Label',
        url='https://x/1"><script>alert(2)</script>',
        application_number='<b>NDA1</b>',
        date='<i>2022</i>',
    )
    html_out = ledger.to_html()
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert 'href="https://x/1&quot;&gt;' in html_out
