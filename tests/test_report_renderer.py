"""Tests for F9 -- report renderer, built against real upstream data."""

from backend.citation_ledger import CitationLedger
from backend.genetics_precedent import get_precedent
from backend.grounding_score import highlight_html, score_claim
from backend.openfda_client import get_label
from backend.report_renderer import (
    EvaluationScore,
    InterpretationCard,
    Report,
    ReportSection,
    render_html,
    render_markdown,
)


def _build_real_report() -> Report:
    ledger = CitationLedger()
    label = get_label("LIRAGLUTIDE")
    label_ref = ledger.register(
        source_type="label",
        title="LIRAGLUTIDE label — Nonclinical Toxicology",
        url=label.source_url,
        application_number=label.application_number,
    )
    genetics = get_precedent("GLP-1R")
    genetics_ref = ledger.register(
        source_type="curated",
        title="Curated knockout-precedent dataset",
        url="internal://genetics-precedent/GLP-1R",
        curated=True,
    )

    nonclinical_text = " ".join(label.nonclinical_toxicology)
    claim = "Thyroid C-cell tumors were observed in rodent carcinogenicity studies."
    grounding = score_claim(claim, nonclinical_text)
    highlighted = highlight_html(claim, grounding)

    sections = [
        ReportSection(1, "Target Profile & Biological Function", "<p>GLP-1R is a class B GPCR.</p>"),
        ReportSection(
            2,
            "Genetic & Knockout Precedent",
            f'<p>{genetics.phenotype_summary} <a class="ref-link" href="#ref-{genetics_ref}">[{genetics_ref}]</a></p>',
        ),
        ReportSection(3, "Class Effects & Known Target Toxicities", "<p>Placeholder pending F5.</p>"),
        ReportSection(4, "Regulatory & Clinical Precedents", "<p>Placeholder pending F5.</p>"),
        ReportSection(
            5,
            "Adversity & Evidence Weight Analysis",
            f'<p>{highlighted} <a class="ref-link" href="#ref-{label_ref}">[{label_ref}]</a> '
            f'(grounding {grounding.score:.2f})</p>',
        ),
        ReportSection(6, "Nonclinical Safety Recommendations & CTD-M2.4-Bridge", "<p>Placeholder pending F5.</p>"),
        ReportSection(7, "References & Regulatory Sources", "<p>See bibliography below.</p>"),
    ]

    interpretation_cards = [
        InterpretationCard(
            stance="artifact",
            interpretation_text="Rodent thyroid C-cell tumors are a species-specific mechanism.",
            reference_ids=[label_ref],
            grounding_scores={label_ref: grounding.score},
            confidence_note="Human relevance of the rodent C-cell mechanism remains debated.",
        ),
        InterpretationCard(
            stance="adverse",
            interpretation_text="Thyroid tumor findings should weigh against the safety profile.",
            reference_ids=[label_ref],
            grounding_scores={label_ref: 0.3},
            confidence_note="Weaker grounding: less directly supported by the cited excerpt.",
        ),
    ]

    return Report(
        target="GLP-1R",
        generated_at="2026-09-11T16:00:00Z",
        sections=sections,
        ledger=ledger,
        interpretation_cards=interpretation_cards,
        evaluation_scores=[
            EvaluationScore("Builtin.Faithfulness", 0.91, "live"),
            EvaluationScore("Builtin.Correctness", 0.0, "gap"),
        ],
        data_source_note=None,
    )


def test_render_html_contains_all_seven_sections():
    report = _build_real_report()
    html = render_html(report)
    for section in report.sections:
        escaped_title = section.title.replace("&", "&amp;")
        assert f"{section.number}. {escaped_title}" in html


def test_render_html_marks_strongest_interpretation_card():
    report = _build_real_report()
    html = render_html(report)
    assert 'class="interpretation-card strongest"' in html
    assert "artifact" in html.lower()


def test_render_html_shows_gap_for_missing_evaluator():
    report = _build_real_report()
    html = render_html(report)
    assert "GAP" in html
    assert "Builtin.Correctness" in html


def test_render_html_includes_curated_badge_for_genetics_source():
    report = _build_real_report()
    html = render_html(report)
    assert "GEMOCKT/KURATIERT" in html


def test_render_html_includes_bibliography_with_real_application_number():
    report = _build_real_report()
    html = render_html(report)
    label = get_label("LIRAGLUTIDE")
    assert label.application_number in html


def test_render_markdown_is_well_formed_and_includes_disclaimer():
    report = _build_real_report()
    markdown = render_markdown(report)
    assert "does not substitute for the judgment of a qualified toxicologist" in markdown
    assert "# Target Safety Assessment: GLP-1R" in markdown


def test_render_html_does_not_duplicate_references_section():
    # Section 7 in the brief IS "References & Regulatory Sources" -- the
    # bibliography must render inside it, not as a second appended block.
    # "References & Regulatory Sources" legitimately appears twice (once in
    # the TOC link, once in the section-7 <h2>), same as every other section
    # title -- what must NOT happen is a second, separate bibliography block.
    report = _build_real_report()
    html = render_html(report)
    assert html.count('class="bibliography"') == 1
    assert html.count('id="bibliography"') == 0  # old standalone wrapper must be gone
    assert html.count("<ol") == 2  # TOC <ol> + bibliography <ol>, nothing extra


def test_render_markdown_does_not_duplicate_references_section():
    report = _build_real_report()
    markdown = render_markdown(report)
    assert markdown.count("## 7. References & Regulatory Sources") == 1
    # the ledger's own top-level "## References..." heading must be suppressed
    # when embedded in the numbered section (include_heading=False)
    assert "## References & Regulatory Sources\n" not in markdown


def test_render_html_has_no_unresolved_jinja_or_missing_closing_tags():
    report = _build_real_report()
    html = render_html(report)
    assert "{{" not in html and "}}" not in html
    assert html.count("<section") == html.count("</section>")
    assert html.count("<div") == html.count("</div>")
