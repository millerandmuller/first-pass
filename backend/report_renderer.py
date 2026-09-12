"""F9 -- report renderer, seven sections. One rendering source, not two.

The rendered webpage is the source of truth (the demo's hero visual: the
progress indicator animates through the seven sections, the Beat-6
bibliography scroll only works in a browser). PDF is produced by the
browser's own print-to-PDF against the print stylesheet embedded in the same
template -- no second renderer, no PDF library, no separate template to keep
in sync with the webpage. Markdown export is a separate lightweight text
dump, unrelated to the PDF-vs-webpage question.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.citation_ledger import CitationLedger
from backend.config import REPORT_SECTIONS

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Approved wording (2026-09-11, the reviewing toxicologist who owns this text).
# Pinned: the three pre-computed demo reports carry this exact string, and
# tests/test_report_renderer.py fails if the constant and the cached copies
# ever drift apart. Any change here means regenerating those reports.
DEFAULT_DISCLAIMER = (
    "This report is a decision-support draft assembled from public regulatory "
    "sources -- openFDA labels, approval records, and FDA review documents. It "
    "does not constitute a nonclinical safety recommendation, does not set a "
    "NOAEL, and does not substitute for the judgment of a qualified "
    "toxicologist. Every claim carries a citation; verifying and weighing that "
    "evidence remains the reviewing scientist's responsibility."
)


@dataclass
class ReportSection:
    number: int
    title: str
    html_body: str  # citations already rendered as <a href="#ref-R-3">[R-3]</a>
    opening_html: str = ""  # first paragraph of html_body, citation anchors preserved -- v4 UI excerpt panel


@dataclass
class GroundedExample:
    """The v4 UI's section-03 provenance callout: one real cited claim from
    the report body, shown beside the source excerpt it matched. Populated
    only when section 3 actually cites a source with retained excerpt text
    and the claim scores above zero against it -- see
    backend/section_graph.py::_build_grounded_example. Never synthesized."""

    claim_html: str
    source_excerpt_html: str
    source_label: str
    overlap_score: float


@dataclass
class InterpretationCard:
    stance: str
    interpretation_text: str
    reference_ids: list[str]
    grounding_scores: dict[str, float]
    confidence_note: str
    is_strongest: bool = False
    highlighted_html: str = ""  # F4: interpretation_text with grounded/ungrounded spans marked

    @property
    def best_grounding_score(self) -> float:
        return max(self.grounding_scores.values()) if self.grounding_scores else 0.0


@dataclass
class EvaluationScore:
    evaluator_id: str
    score: float
    status: str  # "live" | "gap" -- a missing evaluator is a visible gap, never a fabricated value
    explanation: str = ""  # LLM-judge explanation for "live", error text for "gap"


@dataclass
class Report:
    target: str
    generated_at: str
    sections: list[ReportSection]
    ledger: CitationLedger
    interpretation_cards: list[InterpretationCard] = field(default_factory=list)
    evaluation_scores: list[EvaluationScore] = field(default_factory=list)
    disclaimer_text: str = DEFAULT_DISCLAIMER
    data_source_note: Optional[str] = None  # e.g. "cached demo dataset -- openFDA unreachable"
    resolution: Optional[object] = None  # TargetResolution -- kept for the v4 UI's structured payload
    grounded_example: Optional[GroundedExample] = None
    missing_interpretation_stances: list[str] = field(default_factory=list)  # stances with no bid after re-entry

    def __post_init__(self) -> None:
        for section in self.sections:
            if not section.opening_html:
                section.opening_html = first_paragraph_html(section.html_body)


_PARAGRAPH_RE = re.compile(r"<p[^>]*>.*?</p>", re.DOTALL)


def first_paragraph_html(html_body: str) -> str:
    """The opening of a rendered section body: its first <p>...</p> block,
    citation anchors intact. Used by the report excerpt panel, which shows
    only the opening of each section rather than the full document. Falls
    back to the whole body when it is not paragraph-wrapped (e.g. a single
    gap/notice line)."""
    match = _PARAGRAPH_RE.search(html_body)
    return match.group(0) if match else html_body


def _mark_strongest(cards: list[InterpretationCard]) -> list[InterpretationCard]:
    if not cards:
        return cards
    best_score = max(c.best_grounding_score for c in cards)
    for card in cards:
        card.is_strongest = card.best_grounding_score == best_score and best_score > 0
    return cards


def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "jinja"]),
    )


def render_html(report: Report) -> str:
    report.interpretation_cards = _mark_strongest(report.interpretation_cards)
    env = _make_env()
    template = env.get_template("report.html.jinja")
    return template.render(
        report=report,
        bibliography_html=report.ledger.to_html(),
        section_titles=REPORT_SECTIONS,
    )


def render_markdown(report: Report) -> str:
    lines = [f"# Target Safety Assessment: {report.target}", "", f"*Generated {report.generated_at}*", ""]
    if report.data_source_note:
        lines.append(f"> **Data source note:** {report.data_source_note}")
        lines.append("")
    if report.evaluation_scores:
        lines.append("## Evaluation Scores")
        for ev in report.evaluation_scores:
            value = f"{ev.score:.2f}" if ev.status == "live" else "GAP -- evaluator unavailable"
            lines.append(f"- **{ev.evaluator_id}:** {value}")
        lines.append("")

    for section in report.sections:
        lines.append(f"## {section.number}. {section.title}")
        lines.append("")
        # Section 7 IS the bibliography -- rendered from the ledger, not
        # authored content, so it is never duplicated with a second
        # "references" block later in the document.
        if section.number == 7:
            lines.append(report.ledger.to_markdown(include_heading=False))
        else:
            lines.append(section.html_body)
        lines.append("")

        if section.number == 5 and report.interpretation_cards:
            lines.append("### Adversity & Evidence Weight -- All Interpretations")
            for card in report.interpretation_cards:
                marker = " **(strongest grounding)**" if card.is_strongest else ""
                lines.append(f"- **{card.stance}**{marker}: {card.interpretation_text}")
                lines.append(
                    f"  - refs: {', '.join(card.reference_ids)}; confidence note: {card.confidence_note}"
                )
            lines.append("")

    lines.append(f"---\n\n{report.disclaimer_text}")
    return "\n".join(lines)


def _reference_text(source) -> str:
    parts = [source.title]
    if source.application_number:
        parts.append(f"({source.application_number})")
    if source.date:
        parts.append(f"— {source.date}")
    parts.append(f"— {source.url}")
    return " ".join(parts)


def _interpretation_entries(
    cards: list[InterpretationCard], missing_stances: list[str]
) -> list[dict]:
    """One JSON entry per real bid, plus one honest gap entry per stance the
    swarm never produced a bid for (even after its one re-entry retry --
    see backend/interpretation_swarm.py). Never silently drops a missing
    stance from the document, never fabricates a bid to fill it."""
    entries = [
        {
            "stance": c.stance,
            "status": "live",
            "text": c.highlighted_html or c.interpretation_text,
            "score": c.best_grounding_score,
            "is_strongest": c.is_strongest,
        }
        for c in cards
    ]
    entries.extend({"stance": s, "status": "gap"} for s in missing_stances)
    return entries


def report_to_dict(
    report: Report,
    *,
    served: str,  # "cached" | "live"
    run_label: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    nonclinical_label_count: Optional[int] = None,
) -> dict:
    """Serialize `report` into the UI's structured data contract, alongside
    `render_html`'s full HTML string -- additive, never a replacement. The
    full-document path, the print stylesheet, and the disclaimer drift test
    all still depend on `render_html`; this is the second, new payload the
    frontend binds directly instead of parsing HTML.

    `nonclinical_label_count` is counted among the labels actually retrieved
    for evidence (capped by MAX_DRUGS_FOR_DEEP_REVIEW's sibling resolution
    cap), not the full open-ended class total in `resolver.total_labels` --
    the two numbers answer different questions. Pass None when it was not
    computed and the frontend shows total_labels alone.
    """
    report.interpretation_cards = _mark_strongest(report.interpretation_cards)

    resolution = report.resolution
    resolver = None
    if resolution is not None:
        resolver = {
            "path": resolution.resolution_path,
            "matched_phrase": resolution.matched_phrase,
            "total_labels": resolution.total_labels,
            "nonclinical_label_count": nonclinical_label_count,
            "search_trail": [
                {"path": a.path, "phrase": a.phrase, "status": a.status, "total": a.total}
                for a in resolution.search_trail
            ],
        }

    payload: dict = {
        "target": report.target,
        "generated_at": report.generated_at,
        "served": served,
        "resolver": resolver,
        "sections": [
            {
                "number": s.number,
                "title": s.title,
                "html_body": s.html_body,
                "opening_html": s.opening_html,
            }
            for s in report.sections
        ],
        "interpretations": _interpretation_entries(
            report.interpretation_cards, report.missing_interpretation_stances
        ),
        "evaluations": [
            {
                "evaluator_id": e.evaluator_id,
                "score": e.score,
                "status": e.status,
                "explanation": e.explanation,
            }
            for e in report.evaluation_scores
        ],
        "references": [
            {
                "tag": f"[{s.reference_id}]",
                "text": _reference_text(s),
                "url": s.url,
                "kind": "curated" if s.curated else "retrieved",
            }
            for s in report.ledger.bibliography()
        ],
        "grounded_example": (
            {
                "claim_html": report.grounded_example.claim_html,
                "source_excerpt_html": report.grounded_example.source_excerpt_html,
                "source_label": report.grounded_example.source_label,
                "overlap_score": report.grounded_example.overlap_score,
            }
            if report.grounded_example
            else None
        ),
        "disclaimer_text": report.disclaimer_text,
    }
    if served == "cached":
        payload["run_label"] = run_label
        payload["duration_seconds"] = duration_seconds
    return payload
