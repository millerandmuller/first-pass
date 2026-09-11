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
from dataclasses import dataclass, field
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.citation_ledger import CitationLedger
from backend.config import REPORT_SECTIONS

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

DISCLAIMER_PLACEHOLDER = (
    "[PLATZHALTER — Lutfiya formuliert] This report is a decision-support "
    "draft assembled from public regulatory sources. It does not constitute a "
    "nonclinical safety recommendation, does not set a NOAEL, and does not "
    "substitute for expert toxicological judgment."
)


@dataclass
class ReportSection:
    number: int
    title: str
    html_body: str  # citations already rendered as <a href="#ref-R-3">[R-3]</a>


@dataclass
class InterpretationCard:
    stance: str
    interpretation_text: str
    reference_ids: list[str]
    grounding_scores: dict[str, float]
    confidence_note: str
    is_strongest: bool = False

    @property
    def best_grounding_score(self) -> float:
        return max(self.grounding_scores.values()) if self.grounding_scores else 0.0


@dataclass
class EvaluationScore:
    evaluator_id: str
    score: float
    status: str  # "live" | "gap" -- a missing evaluator is a visible gap, never a fabricated value


@dataclass
class Report:
    target: str
    generated_at: str
    sections: list[ReportSection]
    ledger: CitationLedger
    interpretation_cards: list[InterpretationCard] = field(default_factory=list)
    evaluation_scores: list[EvaluationScore] = field(default_factory=list)
    disclaimer_text: str = DISCLAIMER_PLACEHOLDER
    data_source_note: Optional[str] = None  # e.g. "cached demo dataset -- openFDA unreachable"


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
