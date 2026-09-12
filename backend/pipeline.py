"""Orchestrates one target -> report run, emitting progress events as it goes.

Multi-agent hooks into progress (F5/F10): each pipeline stage calls
`on_progress` with a stage name and status before/after doing its work, which
the API layer forwards as Server-Sent Events to the input UI's progress
indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from backend.citation_ledger import CitationLedger
from backend.evaluations import evaluate_correctness, evaluate_faithfulness, run_with_trace_capture
from backend.genetics_precedent import get_precedent
from backend.interpretation_swarm import missing_stances
from backend.openfda_client import DrugEvidence, gather_evidence_for_drugs
from backend.report_renderer import EvaluationScore, InterpretationCard, Report, ReportSection
from backend.section_graph import build_grounded_example, run_section_graph
from backend.target_resolution import resolve_target

ProgressCallback = Callable[[str, str, Optional[str]], None]  # (stage, status, detail)


def _noop_progress(stage: str, status: str, detail: Optional[str] = None) -> None:
    return None


@dataclass
class PipelineResult:
    report: Report
    target_resolution: object
    is_partial: bool  # True if any section hit a content filter or evidence gap
    nonclinical_label_count: Optional[int] = None  # among labels actually retrieved, not the full class total


def _empty_target_report(target: str, resolution, ledger: CitationLedger) -> Report:
    return Report(
        target=target,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sections=[
            ReportSection(
                1,
                "Target Profile & Biological Function",
                f"<p><strong>No evidence found.</strong> Search path: "
                f"{[a.path for a in resolution.search_trail]}, 0 hits, "
                f"retrieved {resolution.searched_at}.</p>",
            )
        ],
        ledger=ledger,
        data_source_note="No drug names resolved for this target -- see Section 1 for the attempted search path.",
        resolution=resolution,
    )


def _pick_primary_finding(evidence: list[DrugEvidence]) -> tuple[str, str]:
    """Choose the single finding Section 5's panel debates.

    Prefers a boxed warning (the FDA's own signal of a serious, often-
    contested risk) over a general nonclinical_toxicology excerpt, so all
    four interpretation agents converge on debating the same real finding
    instead of four independently-chosen ones.
    """
    for item in evidence:
        if item.label.boxed_warning:
            return "boxed_warning", " ".join(item.label.boxed_warning)[:1500]
    for item in evidence:
        if item.label.nonclinical_toxicology:
            return "nonclinical_toxicology", " ".join(item.label.nonclinical_toxicology)[:1500]
    return "none", ""


def run_pipeline(target: str, on_progress: ProgressCallback = _noop_progress) -> PipelineResult:
    ledger = CitationLedger()

    on_progress("target_resolution", "running", None)
    resolution = resolve_target(target)
    if not resolution.found:
        on_progress("target_resolution", "empty", f"no drug names resolved for '{target}'")
        return PipelineResult(
            report=_empty_target_report(target, resolution, ledger),
            target_resolution=resolution,
            is_partial=False,
        )
    on_progress("target_resolution", "done", f"{len(resolution.drug_names)} drug name(s) resolved")

    on_progress("evidence_retrieval", "running", None)
    evidence = gather_evidence_for_drugs(resolution.drug_names)
    # Counted among labels actually retrieved here (capped by resolution's
    # drug-name limit), not the open-ended class total in resolution.total_labels
    # -- the two numbers answer different questions, see report_renderer.report_to_dict.
    nonclinical_label_count = sum(1 for item in evidence if item.label.nonclinical_toxicology)
    excerpt_lines: list[str] = []
    for item in evidence:
        if item.label.source_status in ("live", "cached") and item.label.nonclinical_toxicology:
            excerpt_text = " ".join(item.label.nonclinical_toxicology)[:1500]
            ref = ledger.register(
                source_type="label",
                title=f"{item.label.drug_name} label -- Nonclinical Toxicology",
                url=item.label.source_url,
                application_number=item.label.application_number,
                excerpt_text=excerpt_text,
            )
            excerpt_lines.append(f"[{ref}] ({item.label.drug_name}): " + excerpt_text)
    on_progress("evidence_retrieval", "done", f"{len(excerpt_lines)} label source(s) registered")

    on_progress("genetics_precedent", "running", None)
    precedent = get_precedent(target)
    if precedent:
        genetics_excerpt = f"{precedent.phenotype_summary} {precedent.relevance_to_target_safety}"
        genetics_ref = ledger.register(
            source_type="curated",
            title="Curated knockout-precedent dataset",
            url=f"internal://genetics-precedent/{target}",
            curated=True,
            excerpt_text=genetics_excerpt,
        )
        excerpt_lines.append(f"[{genetics_ref}] (curated knockout precedent, GEMOCKT/KURATIERT): {genetics_excerpt}")
    on_progress("genetics_precedent", "done" if precedent else "empty", None)

    finding_kind, finding_text = _pick_primary_finding(evidence)

    task_context = (
        f"Target: {target}. Resolved approved drugs: {', '.join(resolution.drug_names)}. "
        f"This report cites {len(ledger)} sources total.\n\n"
        f"Evidence excerpts (cite only these reference IDs):\n\n" + "\n\n".join(excerpt_lines)
    )
    if finding_text:
        task_context += (
            f"\n\nPRIMARY FINDING FOR SECTION 5's ADVERSITY PANEL (all four interpretation "
            f"agents must build their case about this specific finding, drawn from a "
            f"{finding_kind.replace('_', ' ')}): {finding_text}"
        )

    on_progress("section_graph", "running", "7-node graph + nested 4-agent swarm")
    (report_and_bids, spans) = run_with_trace_capture(
        lambda: run_section_graph(ledger, task_context=task_context, on_progress=on_progress)
    )
    section_results, bids = report_and_bids
    on_progress("section_graph", "done", f"{len(section_results)} sections written")

    is_partial = any(r.content_filtered or r.hallucinated_reference_ids for r in section_results.values())

    # Section-03 provenance callout: one real cited claim from Section 3's
    # own text, paired with the source excerpt it matched. None when no
    # citation there scores.
    grounded_example = None
    if 3 in section_results:
        grounded_example = build_grounded_example(section_results[3].section.html_body, ledger)

    on_progress("evaluations", "running", None)
    faithfulness = evaluate_faithfulness(spans)
    correctness = evaluate_correctness(spans)
    on_progress(
        "evaluations",
        "done" if faithfulness.status == "live" else "gap",
        f"faithfulness={faithfulness.value}" if faithfulness.status == "live" else faithfulness.error,
    )

    interpretation_cards = [
        InterpretationCard(
            stance=bid.stance,
            interpretation_text=bid.interpretation_text,
            reference_ids=bid.reference_ids,
            grounding_scores=bid.grounding_scores,
            confidence_note=bid.confidence_note,
            highlighted_html=bid.highlighted_html,
        )
        for bid in bids
    ]

    sections = [section_results[n].section for n in sorted(section_results)]
    content_filtered_sections = [n for n, r in section_results.items() if r.content_filtered]
    hallucination_flags = {
        n: r.hallucinated_reference_ids for n, r in section_results.items() if r.hallucinated_reference_ids
    }

    notes = []
    if content_filtered_sections:
        notes.append(f"Sections {content_filtered_sections} were blocked by the model provider's content filter.")
    if hallucination_flags:
        notes.append(f"Unregistered reference IDs were caught and suppressed in sections {list(hallucination_flags)}.")

    report = Report(
        target=target,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sections=sections,
        ledger=ledger,
        interpretation_cards=interpretation_cards,
        evaluation_scores=[
            EvaluationScore(
                "Builtin.Faithfulness",
                faithfulness.value or 0.0,
                faithfulness.status,
                explanation=faithfulness.explanation or faithfulness.error or "",
            ),
            EvaluationScore(
                "Builtin.Correctness",
                correctness.value or 0.0,
                correctness.status,
                explanation=correctness.explanation or correctness.error or "",
            ),
        ],
        data_source_note=" ".join(notes) if notes else None,
        resolution=resolution,
        grounded_example=grounded_example,
        missing_interpretation_stances=missing_stances(bids),
    )
    on_progress("report_render", "done", None)
    return PipelineResult(
        report=report,
        target_resolution=resolution,
        is_partial=is_partial,
        nonclinical_label_count=nonclinical_label_count,
    )
