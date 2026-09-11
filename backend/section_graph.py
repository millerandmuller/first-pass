"""F5 -- Strands Graph: seven section writers, one node per report section.

A real Strands `Graph`, not six independent agent calls: nodes execute in
dependency order determined by edges, and Strands automatically prepends
each dependent node's input with its predecessors' full output (verified
against the framework source, strands/multiagent/graph.py
`_build_node_input` -- "Original Task: ..." followed by "Inputs from
previous nodes: From <dep_id>: ..."). Section 5 is not a separate agent --
it is F3's four-agent interpretation Swarm nested directly as one Graph
node (Strands explicitly supports a MultiAgentBase, e.g. Swarm, as a node),
so the Graph and Swarm are one composed system, not two systems mentioned
in the same README. Section 6 depends on Section 5's node, so the CTD-
bridge writer sees all four interpretation bids before writing -- the
dependency a real interdisciplinary safety writeup would have.

Sections 1-4 and 7 are independent entry points: they all receive the same
rich evidence-context task (target, resolved drug names, citation-tagged
excerpts, genetics precedent) and need nothing from each other.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from strands import Agent
from strands.models import BedrockModel
from strands.multiagent import GraphBuilder

from backend.citation_ledger import CitationLedger
from backend.config import AWS_REGION, BEDROCK_MODEL_ID
from backend.grounding_score import score_claim
from backend.interpretation_swarm import InterpretationBid, build_interpretation_swarm
from backend.report_renderer import ReportSection

_CITE_RE = re.compile(r"\[(R-\d+)\]")

_SHARED_RULES = """Ground every factual claim only in the evidence excerpts given to you in the \
task, each tagged with a reference ID like [R-3]. Cite that exact bracketed ID inline after any \
claim it supports. Never invent a reference ID, and never state a fact not present in the \
excerpts -- if the excerpts do not cover something, say so plainly instead of guessing. Write in \
a regulatory-document register: declarative, unhedged, no chatbot filler ("I think", "it seems"). \
If there is no evidence for a claim, write a clean gap line instead, e.g. "No evidence found. \
Search path: <path>, 0 hits." Output 2-5 short paragraphs of plain text, no markdown headers \
(the report renderer adds the section heading itself)."""

SECTION_PROMPTS: dict[int, tuple[str, str]] = {
    1: (
        "Target Profile & Biological Function",
        f"You write Section 1 (Target Profile & Biological Function) of a Target Safety "
        f"Assessment. Describe the target's biological role, receptor/enzyme class, and the "
        f"approved drug class(es) acting on it, based only on the evidence given. {_SHARED_RULES}",
    ),
    2: (
        "Genetic & Knockout Precedent",
        f"You write Section 2 (Genetic & Knockout Precedent). You will be given a curated "
        f"knockout-phenotype summary for this target (marked GEMOCKT/KURATIERT in the task -- "
        f"say so explicitly in your text, do not present it as an FDA source) plus any real FDA "
        f"label evidence. Explain what the knockout phenotype implies (or does not imply) for "
        f"target safety. {_SHARED_RULES}",
    ),
    3: (
        "Class Effects & Known Target Toxicities",
        f"You write Section 3 (Class Effects & Known Target Toxicities). Synthesize recurring "
        f"toxicity/warning patterns across the approved drugs sharing this target, from the "
        f"label evidence given. Name a pattern only if it appears in at least one excerpt. "
        f"{_SHARED_RULES}",
    ),
    4: (
        "Regulatory & Clinical Precedents",
        f"You write Section 4 (Regulatory & Clinical Precedents). Summarize the approval "
        f"history, application numbers, and any boxed warnings or major safety label language "
        f"for drugs on this target, from the evidence given. {_SHARED_RULES}",
    ),
    6: (
        "Nonclinical Safety Recommendations & CTD-M2.4-Bridge",
        f"You write Section 6 (Nonclinical Safety Recommendations & CTD-M2.4-Bridge). You will "
        f"see all four interpretations from the Section 5 adversity panel (adverse, non-adverse, "
        f"adaptive, artifact) in 'Inputs from previous nodes'. Summarize what further nonclinical "
        f"work or monitoring each interpretation would imply, framed as inputs for a human author "
        f"of CTD Module 2.4 -- structure the open questions, do not resolve them. You must NOT set "
        f"a NOAEL, and you must NOT state that the target/drug class is acceptable or unacceptable "
        f"from a safety standpoint -- that judgment belongs to the human toxicologist, not this "
        f"tool. {_SHARED_RULES}",
    ),
    7: (
        "References & Regulatory Sources",
        "You write a single one-sentence introduction to the bibliography of a Target Safety "
        "Assessment, given the number of sources cited and their types in the task. Do not list "
        "the sources yourself -- that is rendered separately from a citation ledger. Output only "
        "that one sentence, plain text, no citations.",
    ),
}

_ACCEPTANCE_LANGUAGE_PATTERN = re.compile(
    r"\b(is acceptable|is unacceptable|we recommend (accepting|rejecting)|"
    r"the noael (is|was) \d+|safe to proceed|not safe to proceed)\b",
    re.IGNORECASE,
)


@dataclass
class SectionWriteResult:
    section: ReportSection
    hallucinated_reference_ids: list[str]
    dealbreaker_flags: list[str]
    content_filtered: bool = False


CONTENT_FILTERED_NOTE = (
    "[CONTENT FILTER] This section's content was blocked by the model provider's "
    "safety filter before any text was produced. No content is fabricated in its "
    "place -- verified 2026-09-11 that certain legitimate scientific phrasing "
    "(e.g. cross-species epitope-binding language) can trigger this deterministically; "
    "see DECISION_LOG.md. Rephrasing the underlying evidence excerpt is the usual fix."
)


def _text_to_safe_html(text: str, ledger: CitationLedger) -> tuple[str, list[str]]:
    """Escape model text to HTML, turn [R-n] into ref links, and report any
    reference id the model cited that was never registered in the ledger --
    a real hallucination check, not a cosmetic one."""
    hallucinated: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        rid = match.group(1)
        if ledger.get(rid) is None:
            hallucinated.append(rid)
            return html.escape(match.group(0))
        return f'<a class="ref-link" href="#ref-{rid}">[{rid}]</a>'

    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    body_parts = []
    for para in paragraphs:
        # The system prompt tells every writer not to repeat its own section
        # heading (the renderer already adds it) -- verified in practice that
        # a model still does this occasionally, so strip a leading markdown
        # heading line defensively rather than relying on the prompt alone.
        para = re.sub(r"^#{1,6}\s*(?:\d+\.\s*)?.*\n?", "", para, count=1) if para.startswith("#") else para
        if not para.strip():
            continue
        escaped = html.escape(para)
        linked = _CITE_RE.sub(_replace, escaped)
        body_parts.append(f"<p>{linked}</p>")
    return "\n".join(body_parts), hallucinated


def _check_dealbreakers(section_number: int, raw_text: str) -> list[str]:
    if section_number != 6:
        return []
    flags = []
    for match in _ACCEPTANCE_LANGUAGE_PATTERN.finditer(raw_text):
        flags.append(f"possible acceptance/NOAEL-setting language: '{match.group(0)}'")
    return flags


def build_section_graph(ledger: CitationLedger):
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
    builder = GraphBuilder()

    for number, (title, prompt) in SECTION_PROMPTS.items():
        agent = Agent(name=f"section_{number}", model=model, system_prompt=prompt)
        builder.add_node(agent, f"section_{number}")
        builder.set_entry_point(f"section_{number}")

    swarm, bids, agent_order = build_interpretation_swarm(ledger)
    builder.add_node(swarm, "section_5")
    builder.set_entry_point("section_5")

    builder.add_edge("section_5", "section_6")

    return builder.build(), bids


def _process_node_result(
    number: int, title: str, agent_result, ledger: CitationLedger
) -> SectionWriteResult:
    """Turn one graph node's raw AgentResult into a SectionWriteResult.

    Verified 2026-09-11: Bedrock's content filter deterministically blocks
    some legitimate cross-species biology phrasing (e.g. "binds the [X]
    epitope ... in cynomolgus ... tissue") with an empty response and
    stop_reason="content_filtered" -- no exception raised, no account
    Guardrail involved. A silent empty section here would look like a bug
    rather than the honest gap it is.
    """
    if getattr(agent_result, "stop_reason", None) == "content_filtered":
        return SectionWriteResult(
            section=ReportSection(number, title, f"<p>{html.escape(CONTENT_FILTERED_NOTE)}</p>"),
            hallucinated_reference_ids=[],
            dealbreaker_flags=[],
            content_filtered=True,
        )

    raw_text = str(agent_result)
    html_body, hallucinated = _text_to_safe_html(raw_text, ledger)
    dealbreaker_flags = _check_dealbreakers(number, raw_text)
    return SectionWriteResult(
        section=ReportSection(number, title, html_body),
        hallucinated_reference_ids=hallucinated,
        dealbreaker_flags=dealbreaker_flags,
    )


def _score_bids(bids: list[InterpretationBid], ledger: CitationLedger) -> None:
    """Fill in each bid's grounding_scores against the real cited excerpt text.

    build_interpretation_swarm's bids never get scored on their own -- that
    post-processing step only lived in the standalone run_interpretation_swarm
    helper (used for manual verification, not the nested-in-Graph path), so
    every bid from a real pipeline run had an empty grounding_scores dict
    until this was added. Mutates `bids` in place.
    """
    for bid in bids:
        for rid in bid.reference_ids:
            source = ledger.get(rid)
            if source and source.excerpt_text:
                bid.grounding_scores[rid] = score_claim(bid.interpretation_text, source.excerpt_text).score


SECTION_5_TITLE = "Adversity & Evidence Weight Analysis"


def _section_5_result(bids: list[InterpretationBid]) -> SectionWriteResult:
    """Synthesize Section 5's ReportSection from the swarm's bids.

    Section 5 is deliberately absent from SECTION_PROMPTS (its content comes
    from the nested Swarm, not its own Agent node) -- without this, the
    report silently ends up with only 6 of 7 sections, which is exactly what
    happened here before this fix (caught by the full-pipeline integration
    test, not by test_section_graph.py's own check, which compared `results`
    against SECTION_PROMPTS.keys() and so could not see a key SECTION_PROMPTS
    itself never had).
    """
    if not bids:
        body = "<p><strong>No interpretations were produced.</strong> The adversity panel did not return any bids for this run.</p>"
    else:
        body = (
            "<p>Four independent interpretations of the primary finding are presented below, "
            "each citing only the evidence provided. No single interpretation is presented as "
            "this tool's recommendation -- that judgment belongs to the reviewing toxicologist.</p>"
        )
    return SectionWriteResult(
        section=ReportSection(5, SECTION_5_TITLE, body),
        hallucinated_reference_ids=[],
        dealbreaker_flags=[],
    )


def run_section_graph(
    ledger: CitationLedger,
    *,
    task_context: str,
) -> tuple[dict[int, SectionWriteResult], list[InterpretationBid]]:
    """Run the 7-node graph and return per-section results plus the Section 5 bids."""
    graph, bids = build_section_graph(ledger)
    graph_result = graph(task_context)

    results: dict[int, SectionWriteResult] = {}
    for number, (title, _prompt) in SECTION_PROMPTS.items():
        node_result = graph_result.results[f"section_{number}"]
        results[number] = _process_node_result(number, title, node_result.result, ledger)

    _score_bids(bids, ledger)
    results[5] = _section_5_result(bids)

    return results, bids
