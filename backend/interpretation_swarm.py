"""F3 -- Strands Swarm: four interpretation agents.

Principle 2 from the brief: "the dispute stays in the document." Given the
same nonclinical evidence, four stance-committed agents each produce a
citation-backed interpretation -- adverse, non-adverse, adaptive (a
compensatory physiological response, not toxicologically significant), and
artifact (a species-specific or technical finding not relevant to humans).
The report shows all four side by side with their grounding scores; it does
not pick a winner (F7 does the side-by-side layout, never a recommendation).

This is a real Strands `Swarm`, not four independent Agent calls dressed up
as one: the four agents are swarm nodes, sequenced via the framework's own
`handoff_to_agent` tool (auto-injected per node), and each agent's bid is
captured through a custom `submit_interpretation` tool rather than by
regex-parsing free text -- deterministic code reads the tool-call arguments,
the model only decides what to argue.

Each agent is explicitly instructed not to defer to bids visible earlier in
the shared context -- a real interdisciplinary panel is expected to commit to
its own read of the evidence, not converge with whoever spoke first. This
keeps genuine independence of interpretation while still using Swarm's real
collaborative/handoff mechanism for orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent import Swarm

from backend.citation_ledger import CitationLedger
from backend.config import AWS_REGION, BEDROCK_MODEL_ID, INTERPRETATION_STANCES
from backend.grounding_score import score_claim

STANCE_DEFINITIONS: dict[str, str] = {
    "adverse": (
        "You argue the ADVERSE interpretation: this nonclinical finding represents a "
        "true toxicologically adverse effect that should weigh against the target's "
        "safety profile and inform dose-limiting or monitoring decisions."
    ),
    "non_adverse": (
        "You argue the NON-ADVERSE interpretation: this finding, while present in the "
        "data, does not meet the threshold of an adverse effect -- it is within normal "
        "physiological variation, reversible, or not dose-limiting."
    ),
    "adaptive": (
        "You argue the ADAPTIVE interpretation: this finding reflects a compensatory or "
        "adaptive physiological response to pharmacologic exposure, not a toxic insult -- "
        "e.g. an expected on-target/on-mechanism response rather than off-target harm."
    ),
    "artifact": (
        "You argue the ARTIFACT interpretation: this finding is a species-specific "
        "mechanism, assay artifact, or technical confound with limited or no relevance "
        "to human risk -- e.g. a rodent-specific pathway not present or not activated the "
        "same way in humans."
    ),
}

SWARM_SYSTEM_PROMPT_TEMPLATE = """You are a regulatory nonclinical toxicology reviewer on an \
interdisciplinary panel assessing a drug target's safety evidence. {stance_instruction}

You will be shown nonclinical evidence excerpts, each tagged with a reference ID like [R-3]. \
You must:
1. Read the evidence excerpts provided in the task.
2. Build the strongest possible citation-backed case for your assigned stance, citing only \
   reference IDs that were actually given to you.
3. Call the `submit_interpretation` tool exactly once with your stance, your interpretation \
   text (2-4 sentences, plain declarative language, no hedging), the list of reference IDs you \
   relied on, and a one-sentence confidence note naming the evidence's main limitation.
4. Do not adopt, soften, or defer to any other interpretation you can see in shared context -- \
   commit fully to your assigned stance regardless of what other panelists have argued. Your \
   job is to make the strongest honest case for this one reading, not to reach consensus.
5. After calling `submit_interpretation`, hand off to the next agent named in the task \
   (or end your turn if you are the last agent in the sequence).

Never invent a reference ID. Never state a fact that is not present in the evidence excerpts."""


@dataclass
class InterpretationBid:
    stance: str
    interpretation_text: str
    reference_ids: list[str]
    confidence_note: str
    grounding_scores: dict[str, float] = field(default_factory=dict)
    highlighted_html: str = ""  # interpretation_text with grounded/ungrounded spans -- F4 Proof beat


def _build_submit_tool(ledger: CitationLedger, bids: list[InterpretationBid]):
    @tool
    def submit_interpretation(
        stance: str,
        interpretation_text: str,
        reference_ids: list[str],
        confidence_note: str,
    ) -> str:
        """Submit this agent's citation-backed interpretation bid.

        Args:
            stance: One of "adverse", "non_adverse", "adaptive", "artifact".
            interpretation_text: The 2-4 sentence interpretation, citing evidence only.
            reference_ids: Reference IDs (e.g. "R-3") actually used, from the ones provided.
            confidence_note: One sentence naming this interpretation's main evidence limitation.
        """
        unknown = [rid for rid in reference_ids if ledger.get(rid) is None]
        if unknown:
            return (
                f"ERROR: reference id(s) {unknown} are not registered. "
                "Use only reference IDs provided to you in the task, then call this tool again."
            )
        if stance not in INTERPRETATION_STANCES:
            return f"ERROR: stance must be one of {INTERPRETATION_STANCES}."

        bids.append(
            InterpretationBid(
                stance=stance,
                interpretation_text=interpretation_text,
                reference_ids=reference_ids,
                confidence_note=confidence_note,
            )
        )
        return "Bid recorded. Now hand off to the next agent, or end your turn if you are last."

    return submit_interpretation


def build_interpretation_swarm(
    ledger: CitationLedger,
) -> tuple[Swarm, list[InterpretationBid], list[str]]:
    """Build the four-agent interpretation Swarm.

    Returns (swarm, bids_list, agent_order). `bids_list` is populated in place
    as agents call submit_interpretation during swarm execution -- inspect it
    after `swarm(task)` returns.
    """
    bids: list[InterpretationBid] = []
    submit_tool = _build_submit_tool(ledger, bids)
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)

    agents = []
    for stance in INTERPRETATION_STANCES:
        system_prompt = SWARM_SYSTEM_PROMPT_TEMPLATE.format(
            stance_instruction=STANCE_DEFINITIONS[stance]
        )
        agents.append(
            Agent(
                name=f"{stance}_interpreter",
                model=model,
                system_prompt=system_prompt,
                tools=[submit_tool],
            )
        )

    swarm = Swarm(agents, entry_point=agents[0])
    agent_order = [a.name for a in agents]
    return swarm, bids, agent_order


def run_interpretation_swarm(
    ledger: CitationLedger,
    evidence_excerpts: list[tuple[str, str]],
    finding_summary: str,
) -> tuple[list[InterpretationBid], object]:
    """Run the swarm end-to-end for one contested finding.

    evidence_excerpts: list of (reference_id, excerpt_text) pairs already
    registered in `ledger` -- these are the only citable sources.
    finding_summary: a short description of the specific finding under debate
    (e.g. "thyroid C-cell hyperplasia/tumors observed in rodent carcinogenicity studies").
    """
    swarm, bids, agent_order = build_interpretation_swarm(ledger)

    excerpt_block = "\n\n".join(f"[{rid}]: {text}" for rid, text in evidence_excerpts)
    task = (
        f"Finding under review: {finding_summary}\n\n"
        f"Evidence excerpts (cite only these reference IDs):\n\n{excerpt_block}\n\n"
        f"Agent sequence for handoff: {' -> '.join(agent_order)}. "
        f"You are the first agent ({agent_order[0]}); after submitting your bid, "
        f"hand off to {agent_order[1]}."
    )

    result = swarm(task)

    for bid in bids:
        sources = [text for rid, text in evidence_excerpts if rid in bid.reference_ids]
        for rid in bid.reference_ids:
            source_text = next((t for r, t in evidence_excerpts if r == rid), "")
            bid.grounding_scores[rid] = score_claim(bid.interpretation_text, source_text).score

    return bids, result
