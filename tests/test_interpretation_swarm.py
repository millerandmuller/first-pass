"""Live test for F3 -- the four-agent Strands Swarm.

Runs the real swarm against real GLP-1R label evidence (the demo's Beat-3
hero finding: rodent thyroid C-cell tumors). Requires live Bedrock Anthropic
access -- skips cleanly if unavailable rather than failing noisily.
"""

import pytest

from strands.multiagent.base import Status

from backend.citation_ledger import CitationLedger
from backend.config import INTERPRETATION_STANCES
from backend.interpretation_swarm import run_interpretation_swarm
from backend.openfda_client import get_label
from backend.regulatory_reasoning import bedrock_claude_is_reachable

pytestmark = pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable in this environment",
)


def test_swarm_produces_four_distinct_stance_bids_grounded_in_real_evidence():
    label = get_label("LIRAGLUTIDE")
    excerpt_text = " ".join(label.nonclinical_toxicology)[:3000]

    ledger = CitationLedger()
    rid = ledger.register(
        source_type="label",
        title="LIRAGLUTIDE label -- Nonclinical Toxicology section",
        url=label.source_url,
        application_number=label.application_number,
    )

    bids, result = run_interpretation_swarm(
        ledger,
        evidence_excerpts=[(rid, excerpt_text)],
        finding_summary=(
            "Thyroid C-cell hyperplasia and tumors observed in rodent "
            "carcinogenicity studies of GLP-1 receptor agonists."
        ),
    )

    assert result.status == Status.COMPLETED
    assert result.execution_count == 4
    assert [n.node_id for n in result.node_history] == [
        f"{s}_interpreter" for s in INTERPRETATION_STANCES
    ]

    assert len(bids) == 4
    assert {b.stance for b in bids} == set(INTERPRETATION_STANCES)

    for bid in bids:
        assert bid.reference_ids == [rid]  # only the provided reference id was cited
        assert bid.grounding_scores[rid] > 0.3  # real evidence-grounded text, not boilerplate
        assert len(bid.interpretation_text) > 50


def test_swarm_rejects_an_unregistered_reference_id_via_the_tool():
    # Sanity check on the guardrail itself, not a live model behavior --
    # calls the tool function directly rather than routing through an agent.
    from backend.interpretation_swarm import _build_submit_tool

    ledger = CitationLedger()
    bids = []
    tool_fn = _build_submit_tool(ledger, bids)  # a strands DecoratedFunctionTool, directly callable
    result = tool_fn(
        stance="adverse",
        interpretation_text="Some claim.",
        reference_ids=["R-999"],
        confidence_note="note",
    )
    assert "ERROR" in result
    assert bids == []
