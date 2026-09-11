"""Live test for F15 -- AgentCore built-in evaluations.

Runs a real Strands Agent, captures its real OTel trace, and submits it to
the real Builtin.Faithfulness evaluator via `aws bedrock-agentcore evaluate`.
No mocks anywhere in this path -- see backend/evaluations.py's docstring for
the non-obvious span-shape requirements this had to reverse-engineer.
"""

import pytest

from backend.evaluations import evaluate_faithfulness, run_with_trace_capture
from backend.regulatory_reasoning import bedrock_claude_is_reachable

pytestmark = pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable -- see DECISION_LOG.md 2026-09-11",
)


def test_faithfulness_evaluation_runs_against_a_real_agent_trace():
    from strands import Agent
    from strands.models import BedrockModel

    from backend.config import AWS_REGION, BEDROCK_MODEL_ID

    def call_agent():
        model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
        agent = Agent(
            model=model,
            system_prompt="Answer using only the provided context.",
        )
        return agent(
            "Context: Rats given 100 mg/kg showed no adverse effects on fertility. "
            "Question: what happened to fertility in rats given 100 mg/kg?"
        )

    _, spans = run_with_trace_capture(call_agent)
    assert len(spans) > 0
    assert any(s.name.startswith("invoke_agent") for s in spans)

    outcome = evaluate_faithfulness(spans)
    assert outcome.status == "live", outcome.error
    assert outcome.value is not None
    assert 0.0 <= outcome.value <= 1.0
    assert outcome.explanation


def test_evaluate_spans_returns_gap_status_for_empty_spans():
    outcome = evaluate_faithfulness([])
    assert outcome.status == "gap"
    assert outcome.error is not None
