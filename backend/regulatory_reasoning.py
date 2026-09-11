"""Thin regulatory-reasoning agent, used by F12's dossier regression suite
(and available to F5's Section 6 writer for CTD-M2.4-bridge reasoning).

Not a report-generation component -- it answers a standalone nonclinical
regulatory question with guideline citations. Its correctness on T-01..T-05
from expert_dossier.md is exactly what F12 checks: each of those questions
has a documented naive-wrong answer and a correct answer citing a specific
ICH/FDA guideline section, verified against primary sources during dossier
research (see expert_dossier.md "Testfaelle").
"""

from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel

from backend.config import AWS_REGION, BEDROCK_MODEL_ID

SYSTEM_PROMPT = """You are a regulatory nonclinical toxicology reviewer. Answer the question \
precisely, citing the specific ICH guideline (with section/note number where relevant) or FDA \
guidance that governs the answer. If a naive approach would give a different answer than the \
correct one, briefly say what the naive answer would be and why it is wrong. Do not hedge -- \
give a direct, decisive answer as you would to a colleague, not a disclaimer-laden one."""


def answer_regulatory_question(question: str) -> str:
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)
    result = agent(question)
    return str(result)


def bedrock_claude_is_reachable() -> bool:
    """Live capability probe -- lets the regression suite skip gracefully
    instead of failing noisily when Bedrock Anthropic access is unavailable
    (see DECISION_LOG.md 2026-09-11, 'Anthropic-Use-Case-Formular fehlt')."""
    try:
        model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
        agent = Agent(model=model, system_prompt="Reply with exactly one word.")
        agent("Say hello.")
        return True
    except Exception:
        return False
