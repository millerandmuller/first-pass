"""F12 -- regression suite from the six expert_dossier.md test cases.

Each case documents a naive-wrong answer and a verified-correct one, sourced
during dossier research against primary ICH/FDA guidance. T-01..T-05 test
regulatory reasoning correctness and require a live Bedrock Claude call;
T-06 tests the openFDA two-hop retrieval architecture and needs no LLM (its
full assertions live in test_openfda_client.py -- this file re-asserts the
one fact T-06 is actually about, so "T-01..T-06 automated" is one coherent,
literally-checkable claim rather than five tests plus an unrelated import).

T-01..T-05 auto-skip (not fail) when Bedrock Anthropic access is down --
see backend.regulatory_reasoning.bedrock_claude_is_reachable and
DECISION_LOG.md 2026-09-11 ("Anthropic-Use-Case-Formular fehlt").
"""

import pytest

from backend.openfda_client import get_approval_history
from backend.regulatory_reasoning import answer_regulatory_question, bedrock_claude_is_reachable

requires_bedrock = pytest.mark.skipif(
    not bedrock_claude_is_reachable(),
    reason="Bedrock Anthropic access unavailable -- see DECISION_LOG.md 2026-09-11",
)


@requires_bedrock
def test_t01_mrsd_requires_body_surface_area_conversion_before_safety_factor():
    """NOAEL 60 mg/kg/day in a 13-week rat study -> MRSD for healthy volunteers.

    Naive-wrong: 60/10 = 6 mg/kg. Correct: HED = 60/6.2 (rat km factor) =~9.7
    mg/kg, then /10 safety factor =~0.97 mg/kg (~58 mg absolute). The rat km
    divisor (6.2) must appear -- its absence is exactly the naive-wrong path.
    """
    response = answer_regulatory_question(
        "The NOAEL in the 13-week rat study is 60 mg/kg/day. What is the MRSD "
        "for healthy volunteers (60 kg)?"
    )
    assert "6.2" in response or "6,2" in response
    assert "10" in response  # the safety factor


@requires_bedrock
def test_t02_ich_s9_exempts_advanced_oncology_from_carcinogenicity_study():
    """Kinase inhibitor for advanced cholangiocarcinoma, given past 6 months.

    Naive-wrong: ICH S1A requires the 2-year rat carcinogenicity study past 6
    months of continuous use. Correct: ICH S9 Section 2.7 exempts advanced-
    cancer therapeutics from this requirement at the marketing-application
    stage; S1A section 4.4 already carves out advanced-disease oncology
    itself, so there is no real guideline conflict here.
    """
    response = answer_regulatory_question(
        "Our kinase inhibitor for advanced cholangiocarcinoma is given until "
        "progression, so longer than 6 months. Do we need the 2-year rat "
        "carcinogenicity study for the NDA?"
    )
    assert "S9" in response
    assert "2.7" in response or "advanced" in response.lower()


@requires_bedrock
def test_t03_oncology_phase1_starting_dose_uses_std10_or_hnstd_not_mrsd_chain():
    """Phase 1 starting dose for the same oncology drug in advanced-disease patients.

    Naive-wrong: apply the MRSD chain (NOAEL -> HED -> /10) as in T-01.
    Correct: the MRSD guidance is explicitly scoped to healthy volunteers; ICH
    S9 Note 2 anchors oncology starting dose to 1/10 STD10 (rodent) or 1/6
    HNSTD (non-rodent) instead.
    """
    response = answer_regulatory_question(
        "What is the Phase 1 starting dose for the same oncology drug in "
        "patients with advanced disease?"
    )
    assert "STD10" in response or "HNSTD" in response


@requires_bedrock
def test_t04_non_relevant_species_does_not_count_as_second_species():
    """mAb binds only the cynomolgus epitope, plus a 4-week rat study -- is that two species?

    Naive-wrong: yes, two species are covered. Correct: per ICH S6(R1), the
    rat is a non-relevant species here and does not count as a second
    species -- a single relevant species can suffice with justification.
    """
    response = answer_regulatory_question(
        "Our monoclonal antibody binds the epitope only in cynomolgus. We "
        "also have a 4-week rat study. Does that satisfy a two-species package?"
    )
    assert "relevant species" in response.lower() or "S6" in response


@requires_bedrock
def test_t05_ich_s2r1_does_not_apply_to_biologics():
    """Genotoxicity battery for a monoclonal antibody -- standard ICH S2(R1) battery?

    Naive-wrong: run the standard Ames + in vitro micronucleus + in vivo
    micronucleus battery per ICH S2(R1). Correct: ICH S2(R1) explicitly does
    not apply to biologics; ICH S6(R1) governs instead.
    """
    response = answer_regulatory_question(
        "What genotoxicity battery do we need for our monoclonal antibody -- "
        "Ames plus in vitro micronucleus plus in vivo micronucleus?"
    )
    assert "S6" in response
    assert "S2" in response  # should name S2(R1) as the thing that does NOT apply


def test_t06_review_doc_is_two_hop_not_a_direct_api_field():
    """openFDA never returns review text directly -- only a TOC page URL to
    resolve in a second request. This does not need a live model call: it is
    a retrieval-architecture fact, verified directly against the API.
    """
    approval = get_approval_history("FUTIBATINIB")
    assert len(approval.review_docs) >= 1
    assert approval.review_docs[0].url.endswith("TOC.html"), (
        "openFDA's application_docs URL for a Review doc must be a TOC page, "
        "not the review text itself -- this is the fact T-06 is about."
    )
