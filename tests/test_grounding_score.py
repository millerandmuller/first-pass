"""Tests for F4 -- deterministic grounding score."""

from backend.grounding_score import highlight_html, score_claim, score_claim_against_sources
from backend.openfda_client import get_label


def test_verbatim_claim_scores_perfectly():
    source = "Rats given 100 mg/kg showed no adverse effects on fertility."
    claim = "Rats given 100 mg/kg showed no adverse effects on fertility."
    result = score_claim(claim, source)
    assert result.score == 1.0
    assert result.unmatched_words == []


def test_fabricated_number_lowers_score():
    source = "Rats given 100 mg/kg showed no adverse effects on fertility."
    manipulated_claim = "Rats given 900 mg/kg showed severe adverse effects on fertility."
    result = score_claim(manipulated_claim, source)
    assert result.score < 1.0
    assert "900" in result.unmatched_words
    assert "severe" in result.unmatched_words


def test_completely_unrelated_claim_scores_near_zero():
    source = "Rats given 100 mg/kg showed no adverse effects on fertility."
    claim = "Quarterly revenue increased due to favorable currency exchange rates."
    result = score_claim(claim, source)
    assert result.score < 0.2


def test_empty_claim_scores_zero_without_dividing_by_zero():
    result = score_claim("", "some source text")
    assert result.score == 0.0


def test_case_insensitive_matching():
    source = "NOAEL was established at 50 mg/kg/day in the rat study."
    claim = "The noael was 50 mg/kg/day."
    result = score_claim(claim, source)
    assert result.score == 1.0


def test_best_of_multiple_sources_is_used():
    claim = "Hepatotoxicity was observed at high doses."
    sources = ["Completely unrelated text about manufacturing.", "Hepatotoxicity was observed at high doses in dogs."]
    result = score_claim_against_sources(claim, sources)
    assert result.score == 1.0


def test_highlight_html_wraps_uppercase_tokens_correctly():
    source = "HER2 overexpression was linked to cardiotoxicity in the NOAEL study."
    claim = "HER2 overexpression caused cardiotoxicity per the NOAEL finding."
    result = score_claim(claim, source)
    html = highlight_html(claim, result)
    assert '<span class="grounded">HER2</span>' in html
    assert '<span class="grounded">NOAEL</span>' in html


def test_highlight_html_flags_ungrounded_uppercase_token():
    source = "Liraglutide showed thyroid C-cell tumors in rodents."
    claim = "Liraglutide showed KIDNEY tumors in rodents."
    result = score_claim(claim, source)
    html = highlight_html(claim, result)
    assert '<span class="ungrounded">KIDNEY</span>' in html


def test_highlight_html_escapes_stray_markup_in_claim_text():
    # claim_text is model-generated, not developer-controlled, and the
    # rendered result is used with Jinja's `| safe` filter -- a stray
    # "<script>" in the model's own words must not reach the page unescaped.
    source = "Cardiotoxicity was observed in the trial."
    claim = 'Cardiotoxicity <script>alert(1)</script> was observed & noted.'
    result = score_claim(claim, source)
    html_out = highlight_html(claim, result)
    assert "<script>" not in html_out
    assert "&lt;" in html_out and "&gt;" in html_out  # "script" is itself a word token, so it
    # gets wrapped in its own <span> between the escaped angle brackets -- the safety property
    # that matters is that the ORIGINAL "<script>" tag characters never survive unescaped.
    assert "&amp;" in html_out
    assert '<span class="grounded">Cardiotoxicity</span>' in html_out


def test_grounding_against_real_openfda_label_text():
    label = get_label("LIRAGLUTIDE")
    assert label.nonclinical_toxicology, "expected real nonclinical_toxicology text from openFDA"
    source_text = " ".join(label.nonclinical_toxicology)

    # A claim built from words actually present in the real label text.
    faithful_claim = "thyroid C-cell tumors were observed in rodents"
    faithful_result = score_claim(faithful_claim, source_text)

    manipulated_claim = "hepatic angiosarcoma was observed in nonhuman primates at all doses"
    manipulated_result = score_claim(manipulated_claim, source_text)

    assert faithful_result.score > manipulated_result.score
