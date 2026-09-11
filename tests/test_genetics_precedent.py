"""Tests for F8 -- curated genetics/knockout precedent dataset."""

from backend.genetics_precedent import get_precedent


def test_glp1r_precedent_present_and_viable():
    precedent = get_precedent("GLP-1R")
    assert precedent is not None
    assert precedent.knockout_viability == "viable"
    assert "glucose" in precedent.phenotype_summary.lower()


def test_her2_precedent_present_and_lethal():
    precedent = get_precedent("HER2")
    assert precedent is not None
    assert precedent.knockout_viability == "embryonic lethal"
    assert "cardiac" in precedent.phenotype_summary.lower()


def test_kras_precedent_present_and_lethal():
    precedent = get_precedent("KRAS")
    assert precedent is not None
    assert precedent.knockout_viability == "embryonic lethal"


def test_unknown_target_returns_none_not_a_crash():
    assert get_precedent("SOME-UNLISTED-TARGET") is None


def test_normalization_is_case_and_space_insensitive():
    a = get_precedent("glp-1r")
    b = get_precedent("GLP-1R")
    c = get_precedent(" GLP-1R ")
    assert a == b == c
