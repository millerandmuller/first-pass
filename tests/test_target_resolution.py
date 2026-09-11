"""Live tests for F6 -- target resolution.

These hit the real openFDA API deliberately: F6 is the load-bearing hop and a
mocked test would not have caught the original bug (target names are not
searchable at all). openFDA is unauthenticated and rate-limited to 240
req/min, so this suite stays small.
"""

from backend.target_resolution import resolve_target


def test_glp1r_resolves_via_pharm_class_moa():
    result = resolve_target("GLP-1R")
    assert result.found
    assert result.resolution_path == "pharm_class_moa"
    assert result.total_labels >= 40
    assert any("LIRAGLUTIDE" in name.upper() or "SEMAGLUTIDE" in name.upper() for name in result.drug_names)


def test_her2_resolves_via_pharm_class_moa():
    result = resolve_target("HER2")
    assert result.found
    assert result.resolution_path == "pharm_class_moa"
    assert result.total_labels >= 10


def test_kras_falls_through_to_full_text():
    result = resolve_target("KRAS")
    assert result.found
    assert result.resolution_path == "full_text"
    assert result.total_labels >= 5


def test_unknown_target_returns_honest_empty_state():
    result = resolve_target("ZZZ-NOT-A-REAL-TARGET-12345")
    assert not result.found
    assert result.resolution_path == "not_found"
    assert result.drug_names == []
    # every attempted path must be visible in the trail for the empty-state UI
    assert len(result.search_trail) >= 1


def test_search_trail_records_each_attempt():
    result = resolve_target("HER2")
    assert result.search_trail[0].path == "pharm_class_moa"
    assert result.search_trail[0].status in ("live", "cached")
