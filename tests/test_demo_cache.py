"""Tests for F11 -- demo target cache."""

import shutil

import pytest

from backend import demo_cache


@pytest.fixture(autouse=True)
def _clean_cache_dir():
    yield
    if __import__("os").path.exists(demo_cache._CACHE_DIR):
        shutil.rmtree(demo_cache._CACHE_DIR)


def test_is_demo_target_matches_case_and_space_insensitively():
    assert demo_cache.is_demo_target("glp-1r")
    assert demo_cache.is_demo_target("HER2")
    assert demo_cache.is_demo_target(" kras ")


def test_is_demo_target_false_for_unrelated_target():
    assert not demo_cache.is_demo_target("EGFR")


def test_load_returns_none_when_nothing_cached():
    assert demo_cache.load_cached_report("HER2") is None


def test_load_returns_none_for_a_non_demo_target_even_if_files_existed():
    # is_demo_target gates the lookup entirely -- a non-demo target is never
    # served from cache even if a same-named file happened to exist on disk.
    demo_cache.save_cached_report("EGFR", "<p>x</p>", 42.0, "dry-run-1")
    assert demo_cache.load_cached_report("EGFR") is None


def test_save_then_load_roundtrips():
    demo_cache.save_cached_report("HER2", "<p>real report html</p>", 123.4, "dry-run-2")
    cached = demo_cache.load_cached_report("HER2")
    assert cached is not None
    assert cached.html == "<p>real report html</p>"
    assert cached.duration_seconds == 123.4
    assert cached.run_label == "dry-run-2"


def test_cache_banner_mentions_it_is_not_a_live_run():
    demo_cache.save_cached_report("KRAS", "<p>x</p>", 90.0, "dry-run-1")
    cached = demo_cache.load_cached_report("KRAS")
    banner = demo_cache.cache_banner_html(cached)
    assert "pre-computed" in banner.lower()
    assert "dry-run-1" in banner
