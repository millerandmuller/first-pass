"""Tests for backend.config -- specifically the serverless-safe cache dir default.

Verified live 2026-09-11: a real Vercel deployment 500'd trying to write a
PDF cache file next to the deployed code ("[Errno 30] Read-only file
system") -- Vercel's deployment filesystem is read-only except /tmp.
"""

import importlib
import os


def _reload_config():
    import backend.config as config
    return importlib.reload(config)


def test_cache_dir_defaults_to_tmp_when_running_on_vercel(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("FIRST_PASS_CACHE_DIR", raising=False)
    config = _reload_config()
    assert config.CACHE_DIR == "/tmp/first-pass-cache"


def test_cache_dir_defaults_to_local_dir_outside_vercel(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("FIRST_PASS_CACHE_DIR", raising=False)
    config = _reload_config()
    assert config.CACHE_DIR != "/tmp/first-pass-cache"
    assert os.path.basename(os.path.normpath(config.CACHE_DIR)) == "cache"


def test_explicit_env_var_always_overrides_the_default(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("FIRST_PASS_CACHE_DIR", "/custom/path")
    config = _reload_config()
    assert config.CACHE_DIR == "/custom/path"
