"""Shared HTTP helper: retry-with-backoff, then a labeled cached fallback.

Edge case 3 from the brief: "openFDA slow/unreachable -> retry with backoff,
then cached demo dataset, visibly marked as cached." This module is the one
place that rule is implemented so every caller (target resolution, retrieval
layer) gets it for free instead of re-implementing retry logic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from backend.config import (
    CACHE_DIR,
    OPENFDA_MAX_RETRIES,
    OPENFDA_RETRY_BACKOFF_SECONDS,
    OPENFDA_TIMEOUT_SECONDS,
)


@dataclass
class FetchResult:
    ok: bool
    status: str  # "live" | "cached" | "zero_matches" | "not_found" | "error"
    data: Optional[dict[str, Any]]
    url: str
    error: Optional[str] = None


def _cache_path(url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in url)[-180:]
    return os.path.join(CACHE_DIR, f"{safe_name}.json")


def get_json(url: str, *, use_cache_fallback: bool = True) -> FetchResult:
    """GET a JSON endpoint with retry+backoff; fall back to a cached copy on failure.

    Returns FetchResult.status == "cached" only when a live attempt failed AND a
    prior successful response was cached — callers must surface that status to
    the user (never silently present cached data as live).
    """
    last_error: Optional[str] = None
    for attempt in range(1, OPENFDA_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=OPENFDA_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(OPENFDA_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code == 404:
            # openFDA returns 404 for two distinct cases that must not be
            # conflated: a genuine zero-match query (a valid, informative
            # result -- e.g. no drug class named "Kinase Inhibitor") vs a
            # malformed/unreachable endpoint. Only the JSON error code tells
            # them apart.
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if body.get("error", {}).get("code") == "NOT_FOUND":
                return FetchResult(ok=True, status="zero_matches", data=body, url=url)
            return FetchResult(ok=False, status="not_found", data=None, url=url, error="404")

        if resp.status_code != 200:
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(OPENFDA_RETRY_BACKOFF_SECONDS * attempt)
            continue

        data = resp.json()
        if use_cache_fallback:
            try:
                with open(_cache_path(url), "w") as fh:
                    json.dump(data, fh)
            except OSError:
                pass
        return FetchResult(ok=True, status="live", data=data, url=url)

    if use_cache_fallback:
        cache_file = _cache_path(url)
        if os.path.exists(cache_file):
            with open(cache_file) as fh:
                cached = json.load(fh)
            return FetchResult(ok=True, status="cached", data=cached, url=url, error=last_error)

    return FetchResult(ok=False, status="error", data=None, url=url, error=last_error)
