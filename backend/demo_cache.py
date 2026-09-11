"""F11 -- curated demo targets + dry runs. Seed data as a first-class feature.

The three confirmed demo targets (GLP-1R the Beat-3 hero, HER2, KRAS -- see
project_brief.md Section 1.6, decided 2026-09-11) get their reports
pre-computed and cached to disk. This directly implements F13's testability
constraint: a live Bedrock call per juror click is not sustainable for a
project that must stay freely testable through 2026-10-08, so the three
demo targets are served from a pre-computed run instead, visibly marked as
cached -- never silently presented as a fresh live run.

Cache format is the rendered HTML directly (not a serialized Report/
CitationLedger object graph) plus a small metadata sidecar, since the
rendered HTML is exactly what /api/reports/{id}/report already serves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

DEMO_TARGETS = ["GLP-1R", "HER2", "KRAS"]

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_data", "report_cache")


def _normalize(target: str) -> str:
    return target.strip().upper().replace(" ", "")


_DEMO_TARGETS_NORMALIZED = {_normalize(t) for t in DEMO_TARGETS}


def is_demo_target(target: str) -> bool:
    return _normalize(target) in _DEMO_TARGETS_NORMALIZED


def _html_path(target: str) -> str:
    return os.path.join(_CACHE_DIR, f"{_normalize(target)}.html")


def _meta_path(target: str) -> str:
    return os.path.join(_CACHE_DIR, f"{_normalize(target)}.meta.json")


@dataclass
class CachedReport:
    html: str
    generated_at: str
    duration_seconds: float
    run_label: str  # e.g. "dry-run-1", "dry-run-2" -- which pre-warm pass produced this


def load_cached_report(target: str) -> Optional[CachedReport]:
    if not is_demo_target(target):
        return None
    html_path, meta_path = _html_path(target), _meta_path(target)
    if not (os.path.exists(html_path) and os.path.exists(meta_path)):
        return None
    with open(html_path) as fh:
        html = fh.read()
    with open(meta_path) as fh:
        meta = json.load(fh)
    return CachedReport(
        html=html,
        generated_at=meta["generated_at"],
        duration_seconds=meta["duration_seconds"],
        run_label=meta["run_label"],
    )


def save_cached_report(target: str, html: str, duration_seconds: float, run_label: str) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_html_path(target), "w") as fh:
        fh.write(html)
    meta = {
        "target": target,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 1),
        "run_label": run_label,
    }
    with open(_meta_path(target), "w") as fh:
        json.dump(meta, fh, indent=2)


def cache_banner_html(cached: CachedReport) -> str:
    """A visible, honest 'this is cached' banner injected above the cached HTML.

    F13's own rule: cached demo output is never silently presented as a
    fresh live run.
    """
    return (
        '<div style="font-family:\'Helvetica Neue\',Arial,sans-serif;font-size:0.85rem;'
        'background:#e7f0fb;border:1px solid #9db8db;border-radius:6px;'
        'padding:0.6rem 0.9rem;margin-bottom:1rem;">'
        f"<strong>Served from a pre-computed demo run</strong> ({cached.run_label}, "
        f"generated {cached.generated_at}, took {cached.duration_seconds:.0f}s live) -- "
        "this target is one of the three curated demo targets and is not re-run live per "
        "request, to keep this project sustainably testable. Try a different target name "
        "for a live run.</div>"
    )
