#!/usr/bin/env python3
"""Ops script (not part of the web app) -- runs the real pipeline for each F11
demo target and saves the result via backend.demo_cache, so the deployed app
never needs a live Bedrock call for GLP-1R/HER2/KRAS.

Usage: python3 scripts/prewarm_demo_cache.py [--run-label dry-run-1]
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from backend.demo_cache import DEMO_TARGETS, save_cached_report
from backend.pipeline import run_pipeline
from backend.report_renderer import render_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="dry-run-1")
    parser.add_argument("--targets", nargs="*", default=DEMO_TARGETS)
    args = parser.parse_args()

    for target in args.targets:
        print(f"=== {target} ===")
        t0 = time.time()
        result = run_pipeline(target, on_progress=lambda s, st, d: print(f"  [{s}] {st} {d or ''}"))
        duration = time.time() - t0
        html = render_html(result.report)
        save_cached_report(target, html, duration, args.run_label)
        print(f"  -> {duration:.1f}s, {len(html)} bytes, is_partial={result.is_partial}")


if __name__ == "__main__":
    main()
