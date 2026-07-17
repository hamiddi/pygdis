#!/usr/bin/env python3
"""Run the manuscript benchmark and publication-figure pipelines in sequence."""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = [
    ROOT / "examples" / "publication" / "reproduce_benchmarks_and_sensitivity.py",
    ROOT / "examples" / "publication" / "reproduce_publication_figures.py",
]


def main() -> None:
    for pipeline in PIPELINES:
        print(f"\nRunning {pipeline.relative_to(ROOT)}")
        runpy.run_path(str(pipeline), run_name="__main__")


if __name__ == "__main__":
    main()
