"""
CLI entry point for the Prompt Crash Test Lab visualization layer.

Usage:
    python run_visuals.py

Generates three PNGs in outputs/visuals/:
    pipeline.png      — end-to-end pipeline flow (Graphviz)
    architecture.png  — 3D-style layered architecture (matplotlib)
    metrics.png       — model robustness bar chart (matplotlib)
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is on the path so both src/ and backend/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.visualization.graphviz_generator import generate_pipeline_diagram
from backend.visualization.architecture_diagram import generate_architecture_diagram
from backend.visualization.pipeline_visualizer import generate_all

OUTPUT_DIR  = os.path.join("outputs", "visuals")
RESULTS_DIR = os.path.join("data", "results")   # written by src/analysis.py


def _banner(msg: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {msg}")
    print(f"{'─' * 55}")


def main() -> None:
    _banner("Prompt Crash Test Lab — Visualization Generator")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    errors: list[str] = []

    # 1 ── Pipeline diagram (Graphviz) ─────────────────────────────────────────
    print("\n[1/3] Generating pipeline flow diagram (Graphviz)...")
    t0 = time.perf_counter()
    try:
        out = generate_pipeline_diagram(os.path.join(OUTPUT_DIR, "pipeline"))
        print(f"      ✓  Saved → {out}  ({time.perf_counter() - t0:.1f}s)")
    except Exception as exc:
        msg = f"pipeline.png FAILED: {exc}"
        print(f"      ✗  {msg}")
        errors.append(msg)

    # 2 ── Architecture diagram (matplotlib) ───────────────────────────────────
    print("\n[2/3] Generating architecture diagram (matplotlib)...")
    t0 = time.perf_counter()
    try:
        out = generate_architecture_diagram(os.path.join(OUTPUT_DIR, "architecture.png"))
        print(f"      ✓  Saved → {out}  ({time.perf_counter() - t0:.1f}s)")
    except Exception as exc:
        msg = f"architecture.png FAILED: {exc}"
        print(f"      ✗  {msg}")
        errors.append(msg)

    # 3 ── Metrics chart (matplotlib) ──────────────────────────────────────────
    print("\n[3/3] Generating metrics chart (matplotlib)...")
    t0 = time.perf_counter()
    try:
        paths = generate_all(RESULTS_DIR, OUTPUT_DIR)
        for p in paths:
            print(f"      ✓  Saved → {p}  ({time.perf_counter() - t0:.1f}s)")
    except Exception as exc:
        msg = f"metrics.png FAILED: {exc}"
        print(f"      ✗  {msg}")
        errors.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("Done")
    if errors:
        print(f"\n  {len(errors)} error(s) occurred:")
        for e in errors:
            print(f"    • {e}")
        print(
            "\n  Tip: ensure you have run `pip install -r requirements.txt`\n"
            "  and the Graphviz system binary is installed.\n"
            "  See: https://graphviz.org/download/\n"
        )
        sys.exit(1)
    else:
        print(f"\n  All images saved to:  {os.path.abspath(OUTPUT_DIR)}/\n")


if __name__ == "__main__":
    main()
