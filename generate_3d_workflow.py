"""
3D Workflow Diagram Package — Prompt Crash Test Lab
====================================================

Generates a complete set of professional 3D isometric workflow diagrams
and organises them into a structured output folder.

Usage:
    python generate_3d_workflow.py

Output:
    3D_Workflow_Output/
        Overview/    full_workflow_3D.png
        Pipeline/    workflow_pipeline_3D.png
        Steps/       step_01.png … step_06.png
        Components/  component_01.png … component_04.png
        Assets/      Icons / Textures / Labels
        README.md

Requirements:
    pip install matplotlib numpy
    (No API keys needed — pure diagram generation)
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.visualization.diagram_3d.workflow_output import (
    generate_overview,
    generate_steps,
    generate_components,
)
from backend.visualization.diagram_3d.pipeline import generate_pipeline

# ── Output directories ────────────────────────────────────────────────────────

BASE     = os.path.join(os.path.dirname(__file__), "3D_Workflow_Output")
OVERVIEW = os.path.join(BASE, "Overview")
PIPELINE = os.path.join(BASE, "Pipeline")
STEPS    = os.path.join(BASE, "Steps")
COMPS    = os.path.join(BASE, "Components")


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


def _ok(label: str, path: str, elapsed: float) -> None:
    print(f"      ✓  [{label}]  {os.path.basename(path)}  ({elapsed:.1f}s)")


def _fail(label: str, exc: Exception) -> str:
    msg = f"{label} FAILED: {exc}"
    print(f"      ✗  {msg}")
    return msg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("Prompt Crash Test Lab  ·  3D Workflow Diagram Generator")
    print(f"\n  Output root  →  {os.path.abspath(BASE)}\n")

    errors: list[str] = []

    # ── 1 · Full system overview ───────────────────────────────────────────────
    print("[ 1 / 4 ]  Full system overview  (full_workflow_3D.png) …")
    t0 = time.perf_counter()
    try:
        p = generate_overview(os.path.join(OVERVIEW, "full_workflow_3D.png"))
        _ok("Overview", p, time.perf_counter() - t0)
    except Exception as exc:
        errors.append(_fail("Overview", exc))

    # ── 2 · Pipeline flow diagram ──────────────────────────────────────────────
    print("\n[ 2 / 4 ]  Workflow pipeline  (workflow_pipeline_3D.png) …")
    t0 = time.perf_counter()
    try:
        p = generate_pipeline(os.path.join(PIPELINE, "workflow_pipeline_3D.png"))
        _ok("Pipeline", p, time.perf_counter() - t0)
    except Exception as exc:
        errors.append(_fail("Pipeline", exc))

    # ── 3 · Step-by-step diagrams ──────────────────────────────────────────────
    print("\n[ 3 / 4 ]  Step-by-step breakdown  (step_01 … step_06) …")
    t0 = time.perf_counter()
    try:
        paths = generate_steps(STEPS)
        for p in paths:
            _ok("Step", p, 0)
        print(f"             All 6 steps done in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        errors.append(_fail("Steps", exc))

    # ── 4 · Component diagrams ─────────────────────────────────────────────────
    print("\n[ 4 / 4 ]  Component diagrams  (component_01 … component_04) …")
    t0 = time.perf_counter()
    try:
        paths = generate_components(COMPS)
        for p in paths:
            _ok("Component", p, 0)
        print(f"             All 4 components done in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        errors.append(_fail("Components", exc))

    # ── Summary ────────────────────────────────────────────────────────────────
    _banner("Generation Complete")

    if errors:
        print(f"\n  {len(errors)} error(s) — check output above.")
        for e in errors:
            print(f"    • {e}")
        print("\n  Ensure dependencies are installed:\n"
              "    pip install matplotlib numpy\n")
        sys.exit(1)

    total = 1 + 1 + 6 + 4
    print(f"""
  {total} diagrams saved to:  {os.path.abspath(BASE)}/

  ┌── 3D_Workflow_Output/
  │   ├── Overview/
  │   │   └── full_workflow_3D.png        (complete system view)
  │   ├── Pipeline/
  │   │   └── workflow_pipeline_3D.png    (flat data-flow view)
  │   ├── Steps/
  │   │   ├── step_01.png  Input Layer
  │   │   ├── step_02.png  Variant Generator
  │   │   ├── step_03.png  Model Execution
  │   │   ├── step_04.png  Storage & Cache
  │   │   ├── step_05.png  Evaluation Engine
  │   │   └── step_06.png  Visualization Layer
  │   ├── Components/
  │   │   ├── component_01.png  Model Clients
  │   │   ├── component_02.png  Scoring System
  │   │   ├── component_03.png  Variant Types
  │   │   └── component_04.png  Dashboard & Exports
  │   ├── Assets/
  │   │   ├── Icons/
  │   │   ├── Textures/
  │   │   └── Labels/
  │   └── README.md
""")


if __name__ == "__main__":
    main()
