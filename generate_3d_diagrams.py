"""
3D Workflow Diagram Generator — Prompt Crash Test Lab
======================================================

Generates all 3D isometric workflow diagrams into:
    3D_Workflow_Diagrams/
        Overview/         full_workflow_3D.png
        Step_By_Step/     step_01.png … step_06.png
        Components/       component_A.png … component_D.png

Usage:
    python generate_3d_diagrams.py

Requirements:
    pip install matplotlib numpy

No API keys needed — pure diagram generation.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.visualization.diagram_3d.overview    import generate_overview
from backend.visualization.diagram_3d.steps       import generate_all_steps
from backend.visualization.diagram_3d.components  import generate_all_components

# ── Output directories ────────────────────────────────────────────────────────

BASE_DIR    = os.path.join(os.path.dirname(__file__), "3D_Workflow_Diagrams")
OVERVIEW_DIR = os.path.join(BASE_DIR, "Overview")
STEPS_DIR   = os.path.join(BASE_DIR, "Step_By_Step")
COMP_DIR    = os.path.join(BASE_DIR, "Components")


def _banner(msg: str) -> None:
    print(f"\n{'═' * 62}")
    print(f"  {msg}")
    print(f"{'═' * 62}")


def main() -> None:
    _banner("Prompt Crash Test Lab  ·  3D Workflow Diagram Generator")
    print(f"\n  Output root: {os.path.abspath(BASE_DIR)}\n")

    errors: list[str] = []

    # ── 1. Full workflow overview ──────────────────────────────────────────────
    print("[ 1 / 3 ]  Full workflow overview diagram …")
    t0 = time.perf_counter()
    try:
        out = generate_overview(os.path.join(OVERVIEW_DIR, "full_workflow_3D.png"))
        print(f"           ✓  {out}  ({time.perf_counter()-t0:.1f}s)")
    except Exception as exc:
        msg = f"Overview FAILED: {exc}"
        print(f"           ✗  {msg}")
        errors.append(msg)

    # ── 2. Step-by-step diagrams ───────────────────────────────────────────────
    print("\n[ 2 / 3 ]  Step-by-step breakdown (6 diagrams) …")
    t0 = time.perf_counter()
    try:
        paths = generate_all_steps(STEPS_DIR)
        for p in paths:
            print(f"           ✓  {p}")
        print(f"           Completed in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        msg = f"Steps FAILED: {exc}"
        print(f"           ✗  {msg}")
        errors.append(msg)

    # ── 3. Component diagrams ──────────────────────────────────────────────────
    print("\n[ 3 / 3 ]  Component-level diagrams (4 diagrams) …")
    t0 = time.perf_counter()
    try:
        paths = generate_all_components(COMP_DIR)
        for p in paths:
            print(f"           ✓  {p}")
        print(f"           Completed in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        msg = f"Components FAILED: {exc}"
        print(f"           ✗  {msg}")
        errors.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("Done")
    if errors:
        print(f"\n  {len(errors)} error(s):")
        for e in errors:
            print(f"    • {e}")
        print("\n  Make sure dependencies are installed:\n"
              "    pip install matplotlib numpy\n")
        sys.exit(1)
    else:
        total = 1 + 6 + 4   # overview + steps + components
        print(f"\n  {total} PNG diagrams saved to:  {os.path.abspath(BASE_DIR)}/")
        print("""
  Folder layout:
    3D_Workflow_Diagrams/
    ├── Overview/
    │   └── full_workflow_3D.png
    ├── Step_By_Step/
    │   ├── step_01.png  (Input Layer)
    │   ├── step_02.png  (Variant Generator)
    │   ├── step_03.png  (Model Execution)
    │   ├── step_04.png  (Storage Layer)
    │   ├── step_05.png  (Evaluation Engine)
    │   └── step_06.png  (Visualization Layer)
    └── Components/
        ├── component_A.png  (Model Clients)
        ├── component_B.png  (Scoring System)
        ├── component_C.png  (Variant Types)
        └── component_D.png  (Dashboard & Exports)
""")


if __name__ == "__main__":
    main()
