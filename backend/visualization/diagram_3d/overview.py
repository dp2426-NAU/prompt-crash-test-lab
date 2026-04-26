"""
Full workflow overview diagram — complete 3D isometric system view.
Output: 3D_Workflow_Diagrams/Overview/full_workflow_3D.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, make_iso_figure, draw_platform, draw_box,
    draw_glow_cap, label_box, draw_arrow, add_title, add_legend,
)


# ── Stage definitions ─────────────────────────────────────────────────────────

STAGES = [
    {
        "key":      "input",
        "title":    "INPUT LAYER",
        "subtitle": "100 Base Prompts\n2 Task Types",
        "icon":     "📄",
        "color":    PALETTE["input"],
        "pos":      (0.0, 0.0, 0.0),
        "size":     (2.4, 1.6, 0.9),
    },
    {
        "key":      "variants",
        "title":    "VARIANT GENERATOR",
        "subtitle": "20 Variants / Prompt\n5 Mutation Types",
        "icon":     "⚙",
        "color":    PALETTE["variants"],
        "pos":      (2.8, 0.5, 0.5),
        "size":     (2.4, 1.6, 0.9),
    },
    {
        "key":      "models",
        "title":    "MODEL EXECUTION",
        "subtitle": "GPT-4 · Claude · Gemini · Llama\n4 LLM Providers",
        "icon":     "🤖",
        "color":    PALETTE["models"],
        "pos":      (5.6, 1.0, 1.0),
        "size":     (2.4, 1.6, 0.9),
    },
    {
        "key":      "storage",
        "title":    "STORAGE LAYER",
        "subtitle": "SQLite Cache\nJSONL Results",
        "icon":     "🗄",
        "color":    PALETTE["storage"],
        "pos":      (8.4, 1.5, 1.5),
        "size":     (2.4, 1.6, 0.9),
    },
    {
        "key":      "evaluation",
        "title":    "EVALUATION ENGINE",
        "subtitle": "6 Robustness Metrics\nStatistical Analysis",
        "icon":     "📊",
        "color":    PALETTE["evaluation"],
        "pos":      (11.2, 2.0, 2.0),
        "size":     (2.4, 1.6, 0.9),
    },
    {
        "key":      "viz",
        "title":    "VISUALIZATION",
        "subtitle": "Pipeline · Architecture\nMetric Charts",
        "icon":     "🖼",
        "color":    PALETTE["viz"],
        "pos":      (14.0, 2.5, 2.5),
        "size":     (2.4, 1.6, 0.9),
    },
]


def generate_overview(output_path: str) -> str:
    """
    Render the full 3D isometric workflow overview.

    Args:
        output_path: Full path including filename (.png).

    Returns:
        Absolute path to saved image.
    """
    fig, ax = make_iso_figure(figsize=(14, 14), dpi=150, elev=22, azim=-52)

    # Platform
    draw_platform(ax, -0.5, -0.5, 18.5, 5.5, z=-0.25)

    centres = []
    for stage in STAGES:
        centre = draw_box(ax, stage["pos"], stage["size"], stage["color"])
        draw_glow_cap(ax, stage["pos"], stage["size"], stage["color"])

        # Shift text centre to the top face centre
        tx, ty, tz = stage["pos"]
        sw, sd, sh = stage["size"]
        top_centre = np.array([tx + sw / 2, ty + sd / 2, tz + sh])

        label_box(
            ax, top_centre,
            title=stage["title"],
            subtitle=stage["subtitle"],
            icon=stage["icon"],
            title_size=8,
            sub_size=6.5,
            icon_size=11,
        )
        centres.append(centre)

    # Connectors between consecutive stages
    connector_labels = ["generate", "execute", "cache", "score", "render"]
    for i in range(len(centres) - 1):
        sx, sy, sz = STAGES[i]["pos"]
        sw, sd, sh = STAGES[i]["size"]
        ex, ey, ez = STAGES[i + 1]["pos"]

        start = np.array([sx + sw, sy + sd / 2, sz + sh / 2])
        end   = np.array([ex,       ey + sd / 2, ez + sh / 2])
        draw_arrow(ax, start, end, label=connector_labels[i], label_size=7)

    # ── Axes limits ────────────────────────────────────────────────────────────
    ax.set_xlim3d(-1, 18)
    ax.set_ylim3d(-1, 6)
    ax.set_zlim3d(-0.5, 5)

    # ── Title & legend ─────────────────────────────────────────────────────────
    add_title(
        fig,
        "Prompt Crash Test Lab — Full 3D Workflow",
        "LLM Robustness Evaluation Pipeline  ·  End-to-End Overview",
    )

    add_legend(fig, [
        (PALETTE["input"],      "Input Layer"),
        (PALETTE["variants"],   "Variant Generator"),
        (PALETTE["models"],     "Model Execution"),
        (PALETTE["storage"],    "Storage"),
        (PALETTE["evaluation"], "Evaluation Engine"),
        (PALETTE["viz"],        "Visualization"),
    ], y=0.05)

    # Watermark
    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)
