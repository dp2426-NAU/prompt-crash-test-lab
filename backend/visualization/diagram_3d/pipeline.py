"""
Workflow Pipeline 3D Diagram.

Renders the LLM robustness evaluation pipeline as a flat, panoramic
isometric view showing all six stages at the same depth level with
data-volume labels on every connector.

Output: 3D_Workflow_Output/Pipeline/workflow_pipeline_3D.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, make_iso_figure, draw_platform,
    draw_box, draw_glow_cap, label_box, draw_arrow, add_title,
)


# ── Stage data ────────────────────────────────────────────────────────────────

_PIPE_STAGES = [
    {
        "key":     "input",
        "title":   "INPUT\nLAYER",
        "stats":   "100 Prompts\n2 Task Types",
        "icon":    "📄",
        "color":   PALETTE["input"],
        "out_vol": "100 prompts",
    },
    {
        "key":     "variants",
        "title":   "VARIANT\nGENERATOR",
        "stats":   "×20 Variants\n5 Strategies",
        "icon":    "⚙",
        "color":   PALETTE["variants"],
        "out_vol": "2,000 variants",
    },
    {
        "key":     "models",
        "title":   "MODEL\nEXECUTION",
        "stats":   "4 LLM Providers\nParallel Calls",
        "icon":    "🤖",
        "color":   PALETTE["models"],
        "out_vol": "8,000 responses",
    },
    {
        "key":     "storage",
        "title":   "STORAGE\nCACHE",
        "stats":   "SQLite Cache\nJSONL Files",
        "icon":    "🗄",
        "color":   PALETTE["storage"],
        "out_vol": "8,000 cached",
    },
    {
        "key":     "evaluation",
        "title":   "EVALUATION\nENGINE",
        "stats":   "6 Metrics\nStatistical Tests",
        "icon":    "📊",
        "color":   PALETTE["evaluation"],
        "out_vol": "Scored results",
    },
    {
        "key":     "viz",
        "title":   "VISUALIZATION\nLAYER",
        "stats":   "11 Diagrams\nDashboard + PNGs",
        "icon":    "🖼",
        "color":   PALETTE["viz"],
        "out_vol": "Research outputs",
    },
]

# ── Data-pipe connector ───────────────────────────────────────────────────────

def _draw_pipe(
    ax,
    x_start: float, x_end: float,
    y: float, z: float,
    color: str,
    label: str,
    pipe_h: float = 0.18,
    pipe_d: float = 0.22,
) -> None:
    """Draw a 3D rectangular tunnel connecting two stages."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgb

    gap = 0.02
    xs, xe = x_start + gap, x_end - gap

    r, g, b = to_rgb(color)

    # Top face
    top = [[xs, y - pipe_d, z + pipe_h], [xe, y - pipe_d, z + pipe_h],
           [xe, y + pipe_d, z + pipe_h], [xs, y + pipe_d, z + pipe_h]]
    # Bottom face
    bot = [[xs, y - pipe_d, z], [xe, y - pipe_d, z],
           [xe, y + pipe_d, z], [xs, y + pipe_d, z]]
    # Front face
    front = [[xs, y - pipe_d, z], [xe, y - pipe_d, z],
             [xe, y - pipe_d, z + pipe_h], [xs, y - pipe_d, z + pipe_h]]

    faces = [
        (top,   (r,        g,        b,        0.75)),
        (bot,   (r * 0.4,  g * 0.4,  b * 0.4,  0.6)),
        (front, (r * 0.7,  g * 0.7,  b * 0.7,  0.7)),
    ]
    for verts, fc in faces:
        poly = Poly3DCollection([verts], alpha=fc[3])
        poly.set_facecolor(fc[:3])
        poly.set_edgecolor("white")
        poly.set_linewidth(0.3)
        ax.add_collection3d(poly)

    # Arrowhead at destination
    ax.annotate3D = None  # silent no-op guard
    mid_x = (xs + xe) / 2
    ax.text(
        mid_x, y - pipe_d - 0.05, z + pipe_h + 0.12,
        label,
        ha="center", va="bottom",
        fontsize=7.5, color=color,
        fontweight="bold",
        alpha=0.95,
        zorder=8,
    )


def generate_pipeline(output_path: str) -> str:
    """
    Render the flat isometric pipeline flow diagram.

    Args:
        output_path: Full path including .png extension.

    Returns:
        Absolute path to the saved image.
    """
    fig, ax = make_iso_figure(figsize=(20, 11), dpi=150, elev=30, azim=-48)

    BOX_W = 2.8
    BOX_D = 1.8
    BOX_H = 1.4
    GAP   = 1.2   # horizontal gap between boxes
    STEP  = BOX_W + GAP

    draw_platform(ax, -0.4, -0.5, len(_PIPE_STAGES) * STEP + 0.2, BOX_D + 2.5,
                  z=-0.22, alpha=0.35)

    centres = []
    for i, stage in enumerate(_PIPE_STAGES):
        x = i * STEP
        pos  = (x, 0.0, 0.0)
        size = (BOX_W, BOX_D, BOX_H)

        c = draw_box(ax, pos, size, stage["color"], alpha=0.93)
        draw_glow_cap(ax, pos, size, stage["color"], alpha=0.2, layers=2)

        # Top-face label
        top = np.array([x + BOX_W / 2, BOX_D / 2, BOX_H])
        label_box(
            ax, top,
            title=stage["title"],
            subtitle=stage["stats"],
            icon=stage["icon"],
            title_size=8.5,
            sub_size=7,
            icon_size=12,
        )

        # Stage number badge (small inset box on the front face)
        badge_z = 0.08
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.colors import to_rgb
        r2, g2, b2 = to_rgb(stage["color"])
        badge = Poly3DCollection([[
            [x + 0.08, -0.01, badge_z],
            [x + 0.52, -0.01, badge_z],
            [x + 0.52, -0.01, badge_z + 0.30],
            [x + 0.08, -0.01, badge_z + 0.30],
        ]], alpha=0.9)
        badge.set_facecolor((r2 * 0.55, g2 * 0.55, b2 * 0.55))
        badge.set_edgecolor("white")
        badge.set_linewidth(0.4)
        ax.add_collection3d(badge)
        ax.text(x + 0.30, -0.06, badge_z + 0.15,
                f"{i+1:02d}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=9)

        centres.append(c)

    # ── Data-pipe connectors ───────────────────────────────────────────────────
    for i, stage in enumerate(_PIPE_STAGES[:-1]):
        x_start = i * STEP + BOX_W
        x_end   = (i + 1) * STEP
        mid_color = _PIPE_STAGES[i + 1]["color"]
        _draw_pipe(ax, x_start, x_end, BOX_D / 2, BOX_H * 0.45,
                   mid_color, label=stage["out_vol"])

    # ── Axes limits ────────────────────────────────────────────────────────────
    total_w = len(_PIPE_STAGES) * STEP
    ax.set_xlim3d(-1, total_w + 0.5)
    ax.set_ylim3d(-1.5, BOX_D + 2)
    ax.set_zlim3d(-0.5, 3.5)

    add_title(
        fig,
        "LLM Robustness Evaluation — Workflow Pipeline",
        "End-to-end data flow  ·  100 prompts  →  2,000 variants  →  8,000 responses  →  Research outputs",
    )

    # ── Data-volume legend strip at bottom ────────────────────────────────────
    fig.text(0.5, 0.04,
             "INPUT  ──100──▶  VARIANTS  ──2,000──▶  RESPONSES  ──8,000──▶  CACHE  ──8,000──▶  SCORES  ──Metrics──▶  OUTPUTS",
             ha="center", va="center",
             fontsize=9, color="#7FB3D3",
             fontweight="bold",
             family="monospace")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)
