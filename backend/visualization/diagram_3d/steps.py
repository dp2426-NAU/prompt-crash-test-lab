"""
Step-by-step breakdown diagrams.
Each step is rendered as a zoomed-in 3D scene showing internal sub-components.
Outputs: 3D_Workflow_Diagrams/Step_By_Step/step_01.png … step_06.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, make_iso_figure, draw_platform, draw_box,
    draw_glow_cap, label_box, draw_arrow, add_title,
)


# ── Step definitions ──────────────────────────────────────────────────────────

STEPS = [
    {
        "number": "01",
        "title": "INPUT LAYER",
        "subtitle": "Loading & Preparing Base Prompts",
        "color": PALETTE["input"],
        "icon": "📄",
        "sub_components": [
            {"label": "JSON\nExtraction\nPrompts (50)",  "color": "#5DCED5", "pos": (0.0, 0.0, 0.0), "size": (1.8, 1.4, 0.8)},
            {"label": "Grounded\nQ&A Prompts\n(50)",     "color": "#3DB8C0", "pos": (2.4, 0.0, 0.0), "size": (1.8, 1.4, 0.8)},
            {"label": "Schema\nFiles (5)",               "color": "#27A0A8", "pos": (0.0, 2.0, 0.0), "size": (1.8, 1.4, 0.8)},
            {"label": "Ground\nTruth\nLabels",           "color": "#1B8B93", "pos": (2.4, 2.0, 0.0), "size": (1.8, 1.4, 0.8)},
        ],
        "main_box_pos":  (0.8, 3.8, 0.0),
        "main_box_size": (3.2, 1.2, 0.6),
        "main_label":    "JSONL Loader",
        "output_label":  "100 prompts → Variant Generator",
    },
    {
        "number": "02",
        "title": "VARIANT GENERATOR",
        "subtitle": "Generating 20 Semantic Variants per Prompt",
        "color": PALETTE["variants"],
        "icon": "⚙",
        "sub_components": [
            {"label": "Paraphrase\n(5 variants)",   "color": "#55C4DC", "pos": (0.0, 0.0, 0.0), "size": (1.8, 1.2, 0.8)},
            {"label": "Format\n(4 variants)",       "color": "#3AAFCA", "pos": (2.4, 0.0, 0.0), "size": (1.8, 1.2, 0.8)},
            {"label": "Role\n(3 variants)",         "color": "#259BB8", "pos": (4.8, 0.0, 0.0), "size": (1.8, 1.2, 0.8)},
            {"label": "Constraint\n(3 variants)",   "color": "#1588A6", "pos": (0.0, 1.8, 0.0), "size": (1.8, 1.2, 0.8)},
            {"label": "Template\n(5 variants)",     "color": "#0A7594", "pos": (2.4, 1.8, 0.0), "size": (1.8, 1.2, 0.8)},
            {"label": "Unique\nID + Metadata",      "color": "#006282", "pos": (4.8, 1.8, 0.0), "size": (1.8, 1.2, 0.8)},
        ],
        "main_box_pos":  (1.8, 3.4, 0.0),
        "main_box_size": (3.2, 1.2, 0.6),
        "main_label":    "JSONL Batch Writer",
        "output_label":  "2,000 variants → Model Execution",
    },
    {
        "number": "03",
        "title": "MODEL EXECUTION LAYER",
        "subtitle": "Running Variants Across 4 LLM Providers",
        "color": PALETTE["models"],
        "icon": "🤖",
        "sub_components": [
            {"label": "GPT-4 Turbo\ngpt-4-turbo\nOpenAI",          "color": PALETTE["gpt4"],   "pos": (0.0, 0.0, 0.0), "size": (1.8, 1.4, 1.0)},
            {"label": "Claude 3.5\nSonnet\nAnthropic",             "color": PALETTE["claude"], "pos": (2.4, 0.0, 0.0), "size": (1.8, 1.4, 1.0)},
            {"label": "Gemini 1.5\nPro\nGoogle",                   "color": PALETTE["gemini"], "pos": (4.8, 0.0, 0.0), "size": (1.8, 1.4, 1.0)},
            {"label": "Llama 3.1\n70B-Instruct\nTogether AI",      "color": PALETTE["llama"],  "pos": (7.2, 0.0, 0.0), "size": (1.8, 1.4, 1.0)},
        ],
        "main_box_pos":  (2.2, 2.4, 0.0),
        "main_box_size": (5.4, 1.2, 0.6),
        "main_label":    "Retry Logic · Rate Limiting · Cost Tracking",
        "output_label":  "~8,000 responses → Storage Layer",
    },
    {
        "number": "04",
        "title": "STORAGE LAYER",
        "subtitle": "Caching, Deduplication & Persistence",
        "color": PALETTE["storage"],
        "icon": "🗄",
        "sub_components": [
            {"label": "ResponseCache\nSQLite DB\n(SHA-256 key)",   "color": "#E8C833", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Cache Hit\nDetection\n(skip API call)",     "color": "#D4B625", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "JSONL\nResult Files\n(per model)",          "color": "#BFA418", "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Variant\nMetadata\nLinking",                "color": "#AA920B", "pos": (7.8, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
        ],
        "main_box_pos":  (2.6, 2.4, 0.0),
        "main_box_size": (5.4, 1.2, 0.6),
        "main_label":    "Stats: Cache Hits · API Calls · Error Rate",
        "output_label":  "Cached responses → Evaluation Engine",
    },
    {
        "number": "05",
        "title": "EVALUATION ENGINE",
        "subtitle": "Computing 6 Robustness & Quality Metrics",
        "color": PALETTE["evaluation"],
        "icon": "📊",
        "sub_components": [
            {"label": "Robustness\n1−(σ/μ)\nConsistency",         "color": PALETTE["metric_r"], "pos": (0.0, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Semantic\nSimilarity\nEmbeddings",         "color": PALETTE["metric_s"], "pos": (2.4, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Format\nCompliance\nSchema Valid",         "color": PALETTE["metric_f"], "pos": (4.8, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Answer\nCorrectness\nKeyword+Embed",       "color": PALETTE["metric_a"], "pos": (0.0, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Citation\nAccuracy\nQuote Match",          "color": PALETTE["metric_c"], "pos": (2.4, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Cost\nEfficiency\nTokens/Score",           "color": PALETTE["metric_cost"], "pos": (4.8, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
        ],
        "main_box_pos":  (1.2, 4.0, 0.0),
        "main_box_size": (5.0, 1.2, 0.6),
        "main_label":    "Mann-Whitney U · Statistical Significance",
        "output_label":  "Scored CSVs + Summary JSON → Visualization",
    },
    {
        "number": "06",
        "title": "VISUALIZATION LAYER",
        "subtitle": "Generating All Diagrams, Charts & Dashboard",
        "color": PALETTE["viz"],
        "icon": "🖼",
        "sub_components": [
            {"label": "Pipeline\nDiagram\n(Graphviz)",            "color": "#7ADA88", "pos": (0.0, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Architecture\nDiagram\n(matplotlib)",      "color": "#5DC66A", "pos": (2.4, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Metrics\nBar Chart\n(matplotlib)",         "color": "#42B34D", "pos": (4.8, 0.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Streamlit\nDashboard\n(interactive)",      "color": "#2A9F36", "pos": (0.0, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "Statistical\nHeatmaps\n(seaborn)",         "color": "#168C20", "pos": (2.4, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
            {"label": "3D Workflow\nDiagrams\n(this output)",     "color": "#07790A", "pos": (4.8, 2.0, 0.0), "size": (1.8, 1.4, 0.9)},
        ],
        "main_box_pos":  (1.2, 4.0, 0.0),
        "main_box_size": (5.0, 1.2, 0.6),
        "main_label":    "PNG · HTML · CSV · JSON exports",
        "output_label":  "Research-grade outputs → GitHub / Paper",
    },
]


def _generate_step(step: dict, output_path: str) -> None:
    """Render one step diagram."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=26, azim=-50)

    draw_platform(ax, -0.5, -0.5, 12, 7.5, z=-0.25)

    # Sub-component boxes
    sub_centres = []
    for sc in step["sub_components"]:
        c = draw_box(ax, sc["pos"], sc["size"], sc["color"])
        draw_glow_cap(ax, sc["pos"], sc["size"], sc["color"], alpha=0.2)
        px, py, pz = sc["pos"]
        sw, sd, sh = sc["size"]
        top_c = np.array([px + sw / 2, py + sd / 2, pz + sh])
        label_box(ax, top_c, title=sc["label"], title_size=7.5, sub_size=6)
        sub_centres.append(c)

    # Main aggregator box
    mp = step["main_box_pos"]
    ms = step["main_box_size"]
    main_c = draw_box(ax, mp, ms, step["color"], alpha=0.95)
    draw_glow_cap(ax, mp, ms, step["color"])
    mx, my, mz = mp
    mw, md, mh = ms
    main_top = np.array([mx + mw / 2, my + md / 2, mz + mh])
    label_box(ax, main_top, title=step["main_label"], title_size=8, icon=step["icon"])

    # Arrows from sub-components into main box
    for sc_centre in sub_centres:
        draw_arrow(ax, sc_centre, main_c, color="#7FB3D3", lw=1.2)

    # Output arrow from main box
    out_start = np.array([mx + mw / 2, my + md / 2, mz + mh + 0.3])
    out_end   = out_start + np.array([0, 0, 0.8])
    draw_arrow(ax, out_start, out_end, color=step["color"], label=step["output_label"], label_size=7)

    # Axis limits
    ax.set_xlim3d(-1, 11)
    ax.set_ylim3d(-1, 7)
    ax.set_zlim3d(-0.5, 4)

    add_title(
        fig,
        f"Step {step['number']}  ·  {step['title']}",
        step["subtitle"],
    )

    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_all_steps(output_dir: str) -> list[str]:
    """
    Generate step_01.png … step_06.png in output_dir.

    Returns:
        List of absolute paths to saved images.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for step in STEPS:
        out = os.path.join(output_dir, f"step_{step['number']}.png")
        _generate_step(step, out)
        paths.append(os.path.abspath(out))
    return paths
