"""
Component-level diagrams — detailed internal structure of key pipeline modules.
Outputs:
  component_A.png  — Model Clients (4 LLMs)
  component_B.png  — Scoring System (6 metrics with formulas)
  component_C.png  — Variant Types (5 mutation strategies)
  component_D.png  — Dashboard & Export (Streamlit + outputs)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, make_iso_figure, draw_platform, draw_box,
    draw_glow_cap, label_box, draw_arrow, add_title, add_legend,
)


# ─────────────────────────────────────────────────────────────────────────────
# Component A — Model Clients
# ─────────────────────────────────────────────────────────────────────────────

def generate_component_a(output_path: str) -> str:
    """Model Clients: 4 LLM providers with their configuration."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14, 6, z=-0.25)

    models = [
        {
            "label":    "GPT-4 Turbo",
            "sub":      "gpt-4-turbo\nmax_tokens: 1024\ntemp: 0.0",
            "provider": "OpenAI",
            "color":    PALETTE["gpt4"],
            "icon":     "🟢",
            "pos":      (0.0, 0.0, 0.0), "size": (2.6, 1.8, 1.2),
        },
        {
            "label":    "Claude 3.5 Sonnet",
            "sub":      "claude-3-5-sonnet-20241022\nmax_tokens: 1024\ntemp: 0.0",
            "provider": "Anthropic",
            "color":    PALETTE["claude"],
            "icon":     "🟠",
            "pos":      (3.2, 0.0, 0.0), "size": (2.6, 1.8, 1.2),
        },
        {
            "label":    "Gemini 1.5 Pro",
            "sub":      "gemini-1.5-pro\nmax_tokens: 1024\ntemp: 0.0",
            "provider": "Google",
            "color":    PALETTE["gemini"],
            "icon":     "🔵",
            "pos":      (6.4, 0.0, 0.0), "size": (2.6, 1.8, 1.2),
        },
        {
            "label":    "Llama 3.1 70B",
            "sub":      "Meta-Llama-3.1-70B-Instruct\nmax_tokens: 1024\ntemp: 0.0",
            "provider": "Together AI",
            "color":    PALETTE["llama"],
            "icon":     "🟣",
            "pos":      (9.6, 0.0, 0.0), "size": (2.6, 1.8, 1.2),
        },
    ]

    # Base client box
    base_c = draw_box(ax, (3.6, 3.2, 0.0), (5.6, 1.4, 0.8), "#34495e", alpha=0.9)
    draw_glow_cap(ax, (3.6, 3.2, 0.0), (5.6, 1.4, 0.8), "#34495e")
    label_box(ax, np.array([6.4, 3.9, 0.8]),
              title="BaseLLMClient (Abstract)",
              subtitle="generate(prompt, system_prompt, **kwargs)\n→ LLMResponse(text, model, tokens, latency_ms)",
              title_size=8.5, sub_size=7)

    centres = []
    for m in models:
        c = draw_box(ax, m["pos"], m["size"], m["color"])
        draw_glow_cap(ax, m["pos"], m["size"], m["color"])
        px, py, pz = m["pos"]
        sw, sd, sh = m["size"]
        top = np.array([px + sw / 2, py + sd / 2, pz + sh])
        label_box(ax, top, title=m["label"], subtitle=m["sub"],
                  icon=m["icon"], title_size=8, sub_size=6.5, icon_size=10)
        centres.append(c)

    # Arrows from each model up to base class
    for c in centres:
        draw_arrow(ax, c, base_c, color="#7FB3D3", lw=1.4,
                   label="inherits", label_size=6.5)

    ax.set_xlim3d(-1, 13.5)
    ax.set_ylim3d(-1, 5.5)
    ax.set_zlim3d(-0.5, 3.5)

    add_title(fig, "Component A — Model Clients",
              "4 LLM Providers implementing BaseLLMClient")
    add_legend(fig, [
        (PALETTE["gpt4"],   "GPT-4 Turbo  (OpenAI)"),
        (PALETTE["claude"], "Claude 3.5 Sonnet  (Anthropic)"),
        (PALETTE["gemini"], "Gemini 1.5 Pro  (Google)"),
        (PALETTE["llama"],  "Llama 3.1 70B  (Together AI)"),
    ], y=0.04)

    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Component B — Scoring System
# ─────────────────────────────────────────────────────────────────────────────

def generate_component_b(output_path: str) -> str:
    """Scoring System: 6 evaluation metrics with formulas."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-52)
    draw_platform(ax, -0.5, -0.5, 13, 7, z=-0.25)

    metrics = [
        {
            "label": "Robustness\nScore",
            "sub":   "1 − (σ / μ)\nConsistency across variants",
            "color": PALETTE["metric_r"], "icon": "💪",
            "pos": (0.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0),
        },
        {
            "label": "Semantic\nSimilarity",
            "sub":   "Cosine(emb_i, emb_j)\nall-MiniLM-L6-v2",
            "color": PALETTE["metric_s"], "icon": "🔗",
            "pos": (3.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0),
        },
        {
            "label": "Format\nCompliance",
            "sub":   "JSON Schema Valid\n+ Field Accuracy",
            "color": PALETTE["metric_f"], "icon": "✅",
            "pos": (6.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0),
        },
        {
            "label": "Answer\nCorrectness",
            "sub":   "40% Semantic\n30% Keywords\n30% Citations",
            "color": PALETTE["metric_a"], "icon": "🎯",
            "pos": (0.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0),
        },
        {
            "label": "Citation\nAccuracy",
            "sub":   "Regex quote match\nvs. source context",
            "color": PALETTE["metric_c"], "icon": "📝",
            "pos": (3.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0),
        },
        {
            "label": "Cost\nEfficiency",
            "sub":   "Accuracy / Tokens\n$0.001–$0.02 / 1K",
            "color": PALETTE["metric_cost"], "icon": "💰",
            "pos": (6.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0),
        },
    ]

    # Aggregator
    agg_c = draw_box(ax, (1.0, 5.4, 0.0), (7.0, 1.2, 0.7), "#2C3E50", alpha=0.92)
    draw_glow_cap(ax, (1.0, 5.4, 0.0), (7.0, 1.2, 0.7), "#2C3E50")
    label_box(ax, np.array([4.5, 6.0, 0.7]),
              title="aggregate_metrics()  →  per-model summary JSON",
              title_size=8)

    centres = []
    for m in metrics:
        c = draw_box(ax, m["pos"], m["size"], m["color"])
        draw_glow_cap(ax, m["pos"], m["size"], m["color"])
        px, py, pz = m["pos"]
        sw, sd, sh = m["size"]
        top = np.array([px + sw / 2, py + sd / 2, pz + sh])
        label_box(ax, top, title=m["label"], subtitle=m["sub"],
                  icon=m["icon"], title_size=8, sub_size=6.5, icon_size=11)
        centres.append(c)

    for c in centres:
        draw_arrow(ax, c, agg_c, color="#7FB3D3", lw=1.2)

    ax.set_xlim3d(-1, 10)
    ax.set_ylim3d(-1, 7)
    ax.set_zlim3d(-0.5, 3.5)

    add_title(fig, "Component B — Scoring System",
              "6 Evaluation Metrics with Formulas and Weights")
    add_legend(fig, [
        (PALETTE["metric_r"],    "Robustness (1−σ/μ)"),
        (PALETTE["metric_s"],    "Semantic Similarity"),
        (PALETTE["metric_f"],    "Format Compliance"),
        (PALETTE["metric_a"],    "Answer Correctness"),
        (PALETTE["metric_c"],    "Citation Accuracy"),
        (PALETTE["metric_cost"], "Cost Efficiency"),
    ], y=0.04)

    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Component C — Variant Types
# ─────────────────────────────────────────────────────────────────────────────

def generate_component_c(output_path: str) -> str:
    """Variant Types: 5 mutation strategies with examples."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14, 7, z=-0.25)

    variants = [
        {
            "label": "Paraphrase\n(×5)",
            "sub":   '"Please extract…"\n"Your task is to…"\n"I need you to…"',
            "color": "#5BC8F5", "icon": "📝",
            "pos": (0.0, 0.0, 0.0), "size": (2.4, 1.8, 1.0),
        },
        {
            "label": "Format\n(×4)",
            "sub":   "Markdown\nPlaintext\nNumbered List\nXML",
            "color": "#3AB5E2", "icon": "🔧",
            "pos": (3.0, 0.0, 0.0), "size": (2.4, 1.8, 1.0),
        },
        {
            "label": "Role\n(×3)",
            "sub":   "Domain Expert\nHelpful Asst.\nMeticulous Tchr.",
            "color": "#1AA2CF", "icon": "🎭",
            "pos": (6.0, 0.0, 0.0), "size": (2.4, 1.8, 1.0),
        },
        {
            "label": "Constraint\n(×3)",
            "sub":   "Concise\nDetailed Reasoning\nSimplified",
            "color": "#0A8FBC", "icon": "⛓",
            "pos": (9.0, 0.0, 0.0), "size": (2.4, 1.8, 1.0),
        },
        {
            "label": "Template\n(×5)",
            "sub":   "Zero-Shot\nFew-Shot\nChain-of-Thought\nStep-by-Step\nStructured",
            "color": "#007CA9", "icon": "📋",
            "pos": (4.5, 2.8, 0.0), "size": (2.4, 1.8, 1.0),
        },
    ]

    # Input prompt box
    inp_c = draw_box(ax, (4.2, 5.4, 0.0), (3.0, 1.2, 0.7), PALETTE["input"], alpha=0.92)
    draw_glow_cap(ax, (4.2, 5.4, 0.0), (3.0, 1.2, 0.7), PALETTE["input"])
    label_box(ax, np.array([5.7, 6.0, 0.7]),
              title="Base Prompt  →  20 Variants",
              subtitle="8-char unique ID  ·  parent_id linking",
              title_size=8.5, sub_size=7, icon="📄")

    centres = []
    for v in variants:
        c = draw_box(ax, v["pos"], v["size"], v["color"])
        draw_glow_cap(ax, v["pos"], v["size"], v["color"])
        px, py, pz = v["pos"]
        sw, sd, sh = v["size"]
        top = np.array([px + sw / 2, py + sd / 2, pz + sh])
        label_box(ax, top, title=v["label"], subtitle=v["sub"],
                  icon=v["icon"], title_size=8.5, sub_size=6.5, icon_size=11)
        centres.append(c)

    # Arrows from input box to each variant type
    for c in centres:
        draw_arrow(ax, inp_c, c, color=PALETTE["variants"], lw=1.3)

    ax.set_xlim3d(-1, 13)
    ax.set_ylim3d(-1, 7)
    ax.set_zlim3d(-0.5, 3.5)

    add_title(fig, "Component C — Variant Generator",
              "5 Mutation Strategies · 20 Variants per Base Prompt · 2,000 Total")
    add_legend(fig, [
        ("#5BC8F5", "Paraphrase (5)"),
        ("#3AB5E2", "Format (4)"),
        ("#1AA2CF", "Role (3)"),
        ("#0A8FBC", "Constraint (3)"),
        ("#007CA9", "Template (5)"),
    ], y=0.04)

    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Component D — Dashboard & Exports
# ─────────────────────────────────────────────────────────────────────────────

def generate_component_d(output_path: str) -> str:
    """Dashboard & Export: Streamlit views + all output artefacts."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14, 7, z=-0.25)

    # Streamlit section
    views = [
        {"label": "Model\nLeaderboard",    "color": "#7EC8E3", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Robustness\nHeatmap",   "color": "#6AB5D0", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Cost\nAnalysis",        "color": "#56A2BD", "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Significance\nTests",   "color": "#428FAA", "pos": (7.8, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Failure\nAnalysis",     "color": "#2E7C97", "pos": (10.4, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
    ]

    stl_c = draw_box(ax, (3.2, 2.4, 0.0), (5.4, 1.2, 0.7), "#1E6B88", alpha=0.92)
    draw_glow_cap(ax, (3.2, 2.4, 0.0), (5.4, 1.2, 0.7), "#1E6B88")
    label_box(ax, np.array([5.9, 3.0, 0.7]),
              title="Streamlit Dashboard  (python -m src.cli dashboard)",
              title_size=8, icon="🖥")

    v_centres = []
    for v in views:
        c = draw_box(ax, v["pos"], v["size"], v["color"])
        draw_glow_cap(ax, v["pos"], v["size"], v["color"])
        px, py, pz = v["pos"]
        sw, sd, sh = v["size"]
        label_box(ax, np.array([px + sw / 2, py + sd / 2, pz + sh]),
                  title=v["label"], title_size=7.5)
        v_centres.append(c)

    for vc in v_centres:
        draw_arrow(ax, stl_c, vc, color="#7FB3D3", lw=1.2)

    # Export artefacts
    exports = [
        {"label": "pipeline.png\narchitecture.png\nmetrics.png", "color": PALETTE["viz"],        "pos": (0.0, 4.4, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "*_scored.csv\n*_summary.json",                 "color": PALETTE["storage"],    "pos": (3.2, 4.4, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "*_significance.csv\nStatistical tests",        "color": PALETTE["evaluation"], "pos": (6.4, 4.4, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "3D_Workflow_Diagrams/\nAll PNG outputs",       "color": "#9B59B6",             "pos": (9.6, 4.4, 0.0), "size": (2.6, 1.4, 0.9)},
    ]

    for e in exports:
        c = draw_box(ax, e["pos"], e["size"], e["color"])
        draw_glow_cap(ax, e["pos"], e["size"], e["color"])
        px, py, pz = e["pos"]
        sw, sd, sh = e["size"]
        top = np.array([px + sw / 2, py + sd / 2, pz + sh])
        label_box(ax, top, title=e["label"], title_size=7.5)
        draw_arrow(ax, stl_c, c, color="#C9A7EB", lw=1.0, label="export")

    ax.set_xlim3d(-1, 13.5)
    ax.set_ylim3d(-1, 7)
    ax.set_zlim3d(-0.5, 3.5)

    add_title(fig, "Component D — Dashboard & Exports",
              "Streamlit Interactive Views + All Output Artefacts")

    fig.text(0.98, 0.02, "github.com/dp2426-NAU/prompt-crash-test-lab",
             ha="right", fontsize=7, color="#2A5298", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all_components(output_dir: str) -> list[str]:
    """Generate component_A … component_D PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    paths = [
        generate_component_a(os.path.join(output_dir, "component_A.png")),
        generate_component_b(os.path.join(output_dir, "component_B.png")),
        generate_component_c(os.path.join(output_dir, "component_C.png")),
        generate_component_d(os.path.join(output_dir, "component_D.png")),
    ]
    return paths
