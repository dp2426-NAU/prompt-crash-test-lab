"""
Clean 3D diagram generators for 3D_Workflow_Output/ folder structure.

Produces high-resolution (2100×2100 px) isometric PNG diagrams with
no attribution text or contributor credits.

Generated outputs:
    Overview/   full_workflow_3D.png
    Pipeline/   workflow_pipeline_3D.png
    Steps/      step_01.png … step_06.png
    Components/ component_01.png … component_04.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    PALETTE, make_iso_figure, draw_platform,
    draw_box, draw_glow_cap, label_box, draw_arrow,
    add_title, add_legend,
)
from .pipeline import generate_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW  (full_workflow_3D.png)
# ═══════════════════════════════════════════════════════════════════════════════

_OVERVIEW_STAGES = [
    {"title": "INPUT\nLAYER",       "subtitle": "100 Base Prompts\n2 Task Types",           "icon": "📄", "color": PALETTE["input"],      "pos": (0.0,  0.0, 0.0), "size": (2.6, 1.8, 1.0)},
    {"title": "VARIANT\nGENERATOR", "subtitle": "20 Variants / Prompt\n5 Mutation Types",  "icon": "⚙",  "color": PALETTE["variants"],   "pos": (3.2,  0.6, 0.6), "size": (2.6, 1.8, 1.0)},
    {"title": "MODEL\nEXECUTION",  "subtitle": "4 LLM Providers\nGPT-4·Claude·Gemini·Llama", "icon": "🤖","color": PALETTE["models"],     "pos": (6.4,  1.2, 1.2), "size": (2.6, 1.8, 1.0)},
    {"title": "STORAGE\nCACHE",    "subtitle": "SQLite Cache\nJSONL Results",               "icon": "🗄", "color": PALETTE["storage"],    "pos": (9.6,  1.8, 1.8), "size": (2.6, 1.8, 1.0)},
    {"title": "EVALUATION\nENGINE","subtitle": "6 Robustness Metrics\nStatistical Analysis","icon": "📊","color": PALETTE["evaluation"], "pos": (12.8, 2.4, 2.4), "size": (2.6, 1.8, 1.0)},
    {"title": "VISUALIZATION\nLAYER","subtitle":"Diagrams · Charts\nDashboard",             "icon": "🖼","color": PALETTE["viz"],         "pos": (16.0, 3.0, 3.0), "size": (2.6, 1.8, 1.0)},
]

_CONNECTORS = ["generate", "execute", "cache", "score", "render"]


def generate_overview(output_path: str) -> str:
    """Full-system cascading isometric overview."""
    fig, ax = make_iso_figure(figsize=(14, 14), dpi=150, elev=22, azim=-52)
    draw_platform(ax, -0.5, -0.5, 20.5, 7.5, z=-0.25)

    centres = []
    for st in _OVERVIEW_STAGES:
        c = draw_box(ax, st["pos"], st["size"], st["color"])
        draw_glow_cap(ax, st["pos"], st["size"], st["color"])
        x, y, z = st["pos"]
        w, d, h = st["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=st["title"], subtitle=st["subtitle"],
                  icon=st["icon"], title_size=8.5, sub_size=7, icon_size=12)
        centres.append(c)

    for i in range(len(centres) - 1):
        x, y, z = _OVERVIEW_STAGES[i]["pos"]
        w, d, h = _OVERVIEW_STAGES[i]["size"]
        ex, ey, ez = _OVERVIEW_STAGES[i + 1]["pos"]
        draw_arrow(
            ax,
            np.array([x + w, y + d / 2, z + h / 2]),
            np.array([ex,    ey + d / 2, ez + h / 2]),
            label=_CONNECTORS[i], label_size=7.5,
        )

    ax.set_xlim3d(-1, 20); ax.set_ylim3d(-1, 7); ax.set_zlim3d(-0.5, 6)

    add_title(fig, "Prompt Crash Test Lab — Full System Workflow",
              "Complete 3D Isometric Architecture  ·  End-to-End View")
    add_legend(fig, [
        (PALETTE["input"],      "Input Layer"),
        (PALETTE["variants"],   "Variant Generator"),
        (PALETTE["models"],     "Model Execution"),
        (PALETTE["storage"],    "Storage Cache"),
        (PALETTE["evaluation"], "Evaluation Engine"),
        (PALETTE["viz"],        "Visualization Layer"),
    ], y=0.05)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STEPS  (step_01.png … step_06.png)
# ═══════════════════════════════════════════════════════════════════════════════

_STEPS = [
    {
        "num": "01", "color": PALETTE["input"],
        "title": "INPUT LAYER",
        "subtitle": "Loading & Preparing the 100 Base Prompts",
        "icon": "📄",
        "subs": [
            {"label": "JSON Extraction\nPrompts (50)",  "color": "#5DCED5", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Grounded Q&A\nPrompts (50)",     "color": "#3DB8C0", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "JSON Schema\nFiles (×5)",        "color": "#27A0A8", "pos": (0.0, 2.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Ground Truth\nLabels",           "color": "#1B8B93", "pos": (2.6, 2.0, 0.0), "size": (2.0, 1.4, 0.9)},
        ],
        "agg_pos": (0.6, 3.8, 0.0), "agg_size": (4.0, 1.2, 0.7),
        "agg_label": "JSONL Batch Loader",
        "out_label": "100 prompts → Variant Generator",
    },
    {
        "num": "02", "color": PALETTE["variants"],
        "title": "VARIANT GENERATOR",
        "subtitle": "5 Mutation Strategies — 20 Variants per Prompt",
        "icon": "⚙",
        "subs": [
            {"label": "Paraphrase ×5\nRule-based rewording",     "color": "#55C4DC", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.3, 0.9)},
            {"label": "Format ×4\nMarkdown/XML/List",            "color": "#3AAFCA", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.3, 0.9)},
            {"label": "Role ×3\nExpert/Assistant/Teacher",       "color": "#259BB8", "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.3, 0.9)},
            {"label": "Constraint ×3\nConcise/Detail/Simple",    "color": "#1588A6", "pos": (0.0, 2.0, 0.0), "size": (2.0, 1.3, 0.9)},
            {"label": "Template ×5\nZero/Few/CoT/Step/Struct",   "color": "#0A7594", "pos": (2.6, 2.0, 0.0), "size": (2.0, 1.3, 0.9)},
            {"label": "ID + Metadata\n8-char unique key",        "color": "#006282", "pos": (5.2, 2.0, 0.0), "size": (2.0, 1.3, 0.9)},
        ],
        "agg_pos": (1.6, 3.7, 0.0), "agg_size": (5.0, 1.2, 0.7),
        "agg_label": "JSONL Batch Writer  →  2,000 Variants",
        "out_label": "2,000 variants → Model Execution",
    },
    {
        "num": "03", "color": PALETTE["models"],
        "title": "MODEL EXECUTION LAYER",
        "subtitle": "4 LLM Providers — Parallel API Calls with Retry Logic",
        "icon": "🤖",
        "subs": [
            {"label": "GPT-4 Turbo\nOpenAI\ntemp=0.0, max=1024",       "color": PALETTE["gpt4"],   "pos": (0.0, 0.0, 0.0), "size": (2.2, 1.5, 1.1)},
            {"label": "Claude 3.5\nAnthropic\ntemp=0.0, max=1024",     "color": PALETTE["claude"], "pos": (2.8, 0.0, 0.0), "size": (2.2, 1.5, 1.1)},
            {"label": "Gemini 1.5 Pro\nGoogle\ntemp=0.0, max=1024",    "color": PALETTE["gemini"], "pos": (5.6, 0.0, 0.0), "size": (2.2, 1.5, 1.1)},
            {"label": "Llama 3.1 70B\nTogether AI\ntemp=0.0, max=1024","color": PALETTE["llama"],  "pos": (8.4, 0.0, 0.0), "size": (2.2, 1.5, 1.1)},
        ],
        "agg_pos": (2.0, 2.8, 0.0), "agg_size": (6.6, 1.2, 0.7),
        "agg_label": "Rate Limiter · Retry (3×) · Cost Tracker",
        "out_label": "~8,000 responses → Storage Cache",
    },
    {
        "num": "04", "color": PALETTE["storage"],
        "title": "STORAGE & CACHE LAYER",
        "subtitle": "SQLite Deduplication Cache + JSONL Persistence",
        "icon": "🗄",
        "subs": [
            {"label": "ResponseCache\nSQLite DB\nSHA-256 key",           "color": "#E8C833", "pos": (0.0, 0.0, 0.0), "size": (2.2, 1.5, 1.0)},
            {"label": "Cache Hit Check\nskip API call\n~30% savings",    "color": "#D4B625", "pos": (2.8, 0.0, 0.0), "size": (2.2, 1.5, 1.0)},
            {"label": "JSONL Results\nper-model files\n+ token counts",  "color": "#BFA418", "pos": (5.6, 0.0, 0.0), "size": (2.2, 1.5, 1.0)},
            {"label": "Variant Link\nparent_id ref\nmetadata chain",     "color": "#AA920B", "pos": (8.4, 0.0, 0.0), "size": (2.2, 1.5, 1.0)},
        ],
        "agg_pos": (2.0, 2.8, 0.0), "agg_size": (6.6, 1.2, 0.7),
        "agg_label": "Cache Stats  ·  API call count  ·  Error rate",
        "out_label": "Cached responses → Evaluation Engine",
    },
    {
        "num": "05", "color": PALETTE["evaluation"],
        "title": "EVALUATION ENGINE",
        "subtitle": "6 Robustness & Quality Metrics + Statistical Significance",
        "icon": "📊",
        "subs": [
            {"label": "Robustness\n1−(σ/μ)\nVariant consistency", "color": PALETTE["metric_r"],    "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Semantic\nSimilarity\nEmbedding cosine",   "color": PALETTE["metric_s"],    "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Format\nCompliance\nJSON schema valid",    "color": PALETTE["metric_f"],    "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Answer\nCorrectness\n40%+30%+30%",         "color": PALETTE["metric_a"],    "pos": (0.0, 2.1, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Citation\nAccuracy\nQuote matching",       "color": PALETTE["metric_c"],    "pos": (2.6, 2.1, 0.0), "size": (2.0, 1.4, 1.0)},
            {"label": "Cost\nEfficiency\n$/Tokens ratio",         "color": PALETTE["metric_cost"], "pos": (5.2, 2.1, 0.0), "size": (2.0, 1.4, 1.0)},
        ],
        "agg_pos": (0.8, 4.0, 0.0), "agg_size": (6.0, 1.2, 0.7),
        "agg_label": "Mann-Whitney U  ·  Effect sizes  ·  p-values",
        "out_label": "Scored CSVs + Summary JSON → Visualization",
    },
    {
        "num": "06", "color": PALETTE["viz"],
        "title": "VISUALIZATION LAYER",
        "subtitle": "Generating All Diagrams, Charts & Interactive Dashboard",
        "icon": "🖼",
        "subs": [
            {"label": "Pipeline\nDiagram\nGraphviz PNG",         "color": "#7ADA88", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Architecture\nDiagram\nmatplotlib PNG",   "color": "#5DC66A", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Metrics\nBar Chart\nRobustness scores",   "color": "#42B34D", "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Streamlit\nDashboard\nInteractive UI",    "color": "#2A9F36", "pos": (0.0, 2.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "Statistical\nHeatmaps\nVariant analysis", "color": "#168C20", "pos": (2.6, 2.0, 0.0), "size": (2.0, 1.4, 0.9)},
            {"label": "3D Workflow\nDiagrams\n11 PNG outputs",   "color": "#07790A", "pos": (5.2, 2.0, 0.0), "size": (2.0, 1.4, 0.9)},
        ],
        "agg_pos": (0.8, 4.0, 0.0), "agg_size": (6.0, 1.2, 0.7),
        "agg_label": "PNG · HTML · CSV · JSON  ·  Research-grade exports",
        "out_label": "All outputs → GitHub Repository / Research Paper",
    },
]


def _render_step(step: dict, output_path: str) -> None:
    fig, ax = make_iso_figure(figsize=(13, 11), dpi=150, elev=26, azim=-50)
    draw_platform(ax, -0.5, -0.5, 13, 8, z=-0.25)

    sub_centres = []
    for sc in step["subs"]:
        c = draw_box(ax, sc["pos"], sc["size"], sc["color"])
        draw_glow_cap(ax, sc["pos"], sc["size"], sc["color"], alpha=0.2)
        x, y, z = sc["pos"]
        w, d, h = sc["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=sc["label"], title_size=7.5)
        sub_centres.append(c)

    # Aggregator
    ap, asz = step["agg_pos"], step["agg_size"]
    agg_c = draw_box(ax, ap, asz, step["color"], alpha=0.95)
    draw_glow_cap(ax, ap, asz, step["color"])
    ax2, ay2, az2 = ap
    aw, ad, ah = asz
    label_box(ax, np.array([ax2 + aw / 2, ay2 + ad / 2, az2 + ah]),
              title=step["agg_label"], icon=step["icon"],
              title_size=8, icon_size=11)

    for sc in sub_centres:
        draw_arrow(ax, sc, agg_c, color="#7FB3D3", lw=1.2)

    out_s = np.array([ax2 + aw / 2, ay2 + ad / 2, az2 + ah + 0.25])
    draw_arrow(ax, out_s, out_s + [0, 0, 0.85],
               color=step["color"], label=step["out_label"], label_size=7)

    ax.set_xlim3d(-1, 12); ax.set_ylim3d(-1, 7.5); ax.set_zlim3d(-0.5, 4.5)

    add_title(fig, f"Step {step['num']}  ·  {step['title']}", step["subtitle"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_steps(output_dir: str) -> list[str]:
    """Generate step_01.png … step_06.png."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for step in _STEPS:
        out = os.path.join(output_dir, f"step_{step['num']}.png")
        _render_step(step, out)
        paths.append(os.path.abspath(out))
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENTS  (component_01.png … component_04.png)
# ═══════════════════════════════════════════════════════════════════════════════

def _component_01(output_path: str) -> str:
    """Component 01 — Model Clients (4 LLM providers)."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14.5, 7, z=-0.25)

    providers = [
        {"label": "GPT-4 Turbo",         "sub": "gpt-4-turbo\nOpenAI API\n$0.01/1K in",  "color": PALETTE["gpt4"],   "icon": "🟢", "pos": (0.0, 0.0, 0.0), "size": (2.8, 1.8, 1.3)},
        {"label": "Claude 3.5 Sonnet",   "sub": "claude-3-5-sonnet\nAnthropic API\n$0.003/1K in","color": PALETTE["claude"], "icon": "🟠", "pos": (3.4, 0.0, 0.0), "size": (2.8, 1.8, 1.3)},
        {"label": "Gemini 1.5 Pro",      "sub": "gemini-1.5-pro\nGoogle API\n$0.0035/1K in","color": PALETTE["gemini"], "icon": "🔵", "pos": (6.8, 0.0, 0.0), "size": (2.8, 1.8, 1.3)},
        {"label": "Llama 3.1 70B",       "sub": "Meta-Llama-70B\nTogether AI\n$0.0009/1K","color": PALETTE["llama"],  "icon": "🟣", "pos": (10.2, 0.0, 0.0), "size": (2.8, 1.8, 1.3)},
    ]

    base_c = draw_box(ax, (3.6, 3.4, 0.0), (5.8, 1.4, 0.9), "#2C3E50", alpha=0.92)
    draw_glow_cap(ax, (3.6, 3.4, 0.0), (5.8, 1.4, 0.9), "#2C3E50")
    label_box(ax, np.array([6.5, 4.1, 0.9]),
              title="BaseLLMClient  (Abstract Interface)",
              subtitle="generate(prompt, system_prompt) → LLMResponse",
              title_size=8.5, sub_size=7.5)

    for p in providers:
        c = draw_box(ax, p["pos"], p["size"], p["color"])
        draw_glow_cap(ax, p["pos"], p["size"], p["color"])
        x, y, z = p["pos"]; w, d, h = p["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=p["label"], subtitle=p["sub"],
                  icon=p["icon"], title_size=8.5, sub_size=7, icon_size=11)
        draw_arrow(ax, c, base_c, color="#7FB3D3", lw=1.4, label="inherits", label_size=6.5)

    ax.set_xlim3d(-1, 14); ax.set_ylim3d(-1, 6); ax.set_zlim3d(-0.5, 3.5)
    add_title(fig, "Component 01 — Model Clients",
              "4 LLM Providers · Unified BaseLLMClient Interface · LLMResponse Dataclass")
    add_legend(fig, [
        (PALETTE["gpt4"],   "GPT-4 Turbo (OpenAI)"),
        (PALETTE["claude"], "Claude 3.5 Sonnet (Anthropic)"),
        (PALETTE["gemini"], "Gemini 1.5 Pro (Google)"),
        (PALETTE["llama"],  "Llama 3.1 70B (Together AI)"),
    ], y=0.04)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


def _component_02(output_path: str) -> str:
    """Component 02 — Scoring System (6 evaluation metrics)."""
    fig, ax = make_iso_figure(figsize=(13, 11), dpi=150, elev=24, azim=-52)
    draw_platform(ax, -0.5, -0.5, 12, 8, z=-0.25)

    metrics = [
        {"label": "Robustness\nScore",    "sub": "1 − (σ / μ)\nVariant consistency",      "color": PALETTE["metric_r"],    "icon": "💪", "pos": (0.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0)},
        {"label": "Semantic\nSimilarity", "sub": "Cosine similarity\nMiniLM embeddings",   "color": PALETTE["metric_s"],    "icon": "🔗", "pos": (3.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0)},
        {"label": "Format\nCompliance",   "sub": "JSON schema valid\nField accuracy",       "color": PALETTE["metric_f"],    "icon": "✅", "pos": (6.0, 0.0, 0.0), "size": (2.4, 1.6, 1.0)},
        {"label": "Answer\nCorrectness",  "sub": "40% Semantic\n30% Keywords · 30% Cite",  "color": PALETTE["metric_a"],    "icon": "🎯", "pos": (0.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0)},
        {"label": "Citation\nAccuracy",   "sub": "Regex quote match\nvs. source text",      "color": PALETTE["metric_c"],    "icon": "📝", "pos": (3.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0)},
        {"label": "Cost\nEfficiency",     "sub": "Accuracy / Tokens\n$0.001–$0.02/1K",     "color": PALETTE["metric_cost"], "icon": "💰", "pos": (6.0, 2.6, 0.0), "size": (2.4, 1.6, 1.0)},
    ]

    agg_c = draw_box(ax, (0.8, 5.2, 0.0), (7.2, 1.4, 0.8), "#2C3E50", alpha=0.92)
    draw_glow_cap(ax, (0.8, 5.2, 0.0), (7.2, 1.4, 0.8), "#2C3E50")
    label_box(ax, np.array([4.4, 5.9, 0.8]),
              title="aggregate_metrics()  →  per-model summary JSON",
              subtitle="Mann-Whitney U  ·  p-values  ·  Effect sizes",
              title_size=8.5, sub_size=7.5)

    for m in metrics:
        c = draw_box(ax, m["pos"], m["size"], m["color"])
        draw_glow_cap(ax, m["pos"], m["size"], m["color"])
        x, y, z = m["pos"]; w, d, h = m["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=m["label"], subtitle=m["sub"],
                  icon=m["icon"], title_size=8.5, sub_size=6.5, icon_size=11)
        draw_arrow(ax, c, agg_c, color="#7FB3D3", lw=1.2)

    ax.set_xlim3d(-1, 10); ax.set_ylim3d(-1, 7.5); ax.set_zlim3d(-0.5, 3.5)
    add_title(fig, "Component 02 — Scoring System",
              "6 Evaluation Metrics · Formula Definitions · Composite Score Weighting")
    add_legend(fig, [
        (PALETTE["metric_r"],    "Robustness  (1−σ/μ)"),
        (PALETTE["metric_s"],    "Semantic Similarity"),
        (PALETTE["metric_f"],    "Format Compliance"),
        (PALETTE["metric_a"],    "Answer Correctness"),
        (PALETTE["metric_c"],    "Citation Accuracy"),
        (PALETTE["metric_cost"], "Cost Efficiency"),
    ], y=0.04)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


def _component_03(output_path: str) -> str:
    """Component 03 — Variant Generator (5 strategies)."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14, 8, z=-0.25)

    vtypes = [
        {"label": "Paraphrase\n×5",   "sub": '"Please extract…"\n"Your task is to…"\n"I need you to…"', "color": "#5BC8F5", "icon": "📝", "pos": (0.0, 0.0, 0.0), "size": (2.4, 1.8, 1.1)},
        {"label": "Format\n×4",       "sub": "Markdown\nPlaintext\nNumbered List\nXML",                  "color": "#3AB5E2", "icon": "🔧", "pos": (3.0, 0.0, 0.0), "size": (2.4, 1.8, 1.1)},
        {"label": "Role\n×3",         "sub": "Domain Expert\nHelpful Assistant\nMeticulous Teacher",     "color": "#1AA2CF", "icon": "🎭", "pos": (6.0, 0.0, 0.0), "size": (2.4, 1.8, 1.1)},
        {"label": "Constraint\n×3",   "sub": "Concise\nDetailed Reasoning\nSimplified Language",        "color": "#0A8FBC", "icon": "⛓", "pos": (9.0, 0.0, 0.0), "size": (2.4, 1.8, 1.1)},
        {"label": "Template\n×5",     "sub": "Zero-Shot\nFew-Shot\nChain-of-Thought\nStep-by-Step\nStructured", "color": "#007CA9", "icon": "📋", "pos": (4.5, 2.9, 0.0), "size": (2.4, 1.8, 1.1)},
    ]

    inp_c = draw_box(ax, (4.0, 5.5, 0.0), (3.4, 1.3, 0.8), PALETTE["input"], alpha=0.92)
    draw_glow_cap(ax, (4.0, 5.5, 0.0), (3.4, 1.3, 0.8), PALETTE["input"])
    label_box(ax, np.array([5.7, 6.15, 0.8]),
              title="Base Prompt  →  20 Variants",
              subtitle="8-char ID · parent_id linking · task_type tag",
              title_size=8.5, sub_size=7.5, icon="📄")

    for v in vtypes:
        c = draw_box(ax, v["pos"], v["size"], v["color"])
        draw_glow_cap(ax, v["pos"], v["size"], v["color"])
        x, y, z = v["pos"]; w, d, h = v["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=v["label"], subtitle=v["sub"],
                  icon=v["icon"], title_size=9, sub_size=6.5, icon_size=12)
        draw_arrow(ax, inp_c, c, color=PALETTE["variants"], lw=1.3)

    ax.set_xlim3d(-1, 13); ax.set_ylim3d(-1, 7.5); ax.set_zlim3d(-0.5, 3.5)
    add_title(fig, "Component 03 — Variant Generator",
              "5 Mutation Strategies · 20 Variants per Prompt · 2,000 Variants Total")
    add_legend(fig, [
        ("#5BC8F5", "Paraphrase (×5)"),
        ("#3AB5E2", "Format (×4)"),
        ("#1AA2CF", "Role (×3)"),
        ("#0A8FBC", "Constraint (×3)"),
        ("#007CA9", "Template (×5)"),
    ], y=0.04)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


def _component_04(output_path: str) -> str:
    """Component 04 — Dashboard & Exports."""
    fig, ax = make_iso_figure(figsize=(13, 10), dpi=150, elev=24, azim=-50)
    draw_platform(ax, -0.5, -0.5, 14.5, 8, z=-0.25)

    views = [
        {"label": "Model\nLeaderboard",  "color": "#7EC8E3", "pos": (0.0, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Robustness\nHeatmap", "color": "#6AB5D0", "pos": (2.6, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Cost vs.\nAccuracy",  "color": "#56A2BD", "pos": (5.2, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Significance\nTests", "color": "#428FAA", "pos": (7.8, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
        {"label": "Failure\nAnalysis",   "color": "#2E7C97", "pos": (10.4, 0.0, 0.0), "size": (2.0, 1.4, 0.9)},
    ]

    stl_c = draw_box(ax, (2.8, 2.4, 0.0), (5.8, 1.4, 0.8), "#1E6B88", alpha=0.92)
    draw_glow_cap(ax, (2.8, 2.4, 0.0), (5.8, 1.4, 0.8), "#1E6B88")
    label_box(ax, np.array([5.7, 3.1, 0.8]),
              title="Streamlit Dashboard  (python -m src.cli dashboard)",
              title_size=8.5, icon="🖥")

    for v in views:
        c = draw_box(ax, v["pos"], v["size"], v["color"])
        draw_glow_cap(ax, v["pos"], v["size"], v["color"])
        x, y, z = v["pos"]; w, d, h = v["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=v["label"], title_size=8)
        draw_arrow(ax, stl_c, c, color="#7FB3D3", lw=1.2)

    exports = [
        {"label": "pipeline.png\narchitecture.png\nmetrics.png",  "color": PALETTE["viz"],        "pos": (0.0, 4.5, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "*_scored.csv\n*_summary.json",                  "color": PALETTE["storage"],    "pos": (3.2, 4.5, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "Significance\n*_significance.csv",              "color": PALETTE["evaluation"], "pos": (6.4, 4.5, 0.0), "size": (2.6, 1.4, 0.9)},
        {"label": "3D Workflow\n11 PNG diagrams",                  "color": "#9B59B6",             "pos": (9.6, 4.5, 0.0), "size": (2.6, 1.4, 0.9)},
    ]

    for e in exports:
        c = draw_box(ax, e["pos"], e["size"], e["color"])
        draw_glow_cap(ax, e["pos"], e["size"], e["color"])
        x, y, z = e["pos"]; w, d, h = e["size"]
        label_box(ax, np.array([x + w / 2, y + d / 2, z + h]),
                  title=e["label"], title_size=7.5)
        draw_arrow(ax, stl_c, c, color="#C9A7EB", lw=1.0)

    ax.set_xlim3d(-1, 13.5); ax.set_ylim3d(-1, 7); ax.set_zlim3d(-0.5, 3.5)
    add_title(fig, "Component 04 — Dashboard & Exports",
              "Streamlit Interactive Views · All Output Formats · Research Artefacts")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


def generate_components(output_dir: str) -> list[str]:
    """Generate component_01.png … component_04.png."""
    os.makedirs(output_dir, exist_ok=True)
    return [
        _component_01(os.path.join(output_dir, "component_01.png")),
        _component_02(os.path.join(output_dir, "component_02.png")),
        _component_03(os.path.join(output_dir, "component_03.png")),
        _component_04(os.path.join(output_dir, "component_04.png")),
    ]
