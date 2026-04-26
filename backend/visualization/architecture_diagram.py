"""
3D-style layered architecture diagram using matplotlib.
Each layer is rendered with a shadow offset to simulate depth.
"""

from __future__ import annotations

import os


# ── Layer definitions (bottom → top) ───────────────────────────────────────────

_LAYERS = [
    {
        "label": "Storage Layer",
        "sublabel": "SQLite Cache  ·  JSONL Results  ·  Data Persistence",
        "color": "#adb5bd",
        "edge": "#6c757d",
        "text": "#212529",
    },
    {
        "label": "Evaluation Engine",
        "sublabel": "Robustness Score  ·  Schema Validation  ·  Semantic Similarity  ·  Answer Correctness",
        "color": "#b5a7e8",
        "edge": "#6f42c1",
        "text": "#2d0f6b",
    },
    {
        "label": "Model Execution Layer",
        "sublabel": "GPT-4 Turbo     Claude 3.5 Sonnet     Gemini 1.5 Pro     Llama 3.1 70B",
        "color": "#f8b4b4",
        "edge": "#dc3545",
        "text": "#5c0000",
    },
    {
        "label": "Variant Generator",
        "sublabel": "Paraphrase  ·  Format  ·  Role  ·  Constraint  ·  Template   (20 variants / prompt)",
        "color": "#9fd3f5",
        "edge": "#0275d8",
        "text": "#003366",
    },
    {
        "label": "Input Layer",
        "sublabel": "100 Base Prompts  ·  JSON Extraction  ·  Grounded Q&A",
        "color": "#a8d5b5",
        "edge": "#28a745",
        "text": "#0a3d1f",
    },
]


def generate_architecture_diagram(output_path: str) -> str:
    """
    Draw a layered '3D-style' architecture stack and save as PNG.

    Args:
        output_path: Full file path including .png extension.

    Returns:
        Absolute path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError("matplotlib is required: pip install matplotlib") from exc

    fig_w, fig_h = 13, 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    # Title
    ax.text(
        fig_w / 2, fig_h - 0.4,
        "Prompt Crash Test Lab — System Architecture",
        ha="center", va="top",
        fontsize=16, fontweight="bold", color="#212529",
        fontfamily="DejaVu Sans",
    )
    ax.text(
        fig_w / 2, fig_h - 0.85,
        "LLM Robustness Evaluation Framework",
        ha="center", va="top",
        fontsize=11, color="#6c757d",
        fontfamily="DejaVu Sans",
    )

    # ── Draw layers ────────────────────────────────────────────────────────────

    layer_h = 1.0          # height of each layer box
    layer_w = 9.5          # width
    x_left  = (fig_w - layer_w) / 2
    y_start = 1.0          # bottom of lowest layer
    gap     = 0.25         # vertical gap between layers
    shadow  = 0.12         # shadow offset (right + down)

    for i, layer in enumerate(_LAYERS):
        y_bottom = y_start + i * (layer_h + gap)

        # Shadow rectangle
        shadow_patch = mpatches.FancyBboxPatch(
            (x_left + shadow, y_bottom - shadow),
            layer_w, layer_h,
            boxstyle="round,pad=0.04",
            linewidth=0,
            facecolor="#cccccc",
            zorder=2,
        )
        ax.add_patch(shadow_patch)

        # Main rectangle
        main_patch = mpatches.FancyBboxPatch(
            (x_left, y_bottom),
            layer_w, layer_h,
            boxstyle="round,pad=0.04",
            linewidth=1.8,
            edgecolor=layer["edge"],
            facecolor=layer["color"],
            zorder=3,
        )
        ax.add_patch(main_patch)

        # Layer title
        ax.text(
            x_left + layer_w / 2,
            y_bottom + layer_h * 0.65,
            layer["label"],
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=layer["text"],
            zorder=4,
        )

        # Layer sublabel
        ax.text(
            x_left + layer_w / 2,
            y_bottom + layer_h * 0.3,
            layer["sublabel"],
            ha="center", va="center",
            fontsize=9, color=layer["text"],
            alpha=0.85,
            zorder=4,
        )

        # Connector arrow to next layer
        if i < len(_LAYERS) - 1:
            ax.annotate(
                "",
                xy=(x_left + layer_w / 2, y_bottom + layer_h + gap),
                xytext=(x_left + layer_w / 2, y_bottom + layer_h),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#495057",
                    lw=1.6,
                ),
                zorder=5,
            )

    # ── Visualization Layer annotation (right side) ────────────────────────────

    viz_x = x_left + layer_w + 0.5
    viz_y = y_start + 0                    # same height as storage layer
    viz_w = 2.2
    viz_h = len(_LAYERS) * (layer_h + gap) - gap

    viz_patch = mpatches.FancyBboxPatch(
        (viz_x, viz_y),
        viz_w, viz_h,
        boxstyle="round,pad=0.04",
        linewidth=1.8,
        edgecolor="#0c5460",
        facecolor="#d1ecf1",
        zorder=3,
    )
    ax.add_patch(viz_patch)

    ax.text(
        viz_x + viz_w / 2,
        viz_y + viz_h * 0.55,
        "Visualization\nLayer",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="#0c5460",
        zorder=4,
    )
    ax.text(
        viz_x + viz_w / 2,
        viz_y + viz_h * 0.28,
        "Pipeline PNG\nArchitecture PNG\nMetric Charts",
        ha="center", va="center",
        fontsize=8.5, color="#0c5460",
        alpha=0.85,
        zorder=4,
    )

    # Arrow from storage layer to viz box
    ax.annotate(
        "",
        xy=(viz_x, viz_y + viz_h / 2),
        xytext=(x_left + layer_w, viz_y + viz_h / 2),
        arrowprops=dict(
            arrowstyle="->",
            color="#0c5460",
            lw=1.6,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )

    # ── Footer ─────────────────────────────────────────────────────────────────

    ax.text(
        fig_w / 2, 0.25,
        "github.com/dp2426-NAU/prompt-crash-test-lab",
        ha="center", va="bottom",
        fontsize=8, color="#adb5bd",
    )

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)
