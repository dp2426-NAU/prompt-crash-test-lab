"""
Metric summary chart generator.
Loads scored results from analysis output (or uses demo data if none exist)
and exports grouped bar charts as PNG.
"""

from __future__ import annotations

import glob
import json
import os


# ── Demo data (used when no real results are present yet) ──────────────────────

_DEMO_DATA = {
    "json_extraction": {
        "gpt-4-turbo": {
            "robustness_score": 0.87,
            "format_compliance": 0.94,
            "field_accuracy": 0.91,
        },
        "claude-3-5-sonnet": {
            "robustness_score": 0.91,
            "format_compliance": 0.96,
            "field_accuracy": 0.93,
        },
        "gemini-1-5-pro": {
            "robustness_score": 0.79,
            "format_compliance": 0.88,
            "field_accuracy": 0.84,
        },
        "llama-3-1-70b": {
            "robustness_score": 0.73,
            "format_compliance": 0.81,
            "field_accuracy": 0.78,
        },
    },
    "grounded_qa": {
        "gpt-4-turbo": {
            "robustness_score": 0.84,
            "answer_score": 0.88,
            "citation_rate": 0.79,
        },
        "claude-3-5-sonnet": {
            "robustness_score": 0.89,
            "answer_score": 0.92,
            "citation_rate": 0.85,
        },
        "gemini-1-5-pro": {
            "robustness_score": 0.76,
            "answer_score": 0.80,
            "citation_rate": 0.71,
        },
        "llama-3-1-70b": {
            "robustness_score": 0.70,
            "answer_score": 0.74,
            "citation_rate": 0.62,
        },
    },
}


def load_results(results_dir: str) -> dict:
    """
    Scan results_dir for *_summary.json files written by src/analysis.py.
    Returns a dict keyed by task_type with per-model metric dicts.
    Falls back to demo data if no summaries are found.
    """
    summaries: dict = {}
    pattern = os.path.join(results_dir, "*_summary.json")
    for path in glob.glob(pattern):
        task_type = os.path.basename(path).replace("_summary.json", "")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            summaries[task_type] = data
        except (OSError, json.JSONDecodeError):
            continue

    if not summaries:
        print("  [pipeline_visualizer] No summary JSON found — using demo data.")
        return _DEMO_DATA

    return summaries


def generate_metrics_chart(data: dict, output_path: str) -> str:
    """
    Create a grouped bar chart of robustness scores per model and task type.

    Args:
        data:        Output of load_results().
        output_path: Full path for the PNG output.

    Returns:
        Absolute path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required: pip install matplotlib") from exc

    task_types = list(data.keys())
    if not task_types:
        return output_path

    # Collect all model names across task types
    all_models: list[str] = []
    for task_data in data.values():
        for model in task_data:
            if model not in all_models:
                all_models.append(model)

    n_models = len(all_models)
    n_tasks  = len(task_types)
    x        = np.arange(n_models)
    bar_w    = 0.35
    offsets  = np.linspace(-(n_tasks - 1) * bar_w / 2, (n_tasks - 1) * bar_w / 2, n_tasks)

    colors = ["#0275d8", "#5cb85c", "#d9534f", "#f0ad4e"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for t_idx, task_type in enumerate(task_types):
        task_data = data[task_type]
        scores = []
        for model in all_models:
            model_data = task_data.get(model, {})
            # Prefer robustness_score; fall back to any *_score key
            score = model_data.get("robustness_score")
            if score is None:
                for k, v in model_data.items():
                    if "score" in k and isinstance(v, (int, float)):
                        score = v
                        break
            scores.append(float(score) if score is not None else 0.0)

        bars = ax.bar(
            x + offsets[t_idx],
            scores,
            width=bar_w,
            label=task_type.replace("_", " ").title(),
            color=colors[t_idx % len(colors)],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.88,
        )

        # Value labels above each bar
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{score:.2f}",
                ha="center", va="bottom",
                fontsize=8.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("-", "\n") for m in all_models],
        fontsize=10,
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Robustness Score  (0 – 1)", fontsize=11)
    ax.set_title(
        "Model Robustness Comparison — Prompt Crash Test Lab",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#dee2e6")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


def generate_all(results_dir: str, output_dir: str) -> list[str]:
    """
    Load results and generate all metric charts into output_dir.

    Returns:
        List of absolute paths to the generated PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_results(results_dir)
    out_path = os.path.join(output_dir, "metrics.png")
    path = generate_metrics_chart(data, out_path)
    return [path]
