"""Statistical analysis and visualization generation.

Produces publication-quality charts and statistical tests for the technical report.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config.settings import RESULTS_DIR


def load_scored_results(task_type: str) -> pd.DataFrame:
    """Load scored results from CSV."""
    path = RESULTS_DIR / f"{task_type}_scored.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def statistical_significance_tests(df: pd.DataFrame, metric: str = "field_accuracy") -> pd.DataFrame:
    """Run pairwise statistical tests between models.

    Uses Mann-Whitney U test (non-parametric) for robustness.
    Returns DataFrame with model pairs and p-values.
    """
    models = df["model"].unique()
    results = []

    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            scores1 = df[df["model"] == m1][metric].dropna().values
            scores2 = df[df["model"] == m2][metric].dropna().values

            if len(scores1) < 2 or len(scores2) < 2:
                continue

            stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative="two-sided")
            effect_size = abs(np.mean(scores1) - np.mean(scores2))

            results.append({
                "model_1": m1,
                "model_2": m2,
                "mean_1": float(np.mean(scores1)),
                "mean_2": float(np.mean(scores2)),
                "u_statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "effect_size": float(effect_size),
            })

    return pd.DataFrame(results)


def generate_model_comparison_chart(summary: dict, task_type: str, output_dir: Path):
    """Generate bar chart comparing models across key metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(summary.keys())
    if not models:
        return

    if task_type == "json_extraction":
        metrics = {
            "Robustness Score": [summary[m].get("robustness_score", 0) for m in models],
            "Format Compliance": [summary[m].get("format_compliance", 0) for m in models],
            "Field Accuracy": [summary[m].get("mean_field_accuracy", 0) for m in models],
        }
    else:
        metrics = {
            "Robustness Score": [summary[m].get("robustness_score", 0) for m in models],
            "Answer Score": [summary[m].get("mean_answer_score", 0) for m in models],
            "Citation Rate": [summary[m].get("citation_rate", 0) for m in models],
        }

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric_name)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison - {task_type.replace('_', ' ').title()}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task_type}_model_comparison.png", dpi=150)
    plt.close()


def generate_variant_type_heatmap(df: pd.DataFrame, metric: str, task_type: str, output_dir: Path):
    """Generate heatmap of metric scores by model and variant type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    pivot = df.pivot_table(values=metric, index="model", columns="variant_type", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title(f"{metric.replace('_', ' ').title()} by Model and Variant Type")
    plt.tight_layout()
    plt.savefig(output_dir / f"{task_type}_{metric}_heatmap.png", dpi=150)
    plt.close()


def generate_robustness_distribution(df: pd.DataFrame, metric: str, task_type: str, output_dir: Path):
    """Generate violin plot of score distributions per model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    models = df["model"].unique()
    data = [df[df["model"] == m][metric].dropna().values for m in models]

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Score Distribution by Model - {task_type.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task_type}_{metric}_violin.png", dpi=150)
    plt.close()


def generate_cost_efficiency_chart(summary: dict, task_type: str, output_dir: Path):
    """Generate scatter plot of accuracy vs cost (tokens)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(summary.keys())
    if not models:
        return

    if task_type == "json_extraction":
        accuracy_key = "mean_field_accuracy"
    else:
        accuracy_key = "mean_answer_score"

    accuracies = [summary[m].get(accuracy_key, 0) for m in models]
    tokens = [summary[m].get("avg_tokens", 0) for m in models]
    robustness = [summary[m].get("robustness_score", 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(tokens, accuracies, s=[r * 500 + 50 for r in robustness],
                         c=robustness, cmap="RdYlGn", alpha=0.8, edgecolors="black")

    for i, model in enumerate(models):
        ax.annotate(model, (tokens[i], accuracies[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    plt.colorbar(scatter, label="Robustness Score")
    ax.set_xlabel("Average Tokens Used")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(f"Cost Efficiency - {task_type.replace('_', ' ').title()}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task_type}_cost_efficiency.png", dpi=150)
    plt.close()


def run_full_analysis(task_type: str = "json_extraction"):
    """Run complete analysis pipeline and generate all charts."""
    from src.scoring.robustness import compute_all_metrics, aggregate_metrics

    results_path = RESULTS_DIR / f"{task_type}_results.jsonl"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return {}

    print(f"Computing metrics for {task_type}...")
    df = compute_all_metrics(results_path, task_type)

    if df.empty:
        print("No results to analyze.")
        return {}

    # Save scored results
    scored_path = RESULTS_DIR / f"{task_type}_scored.csv"
    df.to_csv(scored_path, index=False)
    print(f"Scored results saved to {scored_path}")

    # Aggregate metrics
    summary = aggregate_metrics(df, task_type)

    # Save summary
    summary_path = RESULTS_DIR / f"{task_type}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Generate charts
    charts_dir = RESULTS_DIR / "charts"
    metric = "field_accuracy" if task_type == "json_extraction" else "answer_score"

    generate_model_comparison_chart(summary, task_type, charts_dir)
    generate_variant_type_heatmap(df, metric, task_type, charts_dir)
    generate_robustness_distribution(df, metric, task_type, charts_dir)
    generate_cost_efficiency_chart(summary, task_type, charts_dir)
    print(f"Charts saved to {charts_dir}")

    # Statistical tests
    if len(df["model"].unique()) >= 2:
        sig_tests = statistical_significance_tests(df, metric)
        sig_path = RESULTS_DIR / f"{task_type}_significance.csv"
        sig_tests.to_csv(sig_path, index=False)
        print(f"Significance tests saved to {sig_path}")
        print("\nStatistical Significance Results:")
        print(sig_tests.to_string(index=False))

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {task_type.replace('_', ' ').upper()}")
    print(f"{'='*60}")
    for model, metrics in summary.items():
        print(f"\n{model}:")
        print(f"  Robustness Score: {metrics.get('robustness_score', 0):.4f}")
        if task_type == "json_extraction":
            print(f"  Format Compliance: {metrics.get('format_compliance', 0):.4f}")
            print(f"  Mean Field Accuracy: {metrics.get('mean_field_accuracy', 0):.4f}")
        else:
            print(f"  Mean Answer Score: {metrics.get('mean_answer_score', 0):.4f}")
            print(f"  Citation Rate: {metrics.get('citation_rate', 0):.4f}")
        print(f"  Error Rate: {metrics.get('error_rate', 0):.4f}")
        print(f"  Avg Tokens: {metrics.get('avg_tokens', 0):.0f}")

    return summary
