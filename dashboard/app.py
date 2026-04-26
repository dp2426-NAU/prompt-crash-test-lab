"""Streamlit Dashboard for Prompt Crash-Test Lab.

Interactive visualization of LLM robustness benchmark results.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RESULTS_DIR

st.set_page_config(
    page_title="Prompt Crash-Test Lab",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Prompt Crash-Test Lab")
st.markdown("**LLM Robustness Benchmark Dashboard** — CS 599: Contemporary Developments")
st.divider()


def load_summary(task_type: str) -> dict:
    path = RESULTS_DIR / f"{task_type}_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_scored(task_type: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{task_type}_scored.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_significance(task_type: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{task_type}_significance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────
task_type = st.sidebar.selectbox(
    "Task Type",
    ["json_extraction", "grounded_qa"],
    format_func=lambda x: "JSON Extraction" if x == "json_extraction" else "Grounded Q&A",
)

summary = load_summary(task_type)
df = load_scored(task_type)
sig_df = load_significance(task_type)

if not summary:
    st.warning("No results found. Run the evaluation pipeline first:")
    st.code("""
# Step 1: Generate variants
python -m src.cli generate

# Step 2: Run evaluation (example with one model)
python -m src.cli run --task json_extraction --models gpt-4-turbo --max-variants 20

# Step 3: Score results
python -m src.cli score --task json_extraction

# Step 4: Launch dashboard
python -m src.cli dashboard
    """)
    st.stop()

# ── Leaderboard ───────────────────────────────────────────
st.header("📊 Model Leaderboard")

leaderboard_data = []
for model, metrics in summary.items():
    row = {"Model": model, "Robustness": metrics.get("robustness_score", 0)}
    if task_type == "json_extraction":
        row["Format Compliance"] = metrics.get("format_compliance", 0)
        row["Field Accuracy"] = metrics.get("mean_field_accuracy", 0)
    else:
        row["Answer Score"] = metrics.get("mean_answer_score", 0)
        row["Citation Rate"] = metrics.get("citation_rate", 0)
    row["Error Rate"] = metrics.get("error_rate", 0)
    row["Avg Tokens"] = metrics.get("avg_tokens", 0)
    leaderboard_data.append(row)

lb_df = pd.DataFrame(leaderboard_data).sort_values("Robustness", ascending=False)
st.dataframe(lb_df, use_container_width=True, hide_index=True)

# ── Key Metrics Cards ─────────────────────────────────────
st.header("📈 Key Metrics")
cols = st.columns(len(summary))

for i, (model, metrics) in enumerate(summary.items()):
    with cols[i]:
        st.subheader(model)
        st.metric("Robustness", f"{metrics.get('robustness_score', 0):.3f}")
        if task_type == "json_extraction":
            st.metric("Accuracy", f"{metrics.get('mean_field_accuracy', 0):.3f}")
            st.metric("Compliance", f"{metrics.get('format_compliance', 0):.1%}")
        else:
            st.metric("Answer Score", f"{metrics.get('mean_answer_score', 0):.3f}")
            st.metric("Citation Rate", f"{metrics.get('citation_rate', 0):.1%}")

# ── Comparison Charts ─────────────────────────────────────
st.header("📊 Model Comparison")

if task_type == "json_extraction":
    metric_col = "field_accuracy"
    metric_label = "Field Accuracy"
else:
    metric_col = "answer_score"
    metric_label = "Answer Score"

if not df.empty:
    # Bar chart - overall comparison
    model_means = df.groupby("model")[metric_col].agg(["mean", "std"]).reset_index()
    fig_bar = px.bar(
        model_means, x="model", y="mean", error_y="std",
        title=f"Mean {metric_label} by Model",
        labels={"mean": metric_label, "model": "Model"},
        color="model",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap - variant type breakdown
    col1, col2 = st.columns(2)

    with col1:
        pivot = df.pivot_table(values=metric_col, index="model", columns="variant_type", aggfunc="mean")
        fig_heat = px.imshow(
            pivot,
            text_auto=".3f",
            title=f"{metric_label} by Model × Variant Type",
            color_continuous_scale="YlGnBu",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Box plot
        fig_box = px.box(
            df, x="model", y=metric_col, color="variant_type",
            title=f"{metric_label} Distribution by Variant Type",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Robustness Deep Dive ──────────────────────────────
    st.header("🔍 Robustness Analysis")

    # Per-prompt robustness
    prompt_robustness = []
    for (base_id, model), group in df.groupby(["base_id", "model"]):
        scores = group[metric_col].values
        mu = scores.mean()
        sigma = scores.std()
        rob = max(0, 1 - sigma / mu) if mu > 0 else 0
        prompt_robustness.append({
            "base_id": base_id, "model": model,
            "robustness": rob, "mean_score": mu, "std_score": sigma,
        })

    rob_df = pd.DataFrame(prompt_robustness)

    fig_scatter = px.scatter(
        rob_df, x="mean_score", y="robustness", color="model",
        hover_data=["base_id"],
        title="Per-Prompt Robustness vs Accuracy",
        labels={"mean_score": f"Mean {metric_label}", "robustness": "Robustness Score"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Cost Analysis ─────────────────────────────────────
    st.header("💰 Cost Analysis")

    cost_data = []
    for model, metrics in summary.items():
        cost_data.append({
            "Model": model,
            "Avg Tokens": metrics.get("avg_tokens", 0),
            "Accuracy": metrics.get("mean_field_accuracy", 0) if task_type == "json_extraction" else metrics.get("mean_answer_score", 0),
            "Robustness": metrics.get("robustness_score", 0),
        })
    cost_df = pd.DataFrame(cost_data)

    fig_cost = px.scatter(
        cost_df, x="Avg Tokens", y="Accuracy", size="Robustness",
        color="Model", text="Model",
        title="Cost Efficiency: Accuracy vs Token Usage (size = Robustness)",
    )
    fig_cost.update_traces(textposition="top center")
    st.plotly_chart(fig_cost, use_container_width=True)

# ── Statistical Significance ──────────────────────────────
if not sig_df.empty:
    st.header("📐 Statistical Significance")
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ── Failure Analysis ──────────────────────────────────────
if not df.empty:
    st.header("⚠️ Failure Analysis")

    selected_model = st.selectbox("Select Model", df["model"].unique())
    model_df = df[df["model"] == selected_model]

    if task_type == "json_extraction":
        failures = model_df[model_df["schema_valid"] == False]
        st.metric("Failed Validations", f"{len(failures)} / {len(model_df)}")
    else:
        failures = model_df[model_df["answer_score"] < 0.3]
        st.metric("Low-Score Responses (<0.3)", f"{len(failures)} / {len(model_df)}")

    if not failures.empty:
        st.subheader("Sample Failures")
        display_cols = ["base_id", "variant_type", "variant_subtype", "response_text"]
        available_cols = [c for c in display_cols if c in failures.columns]
        st.dataframe(failures[available_cols].head(10), use_container_width=True, hide_index=True)

# ── Raw Data Export ───────────────────────────────────────
st.header("📥 Export Data")
col1, col2 = st.columns(2)
with col1:
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button("Download Scored Results (CSV)", csv, f"{task_type}_scored.csv", "text/csv")
with col2:
    if summary:
        st.download_button("Download Summary (JSON)", json.dumps(summary, indent=2), f"{task_type}_summary.json", "application/json")

st.divider()
st.caption("Prompt Crash-Test Lab | CS 599 | Built with Streamlit + Plotly")
