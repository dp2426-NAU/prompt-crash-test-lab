"""
Pipeline flow diagram using Graphviz.
Exports the end-to-end LLM robustness evaluation pipeline as a PNG.
"""

from __future__ import annotations

import os


def generate_pipeline_diagram(output_path: str) -> str:
    """
    Build and render the pipeline flow diagram.

    Args:
        output_path: File path without extension (e.g. 'outputs/visuals/pipeline').
                     A .png file will be saved at this path.

    Returns:
        Absolute path to the generated PNG.

    Raises:
        ImportError: If the graphviz Python package is not installed.
        graphviz.backend.ExecutableNotFound: If the Graphviz system binary is missing.
    """
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "graphviz package is required. Install with: pip install graphviz\n"
            "Also install the Graphviz system binary: https://graphviz.org/download/"
        ) from exc

    dot = graphviz.Digraph(
        name="prompt_crash_pipeline",
        comment="Prompt Crash Test Lab — Evaluation Pipeline",
        format="png",
    )

    dot.attr(
        rankdir="LR",
        bgcolor="white",
        fontname="Helvetica",
        fontsize="13",
        pad="0.5",
        nodesep="0.6",
        ranksep="0.8",
    )

    # Node defaults
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="12",
        margin="0.2,0.12",
    )

    # ── Nodes ──────────────────────────────────────────────────────────────────

    dot.node(
        "input",
        "Input Layer\n100 Base Prompts\n(2 task types)",
        fillcolor="#d4edda",
        color="#28a745",
        fontcolor="#155724",
    )

    dot.node(
        "variants",
        "Variant Generator\n20 variants / prompt\n(paraphrase · format · role\nconstraint · template)",
        fillcolor="#cce5ff",
        color="#004085",
        fontcolor="#004085",
    )

    dot.node(
        "models",
        "Model Execution Layer\nGPT-4 Turbo  |  Claude 3.5\nGemini 1.5 Pro  |  Llama 70B",
        fillcolor="#fff3cd",
        color="#856404",
        fontcolor="#533f03",
    )

    dot.node(
        "eval",
        "Evaluation Engine\nRobustness · Schema Validity\nSemantic Similarity · Correctness\nCitation Accuracy · Cost",
        fillcolor="#e2d9f3",
        color="#6f42c1",
        fontcolor="#3d1a78",
    )

    dot.node(
        "storage",
        "Storage\nSQLite Cache\nJSONL Results",
        fillcolor="#f8d7da",
        color="#721c24",
        fontcolor="#721c24",
    )

    dot.node(
        "viz",
        "Visualization Layer\nPipeline Diagram\nArchitecture Diagram\nMetric Charts",
        fillcolor="#d1ecf1",
        color="#0c5460",
        fontcolor="#0c5460",
    )

    # ── Edges ──────────────────────────────────────────────────────────────────

    edge_attrs = {"color": "#495057", "fontname": "Helvetica", "fontsize": "10"}

    dot.edge("input", "variants", label=" generate", **edge_attrs)
    dot.edge("variants", "models", label=" execute", **edge_attrs)
    dot.edge("models", "eval", label=" score", **edge_attrs)
    dot.edge("eval", "storage", label=" persist", **edge_attrs)
    dot.edge("storage", "viz", label=" render", **edge_attrs)

    # ── Render ─────────────────────────────────────────────────────────────────

    # graphviz appends '.png' automatically; strip it from output_path if present
    base = output_path.removesuffix(".png")
    rendered = dot.render(filename=base, cleanup=True)
    return os.path.abspath(rendered)
