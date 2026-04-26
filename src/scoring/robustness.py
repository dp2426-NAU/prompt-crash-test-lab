"""Robustness score computation - the core metric of this project.

Robustness = 1 - (sigma_variants / mu_accuracy)
Higher score = more consistent behavior across prompt variants.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema_validator import validate_json_response, compute_field_accuracy, extract_json_from_response
from .semantic_similarity import compute_semantic_consistency, compute_similarity_to_reference
from .answer_correctness import compute_answer_correctness


def compute_robustness_score(accuracies: list[float]) -> float:
    """Compute robustness score from a list of accuracy values.

    Robustness = 1 - (std / mean) when mean > 0, else 0.
    """
    if not accuracies:
        return 0.0
    mu = np.mean(accuracies)
    if mu == 0:
        return 0.0
    sigma = np.std(accuracies)
    return float(max(0.0, 1.0 - (sigma / mu)))


def compute_all_metrics(results_path: Path, task_type: str = "json_extraction") -> pd.DataFrame:
    """Compute all evaluation metrics for a results file.

    Args:
        results_path: Path to JSONL results file
        task_type: 'json_extraction' or 'grounded_qa'

    Returns:
        DataFrame with per-variant scores and aggregated metrics
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    if not results:
        return pd.DataFrame()

    scored_results = []

    for r in results:
        response = r.get("response", "")
        model = r.get("model", "unknown")
        base_id = r.get("base_id", "")
        variant_type = r.get("type", "")
        variant_subtype = r.get("subtype", "")

        score_entry = {
            "base_id": base_id,
            "model": model,
            "variant_type": variant_type,
            "variant_subtype": variant_subtype,
            "tokens_used": r.get("tokens_used", 0),
            "latency_ms": r.get("latency_ms", 0),
            "is_error": r.get("error", False),
        }

        if task_type == "json_extraction":
            schema_name = r.get("schema", "")
            ground_truth = r.get("ground_truth", {})

            # Schema validation
            if not schema_name:
                validation = {"valid": False, "parsed": extract_json_from_response(response),
                              "errors": ["No schema specified"], "json_extracted": extract_json_from_response(response) is not None}
            else:
                validation = validate_json_response(response, schema_name)
            score_entry["schema_valid"] = validation["valid"]
            score_entry["json_extracted"] = validation["json_extracted"]
            score_entry["validation_errors"] = len(validation["errors"])

            # Field accuracy
            field_acc = compute_field_accuracy(validation["parsed"], ground_truth)
            score_entry["field_accuracy"] = field_acc["overall_accuracy"]
            score_entry["field_scores"] = json.dumps(field_acc["field_scores"])

        elif task_type == "grounded_qa":
            ground_truth = r.get("ground_truth", {})
            context = r.get("context", "")

            # Answer correctness
            correctness = compute_answer_correctness(response, ground_truth, context)
            score_entry["semantic_similarity"] = correctness["semantic_similarity"]
            score_entry["keyword_overlap"] = correctness["keyword_overlap"]
            score_entry["citation_present"] = correctness["citation_present"]
            score_entry["citation_accuracy"] = correctness["citation_accuracy"]
            score_entry["answer_score"] = correctness["overall_score"]

        score_entry["response_text"] = response[:500]  # truncate for storage
        scored_results.append(score_entry)

    df = pd.DataFrame(scored_results)
    return df


def aggregate_metrics(df: pd.DataFrame, task_type: str = "json_extraction") -> dict:
    """Aggregate per-variant metrics into summary statistics.

    Returns nested dict: model -> metric -> value
    """
    if df.empty:
        return {}

    summary = {}

    for model, group in df.groupby("model"):
        model_summary = {
            "total_variants": len(group),
            "error_rate": float(group["is_error"].mean()),
            "avg_tokens": float(group["tokens_used"].mean()),
            "avg_latency_ms": float(group["latency_ms"].mean()),
        }

        if task_type == "json_extraction":
            accuracies = group["field_accuracy"].tolist()
            model_summary["format_compliance"] = float(group["schema_valid"].mean())
            model_summary["json_extraction_rate"] = float(group["json_extracted"].mean())
            model_summary["mean_field_accuracy"] = float(np.mean(accuracies))
            model_summary["std_field_accuracy"] = float(np.std(accuracies))
            model_summary["robustness_score"] = compute_robustness_score(accuracies)

            # Per variant-type breakdown
            for vtype, vgroup in group.groupby("variant_type"):
                vacc = vgroup["field_accuracy"].tolist()
                model_summary[f"accuracy_{vtype}"] = float(np.mean(vacc))
                model_summary[f"compliance_{vtype}"] = float(vgroup["schema_valid"].mean())

        elif task_type == "grounded_qa":
            scores = group["answer_score"].tolist()
            model_summary["mean_answer_score"] = float(np.mean(scores))
            model_summary["std_answer_score"] = float(np.std(scores))
            model_summary["citation_rate"] = float(group["citation_present"].mean())
            model_summary["mean_semantic_sim"] = float(group["semantic_similarity"].mean())
            model_summary["robustness_score"] = compute_robustness_score(scores)

            for vtype, vgroup in group.groupby("variant_type"):
                vscores = vgroup["answer_score"].tolist()
                model_summary[f"score_{vtype}"] = float(np.mean(vscores))

        summary[model] = model_summary

    return summary


def compute_semantic_consistency_per_prompt(df: pd.DataFrame) -> dict:
    """Compute semantic consistency for each (base_id, model) group.

    Returns dict: (base_id, model) -> consistency metrics
    """
    consistency = {}

    for (base_id, model), group in df.groupby(["base_id", "model"]):
        responses = group["response_text"].tolist()
        metrics = compute_semantic_consistency(responses)
        consistency[(base_id, model)] = metrics

    return consistency
