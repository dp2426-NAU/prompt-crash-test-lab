"""Batch execution engine for running prompt variants across models.

Features:
- Parallel model execution with rate limiting
- Response caching via SQLite
- Progress tracking and cost estimation
- Error recovery with retries
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from config.settings import MODELS, RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_DELAY, CACHE_DB_PATH, RESULTS_DIR
from src.cache import ResponseCache
from src.model_clients import get_client


def load_variants(variants_path: Path) -> list[dict]:
    """Load variant prompts from a JSONL file."""
    variants = []
    with open(variants_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                variants.append(json.loads(line))
    return variants


def estimate_cost(num_prompts: int, models: dict) -> dict:
    """Estimate API costs before execution."""
    # Approximate costs per 1K tokens (input + output)
    cost_per_1k = {
        "gpt-4-turbo": 0.02,
        "claude-3.5-sonnet": 0.009,
        "gemini-1.5-pro": 0.003,
        "llama-3.1-70b": 0.001,
    }
    avg_tokens_per_prompt = 500  # conservative estimate

    estimates = {}
    total = 0.0
    for model_name in models:
        rate = cost_per_1k.get(model_name, 0.01)
        cost = num_prompts * avg_tokens_per_prompt / 1000 * rate
        estimates[model_name] = round(cost, 2)
        total += cost

    estimates["total"] = round(total, 2)
    return estimates


def run_batch(
    variants_path: Path,
    model_names: list[str] | None = None,
    max_variants: int | None = None,
    task_type: str = "json_extraction",
):
    """Execute all variants across specified models.

    Args:
        variants_path: Path to JSONL file with variants
        model_names: List of model names to evaluate (defaults to all)
        max_variants: Limit number of variants (for testing/budget control)
        task_type: 'json_extraction' or 'grounded_qa'
    """
    cache = ResponseCache(CACHE_DB_PATH)
    variants = load_variants(variants_path)

    if max_variants:
        variants = variants[:max_variants]

    if model_names is None:
        model_names = list(MODELS.keys())

    # Cost estimation
    cost_est = estimate_cost(len(variants), {m: MODELS[m] for m in model_names if m in MODELS})
    print(f"Estimated cost: ${cost_est['total']:.2f}")
    print(f"Variants to process: {len(variants)} x {len(model_names)} models = {len(variants) * len(model_names)} calls")

    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        if model_name not in MODELS:
            print(f"Skipping unknown model: {model_name}")
            continue

        model_config = MODELS[model_name]
        client = get_client(
            model_config["provider"],
            model_id=model_config["model_id"],
        )

        print(f"\nRunning {model_name}...")
        cache_hits = 0
        api_calls = 0
        errors = 0

        for variant in tqdm(variants, desc=model_name):
            prompt_text = variant["text"]
            params = {"temperature": model_config["temperature"], "max_tokens": model_config["max_tokens"]}

            # Check cache
            cached = cache.get(prompt_text, model_name, params)
            if cached:
                cache_hits += 1
                result = {
                    **variant,
                    "model": model_name,
                    "response": cached["response"],
                    "tokens_used": cached["tokens_used"],
                    "cached": True,
                    "latency_ms": 0,
                }
                results.append(result)
                continue

            # Call API with retry
            for attempt in range(MAX_RETRIES):
                try:
                    llm_response = client.generate(prompt_text, **params)
                    cache.put(prompt_text, model_name, params, llm_response.text,
                              llm_response.tokens_input + llm_response.tokens_output)

                    result = {
                        **variant,
                        "model": model_name,
                        "response": llm_response.text,
                        "tokens_used": llm_response.tokens_input + llm_response.tokens_output,
                        "cached": False,
                        "latency_ms": llm_response.latency_ms,
                    }
                    results.append(result)
                    api_calls += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {model_name}: {e}")
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"  FAILED after {MAX_RETRIES} attempts: {e}")
                        errors += 1
                        result = {
                            **variant,
                            "model": model_name,
                            "response": f"ERROR: {str(e)}",
                            "tokens_used": 0,
                            "cached": False,
                            "latency_ms": 0,
                            "error": True,
                        }
                        results.append(result)

        print(f"  {model_name}: {api_calls} API calls, {cache_hits} cache hits, {errors} errors")

    # Save results
    out_path = RESULTS_DIR / f"{task_type}_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nResults saved to {out_path}")
    print(f"Cache stats: {cache.stats()}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch evaluation")
    parser.add_argument("--task", choices=["json_extraction", "grounded_qa"], required=True)
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--max-variants", type=int, default=None, help="Limit variants for testing")
    args = parser.parse_args()

    from config.settings import VARIANTS_DIR

    variants_file = VARIANTS_DIR / f"{args.task}_variants.jsonl"
    if not variants_file.exists():
        print(f"Variants file not found: {variants_file}")
        print("Run variant_generator.py first.")
    else:
        run_batch(variants_file, model_names=args.models, max_variants=args.max_variants, task_type=args.task)
