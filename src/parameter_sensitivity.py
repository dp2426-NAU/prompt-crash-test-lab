"""Parameter sensitivity study (Phase 3, Task 3.3).

Tests how model outputs change with different parameter settings:
- Temperature variation: 0.0, 0.3, 0.7, 1.0
- System prompt impact: with vs without system prompts
- Generates comparison data for analysis
"""

import copy
import json
import time
from pathlib import Path

from tqdm import tqdm

from config.settings import MODELS, CACHE_DB_PATH, RESULTS_DIR, RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_DELAY
from src.cache import ResponseCache
from src.model_clients import get_client


TEMPERATURE_VALUES = [0.0, 0.3, 0.7, 1.0]

SYSTEM_PROMPTS = {
    "none": "",
    "basic": "You are a helpful assistant.",
    "strict": "You are a precise data extraction system. Return only the requested output format with no additional text.",
    "creative": "You are a creative and thorough assistant. Feel free to elaborate on your answers.",
}


def run_temperature_study(
    variants_path: Path,
    model_name: str = "gpt-4-turbo",
    max_variants: int = 20,
    temperatures: list[float] | None = None,
):
    """Run the same prompts at different temperatures and compare outputs.

    Args:
        variants_path: Path to variants JSONL
        model_name: Which model to test
        max_variants: Number of variants to test (for budget control)
        temperatures: List of temperature values to test
    """
    if temperatures is None:
        temperatures = TEMPERATURE_VALUES

    cache = ResponseCache(CACHE_DB_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(variants_path, "r", encoding="utf-8") as f:
        variants = [json.loads(line) for line in f if line.strip()][:max_variants]

    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return []

    model_config = MODELS[model_name]
    client = get_client(model_config["provider"], model_id=model_config["model_id"])

    results = []
    for temp in temperatures:
        print(f"\nRunning {model_name} at temperature={temp}...")
        for variant in tqdm(variants, desc=f"temp={temp}"):
            prompt_text = variant["text"]
            params = {"temperature": temp, "max_tokens": model_config["max_tokens"]}

            cached = cache.get(prompt_text, f"{model_name}_t{temp}", params)
            if cached:
                result = {
                    **variant,
                    "model": model_name,
                    "temperature": temp,
                    "response": cached["response"],
                    "tokens_used": cached["tokens_used"],
                    "cached": True,
                }
                results.append(result)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    llm_response = client.generate(prompt_text, temperature=temp, max_tokens=model_config["max_tokens"])
                    cache.put(prompt_text, f"{model_name}_t{temp}", params, llm_response.text,
                              llm_response.tokens_input + llm_response.tokens_output)

                    result = {
                        **variant,
                        "model": model_name,
                        "temperature": temp,
                        "response": llm_response.text,
                        "tokens_used": llm_response.tokens_input + llm_response.tokens_output,
                        "cached": False,
                    }
                    results.append(result)
                    time.sleep(RATE_LIMIT_DELAY)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        results.append({
                            **variant, "model": model_name, "temperature": temp,
                            "response": f"ERROR: {e}", "tokens_used": 0, "error": True,
                        })

    # Save results
    out_path = RESULTS_DIR / f"temperature_study_{model_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nTemperature study saved to {out_path}")
    print(f"Total results: {len(results)} ({len(variants)} variants x {len(temperatures)} temperatures)")
    return results


def run_system_prompt_study(
    variants_path: Path,
    model_name: str = "gpt-4-turbo",
    max_variants: int = 20,
    system_prompts: dict[str, str] | None = None,
):
    """Run the same prompts with different system prompts and compare.

    Args:
        variants_path: Path to variants JSONL
        model_name: Which model to test
        max_variants: Number of variants to test
        system_prompts: Dict of name -> system prompt text
    """
    if system_prompts is None:
        system_prompts = SYSTEM_PROMPTS

    cache = ResponseCache(CACHE_DB_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(variants_path, "r", encoding="utf-8") as f:
        variants = [json.loads(line) for line in f if line.strip()][:max_variants]

    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return []

    model_config = MODELS[model_name]
    client = get_client(model_config["provider"], model_id=model_config["model_id"])

    results = []
    for sp_name, sp_text in system_prompts.items():
        print(f"\nRunning {model_name} with system_prompt='{sp_name}'...")
        for variant in tqdm(variants, desc=f"sys={sp_name}"):
            prompt_text = variant["text"]
            params = {"temperature": 0.0, "max_tokens": model_config["max_tokens"], "system_prompt": sp_name}

            cached = cache.get(prompt_text, f"{model_name}_sp_{sp_name}", params)
            if cached:
                result = {
                    **variant,
                    "model": model_name,
                    "system_prompt_name": sp_name,
                    "system_prompt_text": sp_text,
                    "response": cached["response"],
                    "tokens_used": cached["tokens_used"],
                    "cached": True,
                }
                results.append(result)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    llm_response = client.generate(
                        prompt_text,
                        system_prompt=sp_text,
                        temperature=0.0,
                        max_tokens=model_config["max_tokens"],
                    )
                    cache.put(prompt_text, f"{model_name}_sp_{sp_name}", params, llm_response.text,
                              llm_response.tokens_input + llm_response.tokens_output)

                    result = {
                        **variant,
                        "model": model_name,
                        "system_prompt_name": sp_name,
                        "system_prompt_text": sp_text,
                        "response": llm_response.text,
                        "tokens_used": llm_response.tokens_input + llm_response.tokens_output,
                        "cached": False,
                    }
                    results.append(result)
                    time.sleep(RATE_LIMIT_DELAY)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        results.append({
                            **variant, "model": model_name, "system_prompt_name": sp_name,
                            "response": f"ERROR: {e}", "tokens_used": 0, "error": True,
                        })

    # Save results
    out_path = RESULTS_DIR / f"system_prompt_study_{model_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSystem prompt study saved to {out_path}")
    print(f"Total results: {len(results)} ({len(variants)} variants x {len(system_prompts)} system prompts)")
    return results
