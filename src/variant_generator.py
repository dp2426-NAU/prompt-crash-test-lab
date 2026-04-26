"""Automated prompt variant generator.

Generates 20 variants per base prompt across 5 categories:
- Paraphrases (5): LLM-powered semantic rewording
- Format changes (4): markdown, plaintext, numbered, XML
- Role modifications (3): expert, assistant, teacher
- Constraint additions (3): concise, detailed, simple language
- Template variations (5): zero-shot, few-shot, CoT, structured, step-by-step
"""

import json
import uuid
from pathlib import Path

from tqdm import tqdm


# ── Format Templates ──────────────────────────────────────

FORMAT_TEMPLATES = {
    "markdown": "# Task\n\n{prompt}\n\n## Input\n\n{document}\n\n## Output Format\n\nReturn valid JSON matching the schema.",
    "plaintext": "{prompt}\n\nInput:\n{document}\n\nPlease return JSON output.",
    "numbered": "Instructions:\n1. Read the following input carefully.\n2. {prompt}\n3. Return the result as valid JSON.\n\nInput:\n{document}",
    "xml": "<task>\n  <instruction>{prompt}</instruction>\n  <input>{document}</input>\n  <output_format>JSON</output_format>\n</task>",
}

# ── Role Templates ────────────────────────────────────────

ROLE_TEMPLATES = {
    "expert": "You are a domain expert with 20 years of experience in data extraction. {prompt}",
    "assistant": "You are a helpful AI assistant. Your task is to carefully analyze the input and {prompt}",
    "teacher": "You are a meticulous teacher grading student work. Apply the same rigor to: {prompt}",
}

# ── Constraint Templates ──────────────────────────────────

CONSTRAINT_TEMPLATES = {
    "concise": "Be concise and return only the JSON output with no explanation. {prompt}",
    "detailed": "Think step by step, then provide the final answer. {prompt} Explain your reasoning briefly before the JSON output.",
    "simple": "Use simple, clear language. {prompt} Make sure the output is easy to understand.",
}

# ── Template/Strategy Variations ──────────────────────────

STRATEGY_TEMPLATES = {
    "zero_shot": "{prompt}\n\nInput:\n{document}",
    "few_shot": (
        "Here are examples of the expected output format:\n\n"
        "Example 1 Input: 'The XYZ Widget costs $29.99 in the Tools category by WidgetCo. Rated 4.2 stars, in stock.'\n"
        "Example 1 Output: {{\"product_name\": \"XYZ Widget\", \"price\": 29.99, \"category\": \"Tools\", \"brand\": \"WidgetCo\", \"in_stock\": true, \"rating\": 4.2}}\n\n"
        "Now perform the same task:\n{prompt}\n\nInput:\n{document}"
    ),
    "chain_of_thought": (
        "Let's approach this step by step:\n"
        "1. First, read the input carefully\n"
        "2. Identify each field required by the schema\n"
        "3. Extract the value for each field\n"
        "4. Format as valid JSON\n\n"
        "{prompt}\n\nInput:\n{document}"
    ),
    "structured": (
        "TASK: {prompt}\n"
        "INPUT: {document}\n"
        "SCHEMA: Follow the provided JSON schema exactly.\n"
        "OUTPUT: Return only valid JSON."
    ),
    "step_by_step": (
        "Please complete the following task:\n\n"
        "Step 1: Read the input text below.\n"
        "Step 2: {prompt}\n"
        "Step 3: Validate your output matches the expected schema.\n"
        "Step 4: Return only the final JSON.\n\n"
        "Input text:\n{document}"
    ),
}

# ── Q&A-specific templates ────────────────────────────────

QA_FORMAT_TEMPLATES = {
    "markdown": "# Question Answering Task\n\n## Context\n{context}\n\n## Question\n{question}\n\n## Instructions\n{prompt}",
    "plaintext": "Context:\n{context}\n\nQuestion: {question}\n\n{prompt}",
    "numbered": "1. Read the following context.\n2. Answer the question based ONLY on the context.\n3. Include a supporting quote.\n\nContext:\n{context}\n\nQuestion: {question}",
    "xml": "<task>\n  <context>{context}</context>\n  <question>{question}</question>\n  <instruction>{prompt}</instruction>\n</task>",
}

QA_ROLE_TEMPLATES = {
    "expert": "You are a research analyst. Answer precisely using only the provided context. {prompt}",
    "assistant": "You are a careful reading comprehension assistant. {prompt}",
    "teacher": "You are a teacher evaluating whether the answer is supported by the text. {prompt}",
}

QA_CONSTRAINT_TEMPLATES = {
    "concise": "Answer in one or two sentences only. {prompt}",
    "detailed": "Provide a thorough answer with a direct quote from the context. {prompt}",
    "simple": "Answer in simple language a high school student would understand. {prompt}",
}

QA_STRATEGY_TEMPLATES = {
    "zero_shot": "Context:\n{context}\n\nQuestion: {question}\n\n{prompt}",
    "few_shot": (
        "Example:\nContext: 'Water boils at 100 degrees Celsius at standard pressure.'\n"
        "Question: 'At what temperature does water boil?'\n"
        "Answer: 'Water boils at 100 degrees Celsius.' Quote: 'Water boils at 100 degrees Celsius at standard pressure.'\n\n"
        "Now answer this:\nContext:\n{context}\n\nQuestion: {question}\n\n{prompt}"
    ),
    "chain_of_thought": (
        "Think step by step:\n"
        "1. Read the context carefully\n"
        "2. Identify relevant information for the question\n"
        "3. Formulate the answer using only context information\n"
        "4. Find a supporting quote\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\n{prompt}"
    ),
    "structured": (
        "CONTEXT: {context}\n"
        "QUESTION: {question}\n"
        "TASK: {prompt}\n"
        "FORMAT: Answer followed by supporting quote."
    ),
    "step_by_step": (
        "Step 1: Read the context.\n"
        "Step 2: Find information relevant to the question.\n"
        "Step 3: Write your answer using only context facts.\n"
        "Step 4: Include a direct quote.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
}


def _paraphrase_prompts(base_prompt: str, n: int = 5) -> list[dict]:
    """Generate paraphrased variants using rule-based transformations.

    Uses deterministic rewriting rather than LLM calls to avoid extra API costs.
    """
    paraphrases = [
        f"Please {base_prompt[0].lower()}{base_prompt[1:]}",
        f"Your task is to {base_prompt[0].lower()}{base_prompt[1:]}",
        f"I need you to {base_prompt[0].lower()}{base_prompt[1:]}",
        f"Carefully {base_prompt[0].lower()}{base_prompt[1:]} Be precise.",
        f"Given the input below, {base_prompt[0].lower()}{base_prompt[1:]} Return only the result.",
    ]
    return [{"type": "paraphrase", "variant_id": str(uuid.uuid4())[:8], "text": p} for p in paraphrases[:n]]


def generate_json_variants(item: dict) -> list[dict]:
    """Generate all 20 variants for a JSON extraction prompt."""
    base_prompt = item["base_prompt"]
    document = item["document"]
    variants = []

    # Paraphrases (5)
    variants.extend(_paraphrase_prompts(base_prompt))

    # Format changes (4)
    for fmt_name, template in FORMAT_TEMPLATES.items():
        text = template.format(prompt=base_prompt, document=document)
        variants.append({"type": "format", "subtype": fmt_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Role modifications (3)
    for role_name, template in ROLE_TEMPLATES.items():
        text = template.format(prompt=base_prompt) + f"\n\nInput:\n{document}"
        variants.append({"type": "role", "subtype": role_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Constraint additions (3)
    for con_name, template in CONSTRAINT_TEMPLATES.items():
        text = template.format(prompt=base_prompt) + f"\n\nInput:\n{document}"
        variants.append({"type": "constraint", "subtype": con_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Template variations (5)
    for strat_name, template in STRATEGY_TEMPLATES.items():
        text = template.format(prompt=base_prompt, document=document)
        variants.append({"type": "template", "subtype": strat_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    return variants


def generate_qa_variants(item: dict) -> list[dict]:
    """Generate all 20 variants for a grounded Q&A prompt."""
    base_prompt = item["base_prompt"]
    context = item["context"]
    question = item["question"]
    variants = []

    # Paraphrases (5)
    variants.extend(_paraphrase_prompts(base_prompt))
    # Attach context/question to paraphrases
    for v in variants:
        v["text"] = f"Context:\n{context}\n\nQuestion: {question}\n\n{v['text']}"

    # Format changes (4)
    for fmt_name, template in QA_FORMAT_TEMPLATES.items():
        text = template.format(prompt=base_prompt, context=context, question=question)
        variants.append({"type": "format", "subtype": fmt_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Role modifications (3)
    for role_name, template in QA_ROLE_TEMPLATES.items():
        text = template.format(prompt=base_prompt) + f"\n\nContext:\n{context}\n\nQuestion: {question}"
        variants.append({"type": "role", "subtype": role_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Constraint additions (3)
    for con_name, template in QA_CONSTRAINT_TEMPLATES.items():
        text = template.format(prompt=base_prompt) + f"\n\nContext:\n{context}\n\nQuestion: {question}"
        variants.append({"type": "constraint", "subtype": con_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    # Template variations (5)
    for strat_name, template in QA_STRATEGY_TEMPLATES.items():
        text = template.format(prompt=base_prompt, context=context, question=question)
        variants.append({"type": "template", "subtype": strat_name, "variant_id": str(uuid.uuid4())[:8], "text": text})

    return variants


def generate_all_variants(base_prompts_dir: Path, output_dir: Path):
    """Generate variants for all base prompts and save to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON extraction variants
    json_path = base_prompts_dir / "json_extraction.jsonl"
    if json_path.exists():
        json_variants = []
        with open(json_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        for item in tqdm(items, desc="Generating JSON extraction variants"):
            variants = generate_json_variants(item)
            for v in variants:
                v["base_id"] = item["id"]
                v["schema"] = item["schema"]
                v["ground_truth"] = item["ground_truth"]
                v["document"] = item["document"]
            json_variants.extend(variants)

        out_path = output_dir / "json_extraction_variants.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for v in json_variants:
                f.write(json.dumps(v) + "\n")
        print(f"Generated {len(json_variants)} JSON extraction variants -> {out_path}")

    # Q&A variants
    qa_path = base_prompts_dir / "grounded_qa.jsonl"
    if qa_path.exists():
        qa_variants = []
        with open(qa_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        for item in tqdm(items, desc="Generating Q&A variants"):
            variants = generate_qa_variants(item)
            for v in variants:
                v["base_id"] = item["id"]
                v["context"] = item["context"]
                v["question"] = item["question"]
                v["ground_truth"] = item["ground_truth"]
            qa_variants.extend(variants)

        out_path = output_dir / "grounded_qa_variants.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for v in qa_variants:
                f.write(json.dumps(v) + "\n")
        print(f"Generated {len(qa_variants)} Q&A variants -> {out_path}")


if __name__ == "__main__":
    from config.settings import BASE_PROMPTS_DIR, VARIANTS_DIR
    generate_all_variants(BASE_PROMPTS_DIR, VARIANTS_DIR)
