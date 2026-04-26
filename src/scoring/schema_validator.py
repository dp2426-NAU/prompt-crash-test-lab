"""JSON schema validation for format compliance scoring."""

import json
import re
from pathlib import Path

import jsonschema

from config.settings import SCHEMAS_DIR


def _load_schema(schema_name: str) -> dict:
    """Load a JSON schema by name."""
    if not schema_name:
        raise ValueError("Schema name cannot be empty")
    schema_path = SCHEMAS_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_from_response(response: str) -> dict | None:
    """Extract JSON object from LLM response text.

    Handles cases where JSON is wrapped in markdown code blocks or
    surrounded by extra text.
    """
    # Try parsing the entire response as JSON
    try:
        return json.loads(response.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

    return None


def validate_json_response(response: str, schema_name: str) -> dict:
    """Validate an LLM response against a JSON schema.

    Returns:
        dict with keys:
        - valid: bool - whether the response passes schema validation
        - parsed: dict | None - the parsed JSON (if any)
        - errors: list[str] - validation error messages
        - json_extracted: bool - whether JSON could be extracted at all
    """
    parsed = extract_json_from_response(response)

    if parsed is None:
        return {
            "valid": False,
            "parsed": None,
            "errors": ["Could not extract JSON from response"],
            "json_extracted": False,
        }

    schema = _load_schema(schema_name)
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(parsed))

    return {
        "valid": len(errors) == 0,
        "parsed": parsed,
        "errors": [e.message for e in errors],
        "json_extracted": True,
    }


def compute_field_accuracy(parsed: dict | None, ground_truth: dict) -> dict:
    """Compute per-field accuracy between parsed JSON and ground truth.

    Returns dict with:
    - field_scores: dict of field_name -> 1.0 (match) or 0.0 (mismatch)
    - overall_accuracy: float (0-1)
    """
    if parsed is None:
        return {
            "field_scores": {k: 0.0 for k in ground_truth},
            "overall_accuracy": 0.0,
        }

    field_scores = {}
    for key, expected in ground_truth.items():
        actual = parsed.get(key)
        if actual is None:
            field_scores[key] = 0.0
        elif isinstance(expected, bool):
            # Bool check MUST come before int/float (bool is subclass of int)
            field_scores[key] = 1.0 if actual == expected else 0.0
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Numeric: allow small tolerance
            field_scores[key] = 1.0 if abs(expected - actual) < 0.01 else 0.0
        elif isinstance(expected, list):
            # List: check if sets overlap significantly
            if isinstance(actual, list):
                expected_set = {str(x).lower() for x in expected}
                actual_set = {str(x).lower() for x in actual}
                if expected_set:
                    overlap = len(expected_set & actual_set) / len(expected_set)
                    field_scores[key] = overlap
                else:
                    field_scores[key] = 1.0 if not actual_set else 0.0
            else:
                field_scores[key] = 0.0
        else:
            # String: case-insensitive containment
            expected_str = str(expected).lower().strip()
            actual_str = str(actual).lower().strip()
            if expected_str == actual_str:
                field_scores[key] = 1.0
            elif expected_str in actual_str or actual_str in expected_str:
                field_scores[key] = 0.8
            else:
                field_scores[key] = 0.0

    scores = list(field_scores.values())
    overall = sum(scores) / len(scores) if scores else 0.0

    return {"field_scores": field_scores, "overall_accuracy": overall}
