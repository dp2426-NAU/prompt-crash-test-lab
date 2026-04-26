"""Tests for scoring modules."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.schema_validator import (
    extract_json_from_response,
    validate_json_response,
    compute_field_accuracy,
)
from src.scoring.answer_correctness import (
    keyword_overlap,
    normalize_text,
    check_citation_present,
)
from src.scoring.robustness import compute_robustness_score


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json_from_response('{"name": "test", "value": 42}')
        assert result == {"name": "test", "value": 42}

    def test_markdown_code_block(self):
        response = '```json\n{"name": "test"}\n```'
        result = extract_json_from_response(response)
        assert result == {"name": "test"}

    def test_json_with_surrounding_text(self):
        response = 'Here is the result:\n{"name": "test"}\nThat is the answer.'
        result = extract_json_from_response(response)
        assert result == {"name": "test"}

    def test_invalid_json(self):
        result = extract_json_from_response("This is not JSON at all.")
        assert result is None

    def test_empty_response(self):
        result = extract_json_from_response("")
        assert result is None


class TestValidateJsonResponse:
    def test_valid_ecommerce(self):
        response = json.dumps({
            "product_name": "Test",
            "price": 29.99,
            "category": "Electronics",
            "brand": "TestBrand",
            "in_stock": True,
            "rating": 4.5,
        })
        result = validate_json_response(response, "ecommerce")
        assert result["valid"] is True
        assert result["json_extracted"] is True

    def test_missing_required_field(self):
        response = json.dumps({"product_name": "Test"})
        result = validate_json_response(response, "ecommerce")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_invalid_type(self):
        response = json.dumps({
            "product_name": "Test",
            "price": "not a number",
            "category": "Electronics",
        })
        result = validate_json_response(response, "ecommerce")
        assert result["valid"] is False


class TestFieldAccuracy:
    def test_perfect_match(self):
        parsed = {"name": "Sony", "price": 399.99, "in_stock": True}
        ground = {"name": "Sony", "price": 399.99, "in_stock": True}
        result = compute_field_accuracy(parsed, ground)
        assert result["overall_accuracy"] == 1.0

    def test_no_match(self):
        parsed = {"name": "Wrong", "price": 0.0, "in_stock": False}
        ground = {"name": "Sony", "price": 399.99, "in_stock": True}
        result = compute_field_accuracy(parsed, ground)
        assert result["overall_accuracy"] < 0.5

    def test_none_parsed(self):
        result = compute_field_accuracy(None, {"name": "Test"})
        assert result["overall_accuracy"] == 0.0

    def test_numeric_tolerance(self):
        parsed = {"price": 399.99}
        ground = {"price": 399.99}
        result = compute_field_accuracy(parsed, ground)
        assert result["field_scores"]["price"] == 1.0


class TestKeywordOverlap:
    def test_perfect_overlap(self):
        score = keyword_overlap("Python was created in 1991", "Python was created in 1991")
        assert score > 0.9

    def test_no_overlap(self):
        score = keyword_overlap("cats and dogs", "quantum physics equations")
        assert score == 0.0

    def test_partial_overlap(self):
        score = keyword_overlap("Python was released in 1991 by Guido", "Python released 1991")
        assert 0.5 < score <= 1.0


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_removes_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"


class TestCitationCheck:
    def test_finds_quoted_text(self):
        response = 'The answer is 1991. Quote: "Python was created by Guido van Rossum"'
        context = "Python was created by Guido van Rossum and first released in 1991."
        result = check_citation_present(response, context)
        assert result["has_quote"] is True

    def test_no_citation(self):
        response = "The answer is 1991."
        context = "Python was created by Guido van Rossum."
        result = check_citation_present(response, context)
        assert result["has_quote"] is False


class TestRobustnessScore:
    def test_perfect_robustness(self):
        score = compute_robustness_score([0.9, 0.9, 0.9, 0.9])
        assert score == 1.0

    def test_high_variance(self):
        score = compute_robustness_score([0.1, 0.9, 0.1, 0.9])
        assert score < 0.5

    def test_zero_scores(self):
        score = compute_robustness_score([0.0, 0.0, 0.0])
        assert score == 0.0

    def test_empty_list(self):
        score = compute_robustness_score([])
        assert score == 0.0

    def test_single_value(self):
        score = compute_robustness_score([0.8])
        assert score == 1.0
