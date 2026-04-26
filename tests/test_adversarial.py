"""Tests for adversarial mutation module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adversarial import (
    generate_adversarial_variants,
    _negate_instruction,
    _swap_format_request,
    _add_distractor,
)


@pytest.fixture
def sample_json_item():
    return {
        "id": "json_001",
        "schema": "ecommerce",
        "document": "The Sony WH-1000XM5 headphones cost $399.99.",
        "ground_truth": {"product_name": "Sony WH-1000XM5", "price": 399.99},
        "base_prompt": "Extract the product information from the following text into JSON format.",
    }


@pytest.fixture
def sample_qa_item():
    return {
        "id": "qa_001",
        "context": "Python was created by Guido van Rossum in 1991.",
        "question": "When was Python created?",
        "ground_truth": {"answer": "Python was created in 1991."},
        "base_prompt": "Answer the question based on the context.",
    }


class TestNegateInstruction:
    def test_negates_extract(self):
        result = _negate_instruction("Extract the data from the text.")
        assert "Do not extract" in result

    def test_negates_answer(self):
        result = _negate_instruction("Answer the question.")
        assert "Do not answer" in result

    def test_no_change_if_no_keyword(self):
        original = "Process the following text."
        result = _negate_instruction(original)
        assert result == original


class TestSwapFormat:
    def test_swaps_json_to_yaml(self):
        result = _swap_format_request("Return as JSON.")
        assert "YAML" in result
        assert "JSON" not in result

    def test_adds_xml_if_no_json(self):
        result = _swap_format_request("Return the result.")
        assert "XML" in result


class TestAddDistractor:
    def test_adds_distractor_text(self):
        prompt = "Extract data.\n\nInput:\nThe Sony headphones cost $399."
        result = _add_distractor(prompt, "The Sony headphones cost $399.")
        assert "UNRELATED NOTE" in result
        assert "weather" in result.lower()


class TestGenerateAdversarialVariants:
    def test_generates_9_variants_json(self, sample_json_item):
        variants = generate_adversarial_variants(sample_json_item, "json_extraction")
        assert len(variants) == 9

    def test_generates_9_variants_qa(self, sample_qa_item):
        variants = generate_adversarial_variants(sample_qa_item, "grounded_qa")
        assert len(variants) == 9

    def test_all_have_adversarial_type(self, sample_json_item):
        variants = generate_adversarial_variants(sample_json_item, "json_extraction")
        for v in variants:
            assert v["type"] == "adversarial"

    def test_subtypes_correct(self, sample_json_item):
        variants = generate_adversarial_variants(sample_json_item, "json_extraction")
        subtypes = [v["subtype"] for v in variants]
        assert "contradiction_1" in subtypes
        assert "role_hijack_1" in subtypes
        assert "negate_instruction" in subtypes
        assert "format_swap" in subtypes
        assert "distractor_injection" in subtypes

    def test_all_have_text(self, sample_json_item):
        variants = generate_adversarial_variants(sample_json_item, "json_extraction")
        for v in variants:
            assert "text" in v
            assert len(v["text"]) > 20
