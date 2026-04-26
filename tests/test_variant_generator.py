"""Tests for variant generator module."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variant_generator import (
    generate_json_variants,
    generate_qa_variants,
    _paraphrase_prompts,
)


@pytest.fixture
def sample_json_item():
    return {
        "id": "json_001",
        "schema": "ecommerce",
        "document": "The Sony WH-1000XM5 wireless headphones are priced at $399.99.",
        "ground_truth": {"product_name": "Sony WH-1000XM5", "price": 399.99},
        "base_prompt": "Extract the product information from the following text into JSON format.",
    }


@pytest.fixture
def sample_qa_item():
    return {
        "id": "qa_001",
        "context": "Python was created by Guido van Rossum and first released in 1991.",
        "question": "When was Python first released?",
        "ground_truth": {"answer": "Python was first released in 1991."},
        "base_prompt": "Based on the provided context, answer the following question.",
    }


class TestParaphrasePrompts:
    def test_generates_correct_count(self):
        results = _paraphrase_prompts("Extract data from text.", n=5)
        assert len(results) == 5

    def test_all_have_type_paraphrase(self):
        results = _paraphrase_prompts("Extract data from text.", n=5)
        for r in results:
            assert r["type"] == "paraphrase"

    def test_all_have_variant_id(self):
        results = _paraphrase_prompts("Extract data.", n=3)
        ids = [r["variant_id"] for r in results]
        assert len(set(ids)) == 3  # all unique

    def test_all_contain_base_prompt_intent(self):
        results = _paraphrase_prompts("Extract data from text.", n=5)
        for r in results:
            assert "extract data from text" in r["text"].lower()


class TestGenerateJsonVariants:
    def test_generates_20_variants(self, sample_json_item):
        variants = generate_json_variants(sample_json_item)
        assert len(variants) == 20

    def test_variant_type_distribution(self, sample_json_item):
        variants = generate_json_variants(sample_json_item)
        types = [v["type"] for v in variants]
        assert types.count("paraphrase") == 5
        assert types.count("format") == 4
        assert types.count("role") == 3
        assert types.count("constraint") == 3
        assert types.count("template") == 5

    def test_all_variants_have_text(self, sample_json_item):
        variants = generate_json_variants(sample_json_item)
        for v in variants:
            assert "text" in v
            assert len(v["text"]) > 10

    def test_format_variants_contain_document(self, sample_json_item):
        variants = generate_json_variants(sample_json_item)
        format_variants = [v for v in variants if v["type"] == "format"]
        for v in format_variants:
            assert sample_json_item["document"] in v["text"]


class TestGenerateQaVariants:
    def test_generates_20_variants(self, sample_qa_item):
        variants = generate_qa_variants(sample_qa_item)
        assert len(variants) == 20

    def test_variant_type_distribution(self, sample_qa_item):
        variants = generate_qa_variants(sample_qa_item)
        types = [v["type"] for v in variants]
        assert types.count("paraphrase") == 5
        assert types.count("format") == 4
        assert types.count("role") == 3
        assert types.count("constraint") == 3
        assert types.count("template") == 5

    def test_variants_contain_context(self, sample_qa_item):
        variants = generate_qa_variants(sample_qa_item)
        for v in variants:
            assert sample_qa_item["context"] in v["text"] or "context" in v["text"].lower()
