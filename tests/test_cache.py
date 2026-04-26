"""Tests for the response cache module."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import ResponseCache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "test_cache.db")


class TestResponseCache:
    def test_put_and_get(self, cache):
        cache.put("test prompt", "gpt-4", {"temp": 0.0}, "test response", 100)
        result = cache.get("test prompt", "gpt-4", {"temp": 0.0})
        assert result is not None
        assert result["response"] == "test response"
        assert result["tokens_used"] == 100
        assert result["cached"] is True

    def test_cache_miss(self, cache):
        result = cache.get("nonexistent", "gpt-4", {"temp": 0.0})
        assert result is None

    def test_different_params_different_key(self, cache):
        cache.put("prompt", "gpt-4", {"temp": 0.0}, "response1", 50)
        cache.put("prompt", "gpt-4", {"temp": 1.0}, "response2", 60)

        r1 = cache.get("prompt", "gpt-4", {"temp": 0.0})
        r2 = cache.get("prompt", "gpt-4", {"temp": 1.0})
        assert r1["response"] == "response1"
        assert r2["response"] == "response2"

    def test_different_models_different_key(self, cache):
        cache.put("prompt", "gpt-4", {"temp": 0.0}, "gpt response", 50)
        cache.put("prompt", "claude", {"temp": 0.0}, "claude response", 60)

        r1 = cache.get("prompt", "gpt-4", {"temp": 0.0})
        r2 = cache.get("prompt", "claude", {"temp": 0.0})
        assert r1["response"] == "gpt response"
        assert r2["response"] == "claude response"

    def test_overwrite(self, cache):
        cache.put("prompt", "gpt-4", {"temp": 0.0}, "old response", 50)
        cache.put("prompt", "gpt-4", {"temp": 0.0}, "new response", 70)
        result = cache.get("prompt", "gpt-4", {"temp": 0.0})
        assert result["response"] == "new response"
        assert result["tokens_used"] == 70

    def test_stats(self, cache):
        cache.put("p1", "gpt-4", {}, "r1", 100)
        cache.put("p2", "gpt-4", {}, "r2", 200)
        cache.put("p3", "claude", {}, "r3", 150)

        stats = cache.stats()
        assert stats["total_cached"] == 3
        assert "gpt-4" in stats["by_model"]
        assert stats["by_model"]["gpt-4"]["count"] == 2
        assert stats["by_model"]["claude"]["count"] == 1
