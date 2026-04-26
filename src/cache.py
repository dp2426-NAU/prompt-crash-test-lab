"""SQLite-based response cache to avoid duplicate API calls."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class ResponseCache:
    """Cache LLM responses keyed by hash(prompt + model + parameters)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON cache(model)
            """)

    @staticmethod
    def _make_key(prompt: str, model: str, parameters: dict) -> str:
        raw = json.dumps({"prompt": prompt, "model": model, "params": parameters}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model: str, parameters: dict) -> dict | None:
        key = self._make_key(prompt, model, parameters)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response, tokens_used, timestamp FROM cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "response": row[0],
            "tokens_used": row[1],
            "timestamp": row[2],
            "cached": True,
        }

    def put(self, prompt: str, model: str, parameters: dict, response: str, tokens_used: int = 0):
        key = self._make_key(prompt, model, parameters)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache
                   (cache_key, prompt, model, parameters, response, tokens_used, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, prompt, model, json.dumps(parameters, sort_keys=True), response, tokens_used, time.time()),
            )

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            by_model = conn.execute(
                "SELECT model, COUNT(*), SUM(tokens_used) FROM cache GROUP BY model"
            ).fetchall()
        return {
            "total_cached": total,
            "by_model": {row[0]: {"count": row[1], "tokens": row[2]} for row in by_model},
        }
