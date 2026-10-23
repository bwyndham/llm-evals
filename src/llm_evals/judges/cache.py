"""The judge response cache: one SQLite file, keyed by prompt hash.

Separate from the results store so judge answers can be reused across runs
(and across suites) without touching run data. One connection guarded by a
lock, so concurrent checks can share one cache. `JudgeCache(None)` keeps the
cache in memory, for tests and one-off runs.

Every response the provider returned is cached, including ones that don't
parse: the same prompt must always give the same verdict, so a malformed
answer stays a judge error on rerun instead of being re-rolled. Provider
exceptions (timeouts, 5xx) are never cached.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_cache (
    prompt_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    stop_reason TEXT,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class CachedResponse:
    prompt_hash: str
    model: str
    raw_text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    cost_usd: float  # what the original call cost
    created_at: str


class JudgeCache:
    """Thread-safe get/put of judge responses by prompt hash."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            target = ":memory:"
        else:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            target = str(p)
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(target, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def get(self, prompt_hash: str) -> CachedResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT prompt_hash, model, raw_text, input_tokens, output_tokens, stop_reason, "
                "cost_usd, created_at FROM judge_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()
        return None if row is None else CachedResponse(*row)

    def put(
        self,
        prompt_hash: str,
        *,
        model: str,
        raw_text: str,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str | None,
        cost_usd: float,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO judge_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (prompt_hash, model, raw_text, input_tokens, output_tokens, stop_reason, cost_usd, created_at),
            )

    def __len__(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()
        return int(n)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["CachedResponse", "JudgeCache"]
