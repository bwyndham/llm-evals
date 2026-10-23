"""The replay tool layer: `search` and `fetch` served from a case's frozen corpus.

A `ToolSession` is built per sample from `case.fixtures["docs"]` and records
every call into `calls` (`{"name", "input", "doc_ids"}`, plus `"error"` when
the call failed). That list becomes `result.trace.tool_calls`.

Ranking is BM25 over title + text with a light suffix stemmer, so the same
query always returns the same docs in the same order (ties break by corpus
order). Only docs with a positive score are returned, at most `TOP_K`.

An unknown doc id is an "unrecorded call": the agent asked for something the
recording doesn't hold. With `on_unrecorded_call="fail"` the session is marked
`failed` (the runner makes the sample a SYSTEM error); with `"error_result"`
the agent gets an `is_error` tool result and carries on.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from llm_evals.types import Case

OnUnrecordedCall = Literal["fail", "error_result"]
ON_UNRECORDED_CALL_VALUES = ("fail", "error_result")

TOP_K = 5
SNIPPET_CHARS = 200
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have how i in
    into is it its of on or that the their there these this those to was were what
    when where which who whom whose why will with would you your than then so if
    about after before during over under between can could should may might
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude suffix stripping: "completed"/"completion" -> "complet"."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercased, stemmed content words (stopwords removed)."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _clip(text: str, limit: int = SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "..."


@dataclass(frozen=True)
class ToolResult:
    """What goes into one `tool_result` block: JSON text, and whether it's an error."""

    content: str
    is_error: bool = False


SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": (
        "Search the document collection by keywords. Returns up to 5 matching documents, "
        "best first, each with its doc_id, title and a short snippet. Use fetch to read "
        "a document in full."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to search for."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

FETCH_TOOL: dict[str, Any] = {
    "name": "fetch",
    "description": (
        "Fetch one document by the doc_id a search returned. Returns its doc_id, title "
        "and full text."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "A doc_id from search."}},
        "required": ["doc_id"],
        "additionalProperties": False,
    },
}


def tool_schemas() -> list[dict[str, Any]]:
    """Tool definitions for the Messages API (fresh copies, safe to mutate)."""
    return json.loads(json.dumps([SEARCH_TOOL, FETCH_TOOL]))


class ToolSession:
    def __init__(self, case: Case, on_unrecorded_call: OnUnrecordedCall = "fail") -> None:
        if on_unrecorded_call not in ON_UNRECORDED_CALL_VALUES:
            raise ValueError(
                f"on_unrecorded_call must be one of {ON_UNRECORDED_CALL_VALUES}, "
                f"not {on_unrecorded_call!r}"
            )
        fixtures = case.fixtures or {}
        docs = fixtures.get("docs")
        if not isinstance(docs, list):
            raise ValueError(f"case {case.id!r} has no fixtures.docs list to replay")
        self.case_id = case.id
        self.on_unrecorded_call: OnUnrecordedCall = on_unrecorded_call
        self.docs: list[dict[str, str]] = []
        self._by_id: dict[str, dict[str, str]] = {}
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict) or not {"id", "title", "text"} <= doc.keys():
                raise ValueError(f"case {case.id!r}: fixtures.docs[{i}] needs id, title, text")
            doc_id = str(doc["id"])
            if doc_id in self._by_id:
                raise ValueError(f"case {case.id!r}: duplicate doc id {doc_id!r}")
            clean = {"id": doc_id, "title": str(doc["title"]), "text": str(doc["text"])}
            self.docs.append(clean)
            self._by_id[doc_id] = clean
        self.calls: list[dict[str, Any]] = []
        self.unrecorded: list[str] = []  # one reason per unrecorded call
        self._index()

    # --- state ---------------------------------------------------------------

    @property
    def failed(self) -> bool:
        """True when an unrecorded call happened under `on_unrecorded_call="fail"`."""
        return self.on_unrecorded_call == "fail" and bool(self.unrecorded)

    @property
    def failure_reason(self) -> str | None:
        if not self.failed:
            return None
        return "unrecorded tool call: " + "; ".join(self.unrecorded)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas()

    # --- ranking -------------------------------------------------------------

    def _index(self) -> None:
        self._doc_terms = [Counter(tokenize(d["title"] + " " + d["text"])) for d in self.docs]
        self._doc_len = [sum(c.values()) for c in self._doc_terms]
        n = len(self.docs)
        self._avg_len = (sum(self._doc_len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for terms in self._doc_terms:
            df.update(terms.keys())
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _bm25(self, i: int, query_terms: list[str]) -> float:
        terms, length = self._doc_terms[i], self._doc_len[i]
        norm = 1 - BM25_B + BM25_B * (length / self._avg_len if self._avg_len else 0.0)
        score = 0.0
        for t in set(query_terms):
            tf = terms.get(t, 0)
            if tf:
                score += self._idf[t] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
        return score

    @staticmethod
    def snippet(text: str, query_terms: list[str]) -> str:
        """The sentence sharing the most query terms (earliest on ties), ~200 chars."""
        wanted = set(query_terms)
        sentences = split_sentences(text)
        if not sentences:
            return ""
        best_i, best_hits = 0, -1
        for i, s in enumerate(sentences):
            hits = len(wanted & set(tokenize(s)))
            if hits > best_hits:
                best_i, best_hits = i, hits
        out = sentences[best_i]
        for nxt in sentences[best_i + 1 :]:  # pad short sentences with what follows
            if len(out) + 1 + len(nxt) > SNIPPET_CHARS:
                break
            out = f"{out} {nxt}"
        return _clip(out)

    # --- tools ---------------------------------------------------------------

    def search(self, query: str) -> list[dict[str, str]]:
        """Rank docs for `query`; record the call."""
        hits = self._rank(query)
        self.calls.append(
            {"name": "search", "input": {"query": query}, "doc_ids": [h["doc_id"] for h in hits]}
        )
        return hits

    def _rank(self, query: str) -> list[dict[str, str]]:
        q = tokenize(query)
        scored = [(self._bm25(i, q), i) for i in range(len(self.docs))]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:TOP_K]
        return [
            {
                "doc_id": self.docs[i]["id"],
                "title": self.docs[i]["title"],
                "snippet": self.snippet(self.docs[i]["text"], q),
            }
            for _, i in ranked
        ]

    def fetch(self, doc_id: str) -> dict[str, str] | None:
        """The doc, or None for an unknown id (recorded as an unrecorded call)."""
        doc = self._by_id.get(doc_id)
        if doc is None:
            reason = f"fetch({doc_id!r}) is not in the recorded corpus"
            self.unrecorded.append(reason)
            self.calls.append(
                {"name": "fetch", "input": {"doc_id": doc_id}, "doc_ids": [], "error": reason}
            )
            return None
        self.calls.append({"name": "fetch", "input": {"doc_id": doc_id}, "doc_ids": [doc_id]})
        return {"doc_id": doc["id"], "title": doc["title"], "text": doc["text"]}

    def call(self, name: str, tool_input: Any) -> ToolResult:
        """Execute one tool_use block. Never raises; failures become error results."""
        if name == "search":
            query = tool_input.get("query") if isinstance(tool_input, dict) else None
            if not isinstance(query, str):
                return self._misuse(name, tool_input, "search needs a string 'query'")
            return ToolResult(json.dumps(self.search(query), ensure_ascii=False))
        if name == "fetch":
            doc_id = tool_input.get("doc_id") if isinstance(tool_input, dict) else None
            if not isinstance(doc_id, str):
                return self._misuse(name, tool_input, "fetch needs a string 'doc_id'")
            doc = self.fetch(doc_id)
            if doc is None:
                return ToolResult(
                    f"Error: no document with doc_id {doc_id!r}. Use a doc_id returned by search.",
                    is_error=True,
                )
            return ToolResult(json.dumps(doc, ensure_ascii=False))
        return self._misuse(name, tool_input, f"unknown tool {name!r}; tools are search and fetch")

    def _misuse(self, name: str, tool_input: Any, reason: str) -> ToolResult:
        """Bad tool name or input: an error result the model can recover from."""
        self.calls.append(
            {
                "name": name,
                "input": tool_input if isinstance(tool_input, dict) else {"raw": tool_input},
                "doc_ids": [],
                "error": reason,
            }
        )
        return ToolResult(f"Error: {reason}", is_error=True)


__all__ = [
    "FETCH_TOOL",
    "SEARCH_TOOL",
    "OnUnrecordedCall",
    "ToolResult",
    "ToolSession",
    "split_sentences",
    "stem",
    "tokenize",
    "tool_schemas",
]
