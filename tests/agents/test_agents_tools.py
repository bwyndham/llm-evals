from __future__ import annotations

import json

import pytest

from agents_helpers import DOCS, make_case  # type: ignore[import-not-found]

from llm_evals.agents.tools import (
    SNIPPET_CHARS,
    TOP_K,
    ToolSession,
    stem,
    tokenize,
    tool_schemas,
)


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("When was the Bridge completed?") == ["bridge", "complet"]
    assert stem("completion") == stem("completed") == "complet"
    assert stem("1931") == "1931"


def test_search_ranks_the_answer_doc_first_and_records_the_call() -> None:
    s = ToolSession(make_case())
    hits = s.search("Halvorsen Bridge completed")
    assert [h["doc_id"] for h in hits][:2] == ["d1", "d2"]
    assert "d4" not in [h["doc_id"] for h in hits]  # zero-score docs are excluded
    assert hits[0]["title"] == "Halvorsen Bridge"
    assert "completed in 1931" in hits[0]["snippet"]
    assert s.calls == [
        {"name": "search", "input": {"query": "Halvorsen Bridge completed"},
         "doc_ids": [h["doc_id"] for h in hits]}
    ]


def test_search_is_deterministic_and_capped_at_top_k() -> None:
    docs = [{"id": f"x{i}", "title": f"Doc {i}", "text": "bridge " * (i + 1)} for i in range(9)]
    s = ToolSession(make_case(docs))
    first = s.search("bridge")
    assert len(first) == TOP_K
    assert first == s.search("bridge")
    assert s.search("nothing matches") == []


def test_ties_break_by_corpus_order() -> None:
    docs = [{"id": i, "title": "same", "text": "identical text here"} for i in ("b", "a", "c")]
    assert [h["doc_id"] for h in ToolSession(make_case(docs)).search("identical")] == ["b", "a", "c"]


def test_snippet_is_best_sentence_and_bounded() -> None:
    long = "Filler words go here. " + ("The bridge was painted green with care " * 12) + "end."
    snip = ToolSession.snippet(long, tokenize("bridge painted"))
    assert snip.startswith("The bridge was painted")
    assert len(snip) <= SNIPPET_CHARS + 3


def test_fetch_known_doc_records_doc_id() -> None:
    s = ToolSession(make_case())
    r = s.call("fetch", {"doc_id": "d3"})
    assert not r.is_error
    assert json.loads(r.content) == {"doc_id": "d3", "title": DOCS[2]["title"], "text": DOCS[2]["text"]}
    assert s.calls == [{"name": "fetch", "input": {"doc_id": "d3"}, "doc_ids": ["d3"]}]
    assert not s.failed


def test_unknown_doc_under_fail_marks_the_session() -> None:
    s = ToolSession(make_case(), on_unrecorded_call="fail")
    r = s.call("fetch", {"doc_id": "nope"})
    assert r.is_error
    assert s.failed and "nope" in (s.failure_reason or "")
    assert s.calls[0]["doc_ids"] == [] and "error" in s.calls[0]


def test_unknown_doc_under_error_result_is_recoverable() -> None:
    s = ToolSession(make_case(), on_unrecorded_call="error_result")
    r = s.call("fetch", {"doc_id": "nope"})
    assert r.is_error and r.content.startswith("Error:")
    assert s.unrecorded and not s.failed and s.failure_reason is None


def test_bad_tool_name_or_input_is_an_error_result_not_a_failure() -> None:
    s = ToolSession(make_case())
    assert s.call("browse", {"url": "x"}).is_error
    assert s.call("search", {"q": "x"}).is_error
    assert s.call("fetch", "d1").is_error
    assert not s.failed and len(s.calls) == 3


def test_schemas_are_strict() -> None:
    schemas = tool_schemas()
    assert [t["name"] for t in schemas] == ["search", "fetch"]
    for t in schemas:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False
        assert t["input_schema"]["required"] == list(t["input_schema"]["properties"])


def test_bad_fixtures_raise() -> None:
    with pytest.raises(ValueError, match="fixtures.docs"):
        ToolSession(make_case().model_copy(update={"fixtures": None}))
    with pytest.raises(ValueError, match="duplicate"):
        ToolSession(make_case([DOCS[0], DOCS[0]]))
    with pytest.raises(ValueError, match="on_unrecorded_call"):
        ToolSession(make_case(), on_unrecorded_call="ignore")  # type: ignore[arg-type]
