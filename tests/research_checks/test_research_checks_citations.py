from __future__ import annotations

from research_checks_helpers import FakeContext, fetch, make_case, make_output, make_result, search

from llm_evals.checks.research import citations_exist, retrieved_doc_ids
from llm_evals.types import ErrorKind


def run(output, tool_calls=None, case=None, **params):
    return citations_exist(**params).score(case or make_case(), make_result(output, tool_calls), FakeContext())


def test_defaults():
    check = citations_exist()
    assert (check.name, check.kind, check.critical) == ("citations_exist", "structural", True)
    assert citations_exist(critical=False).critical is False


def test_all_citations_retrieved_passes():
    score = run(make_output())
    assert score.passed and score.value == 1.0
    assert score.sub_scores == []


def test_search_results_count_as_retrieved():
    output = make_output(claims=[{"text": "Founded in 1852.", "citations": ["d2"]}])
    assert run(output, [search("d2")]).passed


def test_citation_in_corpus_but_not_retrieved_fails():
    output = make_output(claims=[
        {"text": "Harbor Town has 12,500 people.", "citations": ["d1"]},
        {"text": "The lighthouse is 31 metres tall.", "citations": ["d3"]},
    ])
    score = run(output, [search("d1", "d2")])
    assert not score.passed
    assert score.value == 0.5
    assert score.sub_scores == [{
        "claim": 1, "doc_id": "d3", "problem": "not_retrieved", "severity": "medium",
        "claim_text": "The lighthouse is 31 metres tall.",
    }]
    assert "never retrieved or fetched: d3" in score.reason


def test_citation_not_in_corpus_fails_as_fabricated():
    output = make_output(claims=[{"text": "Harbor Town has 12,500 people.", "citations": ["d1", "d99"]}])
    # Even a trace that claims d99 was returned can't make a non-corpus doc real.
    score = run(output, [search("d1", "d99")])
    assert not score.passed
    assert [(b["doc_id"], b["problem"], b["severity"]) for b in score.sub_scores] == [("d99", "not_in_corpus", "high")]
    assert "not in the corpus: d99" in score.reason


def test_both_kinds_reported_per_citation():
    output = make_output(claims=[
        {"text": "a", "citations": ["d3", "ghost"]},
        {"text": "b", "citations": ["ghost"]},
    ])
    score = run(output, [search("d1")])
    assert [(b["claim"], b["doc_id"], b["problem"]) for b in score.sub_scores] == [
        (0, "d3", "not_retrieved"), (0, "ghost", "not_in_corpus"), (1, "ghost", "not_in_corpus"),
    ]
    assert score.value == 0.0


def test_no_tool_calls_means_nothing_retrieved():
    assert not run(make_output(), []).passed


def test_no_claims_passes_for_not_found_and_answered():
    not_found = run(make_output(status="not_found", answer="", claims=[]))
    assert not_found.passed and "not_found" in not_found.reason
    answered = run(make_output(claims=[]))
    assert answered.passed and "uncited_claims" in answered.reason


def test_uncited_claims_are_left_to_uncited_claims():
    score = run(make_output(claims=[{"text": "x", "citations": []}]))
    assert score.passed


def test_case_without_corpus_checks_retrieval_only():
    case = make_case(docs=None)
    assert run(make_output(), case=case).passed
    assert not run(make_output(claims=[{"text": "x", "citations": ["d9"]}]), case=case).passed


def test_fetch_without_doc_ids_uses_its_input():
    calls = [{"name": "fetch", "input": {"doc_id": "d1"}}]
    assert retrieved_doc_ids(make_result(make_output(), calls)) == {"d1"}
    assert run(make_output(), calls).passed


def test_malformed_and_missing_output_fail():
    assert run(make_output(claims="d1")).reason.startswith("malformed output: claims must be a list")
    bad_cites = make_output(claims=[{"text": "x", "citations": "d1"}])
    assert run(bad_cites).reason.startswith("malformed output")
    assert run({"status": "maybe", "answer": "", "claims": []}).reason.startswith("malformed output: status")
    errored = citations_exist().score(
        make_case(), make_result(None, error_kind=ErrorKind.SYSTEM, error="bad json"), FakeContext()
    )
    assert not errored.passed and errored.reason == "no output (system error: bad json)"
    assert fetch("d1")["doc_ids"] == ["d1"]
