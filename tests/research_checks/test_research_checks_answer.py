from __future__ import annotations

import sys
import types

import pytest
from research_checks_helpers import REAL_CORPUS_MATCHER, FakeContext, make_case, make_output, make_result

from llm_evals.checks import research
from llm_evals.checks.research import (
    answer_match,
    conflicting_values,
    extract_numbers,
    local_answer_matches,
    normalize_answer,
)


def run(case, output):
    return answer_match().score(case, make_result(output), FakeContext())


# --- normalization -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("12,500", "12500"),
        ("12 500 people", "12500 people"),
        ("The Café  Rouge!", "cafe rouge"),
        ("3.5 km", "3.5 km"),
        ("1,234,567.89", "1234567.89"),
    ],
)
def test_normalize_answer(text, normalized):
    assert normalize_answer(text) == normalized


def test_extract_numbers():
    assert extract_numbers("12,500 people in 2020") == [12500.0, 2020.0]
    assert extract_numbers("$4.2 million") == [4_200_000.0]
    assert extract_numbers("about 12k residents") == [12_000.0]
    assert extract_numbers("12 km") == [12.0]  # "km" isn't the scale k


@pytest.mark.parametrize(
    ("output", "expected", "matches"),
    [
        ("12500 people", {"answer": "12,500"}, True),
        ("12,500", {"answer": "12500 people"}, True),
        ("About 12\u00a0500 people live there.", {"answer": "12,500"}, True),
        ("4,200,000 dollars", {"answer": "$4.2 million"}, True),
        ("12,000", {"answer": "12,500"}, False),
        ("2,500 people", {"answer": "12,500"}, False),
        ("Paris, France", {"answer": "Paris"}, True),
        ("It was founded by Ada Lovelace.", {"answer": "Lovelace", "aliases": []}, True),
        ("The founder was A. Lovelace", {"answer": "Ada Lovelace", "aliases": ["A. Lovelace"]}, True),
        ("May 3, 2021", {"answer": "March 3, 2021"}, False),  # dates aren't number-matched
        ("Parisian", {"answer": "Paris"}, False),  # token boundaries
        ("", {"answer": "Paris"}, False),
    ],
)
def test_local_answer_matches(output, expected, matches):
    assert local_answer_matches(output, expected) is matches


def test_prefers_corpus_helper(monkeypatch):
    calls = []

    def fake(output_answer, expected):
        calls.append((output_answer, expected))
        return expected["answer"] == "magic"

    module = types.ModuleType("llm_evals.cases.corpus")
    module.answer_matches = fake
    monkeypatch.setitem(sys.modules, "llm_evals.cases.corpus", module)
    monkeypatch.setattr(research, "_corpus_matcher", REAL_CORPUS_MATCHER)
    assert research.answer_matches("anything", {"answer": "magic"}) is True
    assert research.answer_matches("12,500", {"answer": "12,500"}) is False  # the helper decides
    assert len(calls) == 2


def test_falls_back_when_corpus_helper_raises(monkeypatch):
    def broken(output_answer, expected):
        raise RuntimeError("boom")

    monkeypatch.setattr(research, "_corpus_matcher", lambda: broken)
    assert research.answer_matches("12500 people", {"answer": "12,500"}) is True


# --- answer_match --------------------------------------------------------------


def test_defaults():
    check = answer_match()
    assert (check.name, check.kind, check.critical) == ("answer_match", "reference", False)


def test_answer_matches_with_number_normalization():
    score = run(make_case(answer="12,500"), make_output(answer="12500 people"))
    assert score.passed, score.reason
    assert score.sub_scores[0]["matched"] is True


def test_answer_matches_alias():
    case = make_case(answer="Harbour Town", aliases=["Harbor Town"])
    assert run(case, make_output(answer="harbor town")).passed


def test_wrong_answer_fails():
    score = run(make_case(answer="12,500"), make_output(answer="11,000"))
    assert not score.passed and "does not match" in score.reason


def test_answer_expected_but_abstained_fails():
    score = run(make_case(), make_output(status="not_found", answer="12,500"))
    assert not score.passed and "status is not_found" in score.reason


def test_abstain_passes_on_not_found_and_ignores_answer_text():
    case = make_case(behavior="abstain", answer=None)
    assert run(case, make_output(status="not_found", answer="maybe 12,500?", claims=[])).passed


def test_abstain_fails_when_answered():
    case = make_case(behavior="abstain", answer=None)
    score = run(case, make_output(answer="12,500"))
    assert not score.passed and "abstain" in score.reason


def conflict_case(**kwargs):
    return make_case(behavior="flag_conflict", answer="12,500", original_answer=None,
                     op="add_conflict", perturbation={"value": "14,000"}, **kwargs)


def test_conflicting_values_collects_expected_and_perturbation():
    assert conflicting_values(conflict_case()) == ["12,500", "14,000"]
    listed = make_case(behavior="flag_conflict", answer="12,500", conflicting_values=["12500", "9,000"])
    assert conflicting_values(listed) == ["12500", "9,000"]  # "12,500" and "12500" are one value


def test_flag_conflict_passes_naming_both_values():
    output = make_output(answer="Sources disagree: 12,500 or 14,000.", conflicts=["d1 says 12,500; d4 says 14,000"])
    score = run(conflict_case(), output)
    assert score.passed, score.reason
    assert score.sub_scores[0]["asserted"] == ["12,500", "14,000"]


def test_flag_conflict_passes_naming_neither_value():
    output = make_output(answer="The documents give different figures.", conflicts=["population differs"])
    assert run(conflict_case(), output).passed


def test_flag_conflict_fails_without_conflicts():
    output = make_output(answer="Sources disagree: 12,500 or 14,000.", conflicts=[])
    score = run(conflict_case(), output)
    assert not score.passed and "conflicts is empty" in score.reason


def test_flag_conflict_fails_when_answer_asserts_one_value():
    output = make_output(answer="14,000 people", conflicts=["d1 and d4 disagree"])
    score = run(conflict_case(), output)
    assert not score.passed and "only one of the conflicting values ('14,000')" in score.reason


def test_flag_conflict_with_one_known_value_checks_conflicts_only():
    case = make_case(behavior="flag_conflict", answer="12,500", op="add_conflict")
    score = run(case, make_output(answer="12,500", conflicts=["disagree"]))
    assert score.passed and "fewer than two" in score.reason


def test_unknown_behavior_is_check_error():
    score = run(make_case(behavior="guess"), make_output())
    assert not score.passed and score.reason.startswith("check error: ValueError: unknown expected.behavior")


def test_non_string_answer_is_malformed():
    score = run(make_case(), make_output(answer=12500))
    assert score.reason.startswith("malformed output: answer must be a string")
