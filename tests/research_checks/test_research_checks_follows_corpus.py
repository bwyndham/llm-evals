from __future__ import annotations

import pytest
from research_checks_helpers import FakeContext, make_case, make_output, make_result

from llm_evals.checks.research import follows_corpus


def run(case, output):
    return follows_corpus().score(case, make_result(output), FakeContext())


def changed(**kwargs):
    return make_case(answer="15,200", original_answer="12,500", op="change_fact",
                     perturbation={"fact_id": "population"}, **kwargs)


def test_defaults():
    check = follows_corpus()
    assert (check.name, check.kind, check.critical) == ("follows_corpus", "metamorphic", False)


@pytest.mark.parametrize("op", ["none", "plant_fact", "remove_answer_doc", "add_conflict"])
def test_not_applicable_on_other_perturbations(op):
    score = run(make_case(op=op), make_output(answer="anything"))
    assert score.passed
    assert score.reason == "not applicable"
    assert score.sub_scores == [{"applicable": False, "op": op}]


def test_not_applicable_without_source():
    case = make_case().model_copy(update={"source": None})
    score = run(case, make_output())
    assert score.passed and score.sub_scores[0]["applicable"] is False


def test_follows_the_changed_fact():
    score = run(changed(), make_output(answer="15200 people"))
    assert score.passed, score.reason
    detail = score.sub_scores[0]
    assert detail["applicable"] is True
    assert (detail["matches_new"], detail["matches_original"]) == (True, False)


def test_following_the_original_value_fails():
    score = run(changed(), make_output(answer="12,500 people"))
    assert not score.passed
    assert "original value '12,500'" in score.reason
    assert score.sub_scores[0]["matches_original"] is True


def test_hedging_with_both_values_fails():
    score = run(changed(), make_output(answer="15,200 (older sources say 12,500)"))
    assert not score.passed and "both" in score.reason


def test_neither_value_fails():
    score = run(changed(), make_output(answer="about 9,000"))
    assert not score.passed and "neither" in score.reason


def test_abstaining_fails():
    score = run(changed(), make_output(status="not_found", answer="", claims=[]))
    assert not score.passed and "status is not_found" in score.reason


def test_change_fact_alias_counts():
    case = changed(aliases=["fifteen thousand two hundred"])
    assert run(case, make_output(answer="Fifteen thousand two hundred")).passed


def test_change_fact_without_original_is_check_error():
    case = make_case(answer="15,200", original_answer=None, op="change_fact")
    score = run(case, make_output(answer="15,200"))
    assert not score.passed and score.reason.startswith("check error")
