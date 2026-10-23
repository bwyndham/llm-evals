"""The generic checks any suite can use."""

from __future__ import annotations

import pytest

from llm_evals import registry
from llm_evals.types import Case, Result


class Ctx:
    suite_dir = "."


def score(name, output, expected=None, **params):
    check = registry.get("check", name)(**params)
    case = Case(id="c1", input={"message": "hi"}, expected=expected or {}, provenance="constructed")
    result = Result(run_id="r", case_id="c1", version_id="v", sample_idx=0, output=output)
    return check.score(case, result, Ctx())


def test_field_matches_ignores_case_and_spacing_and_takes_alternatives():
    assert score("field_matches", {"category": " Billing "}, {"category": "billing"}, field="category").passed
    assert score("field_matches", {"category": "refund"}, {"category": ["billing", "refund"]}, field="category").passed
    s = score("field_matches", {"category": "bug"}, {"category": "billing"}, field="category")
    assert not s.passed and "expected 'billing'" in s.reason
    assert not score("field_matches", {"x": 1}, {"category": "billing"}, field="category").passed  # missing field


def test_field_matches_nested_paths_and_cases_without_an_answer():
    assert score("field_matches", {"answer": {"label": "yes"}}, {"label": "YES"}, field="answer.label",
                 expected_key="label").passed
    s = score("field_matches", {"category": "bug"}, {}, field="category")
    assert s.passed and s.sub_scores == [{"applicable": False}]
    s = score("field_matches", {"category": "Bug"}, {"category": "bug"}, field="category", ignore_case=False)
    assert not s.passed


def test_regex_and_its_flags():
    assert score("regex", {"reply": "Order #A123"}, field="reply", pattern=r"#[A-Z]\d+").passed
    assert score("regex", {"reply": "sorry!"}, field="reply", pattern="SORRY", flags="i").passed
    assert not score("regex", {"reply": "As an AI model"}, field="reply", pattern="as an ai", flags="i",
                     must_match=False).passed
    with pytest.raises(ValueError, match="unknown flag"):
        registry.get("check", "regex")(field="reply", pattern="x", flags="q")


def test_keywords_present_and_max_length():
    expected = {"keywords": ["refund", "5 days"]}
    assert score("keywords_present", {"reply": "Your REFUND arrives in 5 days."}, expected, field="reply").passed
    s = score("keywords_present", {"reply": "Your refund is on its way."}, expected, field="reply")
    assert not s.passed and "'5 days'" in s.reason
    assert score("keywords_present", {"reply": "refund"}, expected, field="reply", min_share=0.5).passed
    assert score("max_length", {"reply": "short"}, field="reply", max_chars=10).passed
    assert not score("max_length", {"reply": "x" * 11}, field="reply", max_chars=10).passed
