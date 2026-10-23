from __future__ import annotations

import pytest
from checks_helpers import DROP, make_output, make_result

from llm_evals.checks.format_rules import format_rules


def score(output, case, ctx, **params):
    return format_rules(**params).score(case, make_result(output), ctx)


def test_defaults_and_attributes():
    check = format_rules()
    assert check.name == "format_rules"
    assert check.kind == "deterministic"
    assert check.critical is False
    assert check.required_fields == ["title", "description", "acceptance_criteria", "assumptions", "open_questions"]
    assert check.title_max_chars == 80
    assert check.max_open_questions is None


def test_unknown_param_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_title'"):
        format_rules(max_title=10)


@pytest.mark.parametrize(
    "params",
    [{"title_max_chars": "80"}, {"title_max_chars": 0}, {"max_open_questions": -1}, {"required_fields": "title"}],
)
def test_bad_param_values_raise_type_error(params):
    with pytest.raises(TypeError, match="format_rules:"):
        format_rules(**params)


def test_good_output_passes(case, ctx):
    s = score(make_output(), case, ctx)
    assert s.passed, s.reason
    assert s.sub_scores == []


def test_empty_lists_are_allowed(case, ctx):
    assert score(make_output(assumptions=[], open_questions=[]), case, ctx).passed


def test_missing_and_empty_required_fields(case, ctx):
    s = score(make_output(description=DROP, title="  ", assumptions=None), case, ctx)
    assert not s.passed
    assert s.reason.startswith("3 format violation(s):")
    assert "description is missing" in s.reason
    assert "title is empty" in s.reason
    assert "assumptions is missing" in s.reason


def test_empty_acceptance_criteria(case, ctx):
    s = score(make_output(acceptance_criteria=""), case, ctx)
    assert not s.passed
    assert "acceptance_criteria is empty" in s.reason


def test_custom_required_fields(case, ctx):
    out = make_output(open_questions=DROP)
    assert not score(out, case, ctx).passed
    assert score(out, case, ctx, required_fields=["title", "description", "acceptance_criteria"]).passed


def test_title_too_long(case, ctx):
    s = score(make_output(title="x" * 81), case, ctx)
    assert not s.passed
    assert "title is 81 chars (max 80)" in s.reason
    assert score(make_output(title="x" * 80), case, ctx).passed
    assert not score(make_output(title="x" * 41), case, ctx, title_max_chars=40).passed


def test_title_trailing_period_and_line_break(case, ctx):
    s = score(make_output(title="Export the report.\nNow"), case, ctx)
    assert "title contains a line break" in s.reason
    s = score(make_output(title="Export the report."), case, ctx)
    assert not s.passed
    assert s.sub_scores == [{"field": "title", "rule": "trailing_period", "detail": "title ends with a period"}]


def test_non_string_fields(case, ctx):
    s = score(make_output(title=42, description=["a"]), case, ctx)
    assert not s.passed
    assert "title must be a string, got int" in s.reason
    assert "description must be a string, got list" in s.reason


def test_non_list_assumptions(case, ctx):
    s = score(make_output(assumptions="CSV uses UTF-8"), case, ctx)
    assert not s.passed
    assert "assumptions must be a list of strings, got str" in s.reason


def test_list_with_bad_items(case, ctx):
    s = score(make_output(open_questions=["ok", "", 3]), case, ctx)
    assert not s.passed
    assert "open_questions items [1, 2] are not non-empty strings" in s.reason


def test_max_open_questions(case, ctx):
    out = make_output(open_questions=["a?", "b?", "c?"])
    assert score(out, case, ctx).passed
    s = score(out, case, ctx, max_open_questions=2)
    assert not s.passed
    assert "open_questions has 3 items (max 2)" in s.reason


def test_error_results_fail_without_raising(case, ctx, error_results):
    check = format_rules()
    for r in error_results:
        s = check.score(case, r, ctx)
        assert not s.passed
        assert s.reason.startswith("no output (")
