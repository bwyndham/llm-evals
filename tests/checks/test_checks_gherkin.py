from __future__ import annotations

import pytest
from checks_helpers import (
    DROP,
    GOOD_GHERKIN,
    MISSING_THEN,
    NO_SCENARIOS,
    SYNTAX_ERROR,
    TWO_FEATURES,
    make_output,
    make_result,
)

from llm_evals.checks.gherkin import analyze_gherkin, gherkin_parses, scenario_count


def score(output, case, ctx, **params):
    return gherkin_parses(**params).score(case, make_result(output), ctx)


def test_defaults_and_attributes():
    check = gherkin_parses()
    assert check.name == "gherkin_parses"
    assert check.kind == "deterministic"
    assert check.critical is True
    assert gherkin_parses(critical=False).critical is False


def test_unknown_param_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'strict'"):
        gherkin_parses(strict=True)


def test_valid_gherkin_passes_with_sub_scores(case, ctx):
    s = score(make_output(), case, ctx)
    assert s.passed, s.reason
    assert s.value is None
    assert [(sub["scenario"], sub["steps"]) for sub in s.sub_scores] == [
        ("Admin exports the report as CSV", 5),
        ("Non-admins cannot export", 3),
    ]
    assert "2 scenario(s)" in s.reason


def test_missing_then_fails(case, ctx):
    s = score(make_output(acceptance_criteria=MISSING_THEN), case, ctx)
    assert not s.passed
    assert "'Admin exports the report'" in s.reason
    assert "no Then step" in s.reason
    assert s.sub_scores == [{"scenario": "Admin exports the report", "steps": 3, "has_when": True, "has_then": False}]


def test_missing_when_fails(case, ctx):
    text = "Feature: F\n  Scenario: S\n    Given a\n    Then b\n"
    s = score(make_output(acceptance_criteria=text), case, ctx)
    assert not s.passed
    assert "no When step" in s.reason


def test_and_but_inherit_previous_type():
    # "And" after When counts as When; "But" after Then counts as Then.
    text = "Feature: F\n  Scenario: S\n    Given a\n    When b\n    And c\n    Then d\n    But e\n"
    report = analyze_gherkin(text)
    assert report.ok
    # "And" right after Given stays a Given: no When here.
    text2 = "Feature: F\n  Scenario: S\n    Given a\n    And b\n    Then c\n"
    report2 = analyze_gherkin(text2)
    assert not report2.ok
    assert report2.scenarios[0].has_when is False


def test_star_steps_inherit_previous_type():
    text = "Feature: F\n  Scenario: S\n    When a\n    * b\n    Then c\n    * d\n"
    assert analyze_gherkin(text).ok


def test_background_when_counts_for_scenarios():
    text = "Feature: F\n  Background:\n    Given a\n    When b\n  Scenario: S\n    Then c\n"
    report = analyze_gherkin(text)
    assert report.ok
    assert report.scenarios[0].steps == 1


def test_scenarios_inside_rules_count():
    text = (
        "Feature: F\n  Rule: R\n    Scenario: A\n      When a\n      Then b\n"
        "    Scenario: B\n      When c\n      Then d\n"
    )
    assert scenario_count(text) == 2
    assert analyze_gherkin(text).ok


def test_two_features_fail(case, ctx):
    s = score(make_output(acceptance_criteria=TWO_FEATURES), case, ctx)
    assert not s.passed
    assert "expected exactly one Feature, found 2" in s.reason
    assert "Feature: Import usage report" in s.reason


def test_second_feature_after_scenarios_is_a_parse_error(case, ctx):
    text = GOOD_GHERKIN + "\nFeature: Another\n  Scenario: X\n    When a\n    Then b\n"
    s = score(make_output(acceptance_criteria=text), case, ctx)
    assert not s.passed
    assert "Gherkin parse error at line" in s.reason


def test_syntax_error_reports_line_and_column(case, ctx):
    s = score(make_output(acceptance_criteria=SYNTAX_ERROR), case, ctx)
    assert not s.passed
    assert s.reason.startswith("Gherkin parse error at line 5, column 3:")
    assert "this line is not Gherkin" in s.reason


def test_no_feature_fails(case, ctx):
    s = score(make_output(acceptance_criteria="# just a comment\n"), case, ctx)
    assert not s.passed
    assert "no Feature found" in s.reason


def test_no_scenarios_fails(case, ctx):
    s = score(make_output(acceptance_criteria=NO_SCENARIOS), case, ctx)
    assert not s.passed
    assert "no Scenario" in s.reason


@pytest.mark.parametrize(
    "value, expected", [("", "empty"), ("   \n", "empty"), (None, "not a string"), (["x"], "not a string")]
)
def test_empty_or_non_string_acceptance_criteria_fails(case, ctx, value, expected):
    s = score(make_output(acceptance_criteria=value), case, ctx)
    assert not s.passed
    assert expected in s.reason


def test_missing_acceptance_criteria_fails(case, ctx):
    s = score(make_output(acceptance_criteria=DROP), case, ctx)
    assert not s.passed
    assert "acceptance_criteria is missing" in s.reason


def test_non_english_dialect_parses():
    text = "# language: fr\nFonctionnalité: F\n  Scénario: S\n    Quand a\n    Alors b\n"
    assert analyze_gherkin(text).ok


def test_error_results_fail_without_raising(case, ctx, error_results):
    check = gherkin_parses()
    reasons = [check.score(case, r, ctx).reason for r in error_results]
    assert all(not check.score(case, r, ctx).passed for r in error_results)
    assert reasons[0] == "no output (system error: invalid JSON)"
    assert reasons[1] == "no output (infra error: timeout)"
    assert reasons[2] == "no output (system error: output is None)"


def test_scenario_count():
    assert scenario_count(GOOD_GHERKIN) == 2
    assert scenario_count(MISSING_THEN) == 1
    assert scenario_count(SYNTAX_ERROR) is None
    assert scenario_count("") is None
    assert scenario_count(None) is None
    assert scenario_count("# nothing\n") == 0
