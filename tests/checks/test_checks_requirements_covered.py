"""`requirements_covered` and the shared keyword matcher."""

from __future__ import annotations

import pytest
from checks_helpers import FakeContext, make_result

from llm_evals import registry
from llm_evals.checks.atom_keywords import AtomKeywords
from llm_evals.checks.keywords import keyword_in, keyword_in_any
from llm_evals.checks.requirements_covered import RequirementsCovered, requirements_of
from llm_evals.types import Case


@pytest.mark.parametrize("keyword, text", [
    ("api keys", "An integration partner can create an API key from Developer settings."),
    ("wcag 2.1 aa", "The invoice editor meets WCAG 2.1 level AA."),
    ("named tax rate", 'A bookkeeper can create a tax rate with a name, such as "State sales tax 7%".'),
    ("paid invoices", "A paid invoice syncs within 15 minutes."),
    ("approv", "A freelancer can send an estimate for approval."),
    ("overdue", "Bookkeepers can turn overdue-invoice reminders on."),
    ("0-30", "Invoices are grouped into 0-30, 31-60 and 61-90 days."),
    ("read-only", "The workspace becomes read-only."),
])
def test_words_mode_matches_forms_order_and_prefixes(keyword, text):
    assert keyword_in(keyword, text, "words")


@pytest.mark.parametrize("keyword, text", [
    ("0-30", "Invoices older than 30 days"),          # a range is one token
    ("weekly report", "The monthly report is emailed"),
    ("api keys", "An API token is created"),
])
def test_words_mode_still_requires_every_word(keyword, text):
    assert not keyword_in(keyword, text, "words")


def test_literal_mode_is_the_old_substring_rule():
    assert keyword_in("api key", "Create an API key", "literal")
    assert not keyword_in("api keys", "Create an API key", "literal")
    with pytest.raises(ValueError):
        keyword_in("x", "x", "fuzzy")  # type: ignore[arg-type]


def test_words_must_meet_in_one_place():
    assert not keyword_in_any("named tax rate", ["A tax rate can be created", "Each has a name"], "words")


def _check(**kw) -> RequirementsCovered:
    params = {"fields": ["acceptance_criteria"], "min_keyword_share": 0.6, "match": "words", "split": "none",
              "requirements_key": "requirements", "critical": False, **kw}
    return RequirementsCovered(**params)


def _case(expected=None, source=None) -> Case:
    return Case(id="c1", input={"ask": "..."}, provenance="constructed", expected=expected, source=source)


AC = ["An admin can export the usage report as a CSV file", "The export covers the last 90 days"]


def test_covered_from_expected_requirements():
    case = _case(expected={"requirements": [{"id": "r1", "keywords": ["csv", "usage report"]},
                                            {"id": "r2", "keywords": ["90 days"]}]})
    s = _check().score(case, make_result({"acceptance_criteria": AC, "summary": "x"}), FakeContext())
    assert s.passed and s.value == 1.0, s.reason


def test_only_the_named_fields_count():
    case = _case(expected={"requirements": [["90 days"]]})
    out = {"acceptance_criteria": ["An admin can export"], "summary": "covers 90 days"}
    s = _check().score(case, make_result(out), FakeContext())
    assert not s.passed and s.value == 0.0 and "acceptance_criteria" in s.reason


def test_hinted_requirements_are_reported_not_required():
    case = _case(source={"atoms": [{"id": "a1", "keywords": ["csv"], "text": "CSV export"},
                                   {"id": "a2", "keywords": ["audit log"], "mode": "hinted"}]})
    s = _check().score(case, make_result({"acceptance_criteria": AC}), FakeContext())
    assert s.passed and "1 hinted" in s.reason
    assert [r["id"] for r in requirements_of(case, "requirements")] == ["a1", "a2"]


def test_string_fields_and_split_lines():
    case = _case(expected={"requirements": [["named tax rate"]]})
    text = "Scenario: tax\nGiven a tax rate\nThen it has a name"
    assert _check(split="none").score(case, make_result({"acceptance_criteria": text}), FakeContext()).passed
    assert not _check(split="lines").score(case, make_result({"acceptance_criteria": text}), FakeContext()).passed


def test_not_applicable_without_requirements_and_failing_without_fields():
    assert not _check().applies(_case())
    case = _case(expected={"requirements": [["csv"]]})
    s = _check(fields=["nope"]).score(case, make_result({"acceptance_criteria": AC}), FakeContext())
    assert not s.passed and "none of the fields" in s.reason


def test_registered_with_validation():
    factory = registry.get("check", "requirements_covered")
    assert factory(fields=["acceptance_criteria"]).match == "words"
    with pytest.raises(TypeError):
        factory(fields=[])
    with pytest.raises(TypeError):
        factory(fields=["x"], split="sentences")


def test_atom_keywords_words_mode_is_opt_in():
    case = _case(source={"atoms": [{"id": "s/a1", "keywords": ["api keys", "developer settings"]}]})
    ticket = {"title": "API keys", "description": "d",
              "acceptance_criteria": "Feature: f\nScenario: create an API key\nGiven I am on Developer settings\n"
                                     "When I create a key\nThen I see the API key"}
    # Title says "API keys" in both modes; the description-free variant shows the difference.
    literal = AtomKeywords(min_keyword_share=1.0, critical=False)
    words = AtomKeywords(min_keyword_share=1.0, critical=False, match="words")
    ticket["title"] = "Create an API key"
    assert not literal.score(case, make_result(ticket), FakeContext()).passed
    assert words.score(case, make_result(ticket), FakeContext()).passed
    with pytest.raises(TypeError):
        AtomKeywords(min_keyword_share=0.6, critical=False, match="fuzzy")  # type: ignore[arg-type]
