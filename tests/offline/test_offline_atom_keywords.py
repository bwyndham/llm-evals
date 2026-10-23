"""Unit tests for the deterministic `atom_keywords` check (docs/contracts.md, Keyword faithfulness)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_evals import registry
from llm_evals.checks.atom_keywords import AtomKeywords, atom_keywords, scenario_texts
from llm_evals.types import Case, ErrorKind, Result


class _Ctx:
    suite_dir = Path(".")

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:  # pragma: no cover - unused
        raise AssertionError("atom_keywords must not rerun")


def _case(atoms: list[dict[str, Any]], hinted: list[str] | None = None) -> Case:
    return Case(
        id="c1",
        input={"ask": "..."},
        provenance="constructed",
        source={"atoms": atoms, "hinted_atom_ids": hinted or []},
    )


ATOMS = [
    {"id": "s/a1", "keywords": ["invoice list", "csv"], "mode": "explicit"},
    {"id": "s/a2", "keywords": ["90 days"], "mode": "explicit"},
    {"id": "s/a3", "keywords": ["invoice number", "customer name", "status"], "mode": "explicit"},
]


def _ticket(ac: str, *, title: str = "Invoice export", description: str = "Export.", **extra: Any) -> dict[str, Any]:
    return {"title": title, "description": description, "acceptance_criteria": ac,
            "assumptions": [], "open_questions": [], **extra}


def _result(output: dict[str, Any] | None, **kw: Any) -> Result:
    return Result(run_id="r", case_id="c1", version_id="v", sample_idx=0, output=output, **kw)


GOOD_AC = """Feature: Invoice export

  Scenario: Export the invoice list as CSV
    Given an admin is signed in
    When they export the invoice list
    Then a CSV file downloads

  Scenario: Only the last 90 days
    Given invoices older than 90 days exist
    When the export runs
    Then only invoices from the last 90 days are included

  Scenario: Row contents
    Given an export
    When it is opened
    Then each row has the invoice number, customer name and status
"""


# Same as GOOD_AC, but the 90-day scenario no longer states the number (it still
# names the CSV, so it is not an invented scenario).
AC_NO_90 = GOOD_AC.replace("Only the last 90 days", "Recent invoices only").replace(
    "invoices older than 90 days exist", "older invoices exist").replace(
    "only invoices from the last 90 days are included", "only recent invoices go into the CSV")


def check(**kw: Any) -> AtomKeywords:
    return atom_keywords(**kw)


def test_registered_as_deterministic():
    factory = registry.get("check", "atom_keywords")
    c = factory()
    assert c.name == "atom_keywords" and c.kind == "deterministic" and c.critical is False
    assert c.min_keyword_share == 0.6


def test_all_atoms_covered_passes():
    score = check().score(_case(ATOMS), _result(_ticket(GOOD_AC)), _Ctx())
    assert score.passed, score.reason
    assert score.value == 1.0
    atoms = [s for s in score.sub_scores if "atom" in s]
    assert [a["covered"] for a in atoms] == [True, True, True]
    assert atoms[2]["matched"] == ["invoice number", "customer name", "status"]


def test_threshold_is_ceil_of_share():
    c = check()
    assert [c.needed(n) for n in (1, 2, 3, 4, 5, 10)] == [1, 2, 2, 3, 3, 6]
    assert check(min_keyword_share=1.0).needed(3) == 3
    # a3 has 3 keywords: 2 of them (ceil(1.8) = 2) are enough, 1 is not.
    two = GOOD_AC.replace("invoice number, customer name and status", "customer name and status")
    assert check().score(_case(ATOMS), _result(_ticket(two)), _Ctx()).passed
    one = GOOD_AC.replace("invoice number, customer name and status", "status")
    score = check().score(_case(ATOMS), _result(_ticket(one)), _Ctx())
    assert not score.passed
    assert "atom s/a3 uncovered: 1/3 keywords (needs 2" in score.reason


def test_matching_is_case_insensitive_and_uses_title_and_description():
    ac = GOOD_AC.replace("a CSV file downloads", "a file downloads").replace("Export the invoice list as CSV", "Export")
    ac = ac.replace("they export the invoice list", "they export the INVOICE LIST")
    ticket = _ticket(ac, title="Invoice export", description="Delivered as a Csv.")
    assert check().score(_case(ATOMS), _result(ticket), _Ctx()).passed


def test_assumptions_and_open_questions_do_not_count():
    ac = AC_NO_90
    ticket = _ticket(ac, assumptions=["Only the last 90 days"], open_questions=["90 days?"])
    score = check().score(_case(ATOMS), _result(ticket), _Ctx())
    assert not score.passed and "atom s/a2 uncovered" in score.reason
    assert score.value == pytest.approx(2 / 3)


def test_hinted_atoms_are_not_required():
    ac = AC_NO_90
    by_list = _case(ATOMS, hinted=["s/a2"])
    by_mode = _case([*ATOMS[:1], {**ATOMS[1], "mode": "hinted"}, ATOMS[2]])
    for case in (by_list, by_mode):
        score = check().score(case, _result(_ticket(ac)), _Ctx())
        assert score.passed, score.reason
        hinted = [s for s in score.sub_scores if s.get("hinted")]
        assert hinted == [{"atom": "s/a2", "covered": False, "hinted": True, "matched": [],
                           "missing": ["90 days"], "needed": 1}]
        assert "1 hinted atom(s) not required" in score.reason


def test_invented_scenario_fails():
    ac = GOOD_AC + """
  Scenario: Only admins may open the page
    Given a staff member without rights
    When they open the page
    Then they see an error
"""
    score = check().score(_case(ATOMS), _result(_ticket(ac)), _Ctx())
    assert not score.passed
    assert "invented scenario 'Only admins may open the page'" in score.reason
    assert {"invented_scenario": "Only admins may open the page",
            "reason": "no atom keyword in name or steps"} in score.sub_scores


def test_invented_detection_ignores_step_keywords_but_reads_tables():
    atoms = [{"id": "a", "keywords": ["when"]}, {"id": "b", "keywords": ["sku"]}]
    ac = """Feature: X
  Scenario: Nothing relevant
    Given something
    When it happens
    Then fine

  Scenario Outline: Codes
    Given a product
    When I look up <code>
    Then it shows

    Examples:
      | code |
      | SKU1 |
"""
    score = check().score(_case(atoms), _result(_ticket(ac, title="t", description="d")), _Ctx())
    invented = [s["invented_scenario"] for s in score.sub_scores if "invented_scenario" in s]
    assert invented == ["Nothing relevant"]  # the "When" keyword is not step text; "SKU1" is a table cell


def test_unparseable_gherkin_still_scans_scenarios():
    ac = "Scenario: About CSV\n  Given x\n  When y\n  Then z\nScenario: Lunch\n  Given a\n"
    assert [name for name, _ in scenario_texts(ac)] == ["About CSV", "Lunch"]


def test_error_results_and_missing_atoms_fail():
    c = check()
    assert not c.score(_case(ATOMS), _result(None, error_kind=ErrorKind.SYSTEM, error="bad"), _Ctx()).passed
    no_atoms = Case(id="c1", input={}, provenance="real")
    score = c.score(no_atoms, _result(_ticket(GOOD_AC)), _Ctx())
    assert not score.passed and "no source atoms" in score.reason


@pytest.mark.parametrize("bad", [0, 0.0, 1.5, "0.6", True])
def test_rejects_bad_share(bad):
    with pytest.raises(TypeError):
        atom_keywords(min_keyword_share=bad)
