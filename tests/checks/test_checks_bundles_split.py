"""bundles_split: bundled requests must each get their own scenario."""

from __future__ import annotations

from llm_evals.checks.bundles_split import bundles_split
from llm_evals.types import Case, Result


def _case(bundled: bool = True) -> Case:
    atoms = [
        {"id": "exp/a1", "set_id": "exp", "keywords": ["export", "csv"], "mode": "explicit"},
        {"id": "exp/a2", "set_id": "exp", "keywords": ["90 days"], "mode": "explicit"},
        {"id": "rem/a1", "set_id": "rem", "keywords": ["reminder", "overdue"], "mode": "explicit"},
        {"id": "rem/a2", "set_id": "rem", "keywords": ["sms"], "mode": "hinted"},
    ]
    if not bundled:
        atoms = atoms[:2]
    return Case(id="c", input={"ask": "..."}, provenance="constructed",
                source={"atoms": atoms, "atom_set_ids": ["exp", "rem"] if bundled else ["exp"],
                        "hinted_atom_ids": ["rem/a2"]})


def _result(ac: str) -> Result:
    return Result(run_id="r", case_id="c", version_id="v", sample_idx=0,
                  output={"title": "t", "description": "d", "acceptance_criteria": ac,
                          "assumptions": [], "open_questions": []})


SPLIT = """Feature: Billing
Scenario: Export invoices as CSV
  Given invoices exist
  When the admin exports the list as CSV
  Then the file covers the last 90 days
Scenario: Overdue reminder
  Given an invoice is overdue
  When the due date passes
  Then a reminder is emailed"""

MERGED = """Feature: Billing
Scenario: Export and reminders
  Given invoices exist
  When the admin exports the list as CSV and an invoice is overdue
  Then the file covers the last 90 days and a reminder is sent"""

ONE_MISSING = """Feature: Billing
Scenario: Export invoices as CSV
  Given invoices exist
  When the admin exports the list as CSV
  Then the file covers the last 90 days"""


def test_split_bundle_passes():
    s = bundles_split().score(_case(), _result(SPLIT), None)
    assert s.passed, s.reason


def test_merged_bundle_fails():
    s = bundles_split().score(_case(), _result(MERGED), None)
    assert not s.passed and "merged" in s.reason


def test_dropped_request_fails():
    s = bundles_split().score(_case(), _result(ONE_MISSING), None)
    assert not s.passed and "rem" in s.reason


def test_not_bundled_is_not_applicable():
    s = bundles_split().score(_case(bundled=False), _result(MERGED), None)
    assert s.passed and s.reason.startswith("not applicable")


def test_unparseable_gherkin_is_left_to_gherkin_parses():
    s = bundles_split().score(_case(), _result("Feature: x\nnot gherkin at all\nFeature: y"), None)
    assert s.passed and "gherkin_parses" in s.reason
