"""Fixtures and fakes shared by the checks tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from llm_evals.types import Case, Result

GOOD_GHERKIN = """\
Feature: Export usage report
  Background:
    Given I am signed in as an account admin

  Scenario: Admin exports the report as CSV
    Given the usage report has data for the last 90 days
    And I am on the usage report page
    When I click "Export CSV"
    Then a CSV file downloads
    And it covers the last 90 days

  Scenario Outline: Non-admins cannot export
    Given I am signed in as a <role>
    When I open the usage report page
    Then I do not see the "Export CSV" button

    Examples:
      | role   |
      | viewer |
      | editor |
"""

MISSING_THEN = """\
Feature: Export usage report
  Scenario: Admin exports the report
    Given I am an admin
    When I click "Export CSV"
    And I wait
"""

TWO_FEATURES = """\
Feature: Export usage report
Feature: Import usage report
  Scenario: Admin exports the report
    Given I am an admin
    When I click "Export CSV"
    Then a CSV file downloads
"""

SYNTAX_ERROR = """\
Feature: Export usage report
  Scenario: Admin exports the report
    Given I am an admin
    When I click "Export CSV"
  this line is not Gherkin
    Then a CSV file downloads
"""

NO_SCENARIOS = "Feature: Export usage report\n  Just a description.\n"

DROP = object()  # sentinel: remove this key in make_output

GOOD_OUTPUT: dict[str, Any] = {
    "title": "Let admins export the usage report as CSV",
    "description": "Account admins need a CSV export of the last 90 days of usage.",
    "acceptance_criteria": GOOD_GHERKIN,
    "assumptions": ["CSV uses UTF-8 encoding"],
    "open_questions": ["Should the export include deleted users?"],
}


def make_output(**overrides: Any) -> dict[str, Any]:
    out = copy.deepcopy(GOOD_OUTPUT)
    for key, value in overrides.items():
        if value is DROP:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def make_result(output: dict[str, Any] | None = None, **fields: Any) -> Result:
    base: dict[str, Any] = {"run_id": "r1", "case_id": "c1", "version_id": "v1", "sample_idx": 0, "output": output}
    base.update(fields)
    return Result(**base)


class FakeContext:
    """A CheckContext whose rerun returns scripted results and records calls."""

    def __init__(self, *results: Result | Exception) -> None:
        self.suite_dir = Path(".")
        self._results = list(results)
        self.calls: list[tuple[Case, dict[str, Any]]] = []

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:
        self.calls.append((case, new_input))
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
