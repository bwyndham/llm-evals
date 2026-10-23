"""Fixtures and fakes shared by the judge tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_evals.types import Case, Result

REPO = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO / "suites" / "spec-writing"

_FENCE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)

TICKET: dict[str, Any] = {
    "title": "Let bookkeepers send customer statements of open invoices",
    "description": "Bookkeepers need a statement of a customer's open invoices with aging buckets.",
    "acceptance_criteria": (
        "Feature: Customer statements\n"
        "  Scenario: Generate a statement\n"
        "    Given a customer with open invoices\n"
        "    When the bookkeeper generates a statement\n"
        "    Then it lists every open invoice\n"
    ),
    "assumptions": [],
    "open_questions": [],
}


def payload_of(call: dict[str, Any]) -> dict[str, Any]:
    """The JSON payload in the last fenced json block of a provider call's user prompt."""
    blocks = _FENCE.findall(call["user"])
    assert blocks, "no fenced json payload in the user prompt"
    return json.loads(blocks[-1])


def make_case(atoms: list[dict[str, Any]] | None = None, **source_extra: Any) -> Case:
    if atoms is None:
        atoms = [
            {"id": "set/a1", "text": "Bookkeepers can generate a statement of open invoices", "mode": "explicit"},
            {"id": "set/a2", "text": "The statement shows 0-30, 31-60, 61-90 and 90+ day aging", "mode": "vague"},
            {"id": "set/a3", "text": "Statements can be emailed to the billing contact", "mode": "hinted"},
        ]
    return Case(
        id="c1",
        input={"ask": "can bookkeepers get a statement of open invoices w/ aging buckets?"},
        source={"atoms": atoms, **source_extra},
        provenance="constructed",
    )


def make_result(output: dict[str, Any] | None = TICKET, **fields: Any) -> Result:
    base: dict[str, Any] = {"run_id": "r1", "case_id": "c1", "version_id": "v1", "sample_idx": 0, "output": output}
    base.update(fields)
    return Result(**base)


class Ctx:
    def __init__(self, suite_dir: Path = SUITE_DIR) -> None:
        self.suite_dir = suite_dir

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:  # pragma: no cover
        raise AssertionError("judge checks never rerun")
