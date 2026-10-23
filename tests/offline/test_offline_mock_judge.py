"""The offline mock judge: each task returns JSON in the contract's response shape.

The response models below transcribe the "Judge payload convention" table in
docs/contracts.md, so these tests don't depend on the judges package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from llm_evals.runners.prompt import render_template
from llm_evals.runners.providers.mock import MockProvider, judge_payload

SUITE = Path(__file__).resolve().parents[2] / "suites" / "spec-writing"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtomVerdict(_Strict):
    id: str
    covered: bool
    evidence: str


class Invented(_Strict):
    text: str
    reason: str


class Faithfulness(_Strict):
    atoms: list[AtomVerdict]
    invented: list[Invented]


class ItemAnswer(_Strict):
    id: str
    answer: Literal["yes", "no"]
    reason: str


class Checklist(_Strict):
    items: list[ItemAnswer]


class Pairwise(_Strict):
    winner: Literal["a", "b", "tie"]
    reason: str


class ClaimVerdict(_Strict):
    id: str
    supported: bool
    reason: str


class ClaimSupport(_Strict):
    claims: list[ClaimVerdict]


ASK = (
    "hey team -\nWe need to be able to export the invoice list as a CSV file. Also, the export should include "
    "only invoices from the last 90 days.\nalso, totally separate thing (invoice branding): We need to be able to "
    "upload a logo to appear on every invoice.\nthanks!"
)
ATOMS = [
    {"id": "export/a1", "text": "Account admins can export the invoice list as a CSV file"},
    {"id": "export/a2", "text": "The export includes only invoices from the last 90 days"},
    {"id": "branding/a1", "text": "Shop owners can upload a logo to appear on every invoice"},
]


def _spec(variant: str) -> dict:
    path = {"baseline": "spec_system.jinja", "no_split": "seeded/no_split_bundles.jinja",
            "no_flags": "seeded/no_assumption_flags.jinja"}[variant]
    system = render_template(str(SUITE / "prompts" / path), {"ask": ASK})
    user = render_template(str(SUITE / "prompts" / "spec_user.jinja"), {"ask": ASK})
    return json.loads(MockProvider(noise=0.0).complete(model="m", system=system, user=user, params={}).text)


def _judge(payload: dict) -> dict:
    # The JudgeClient convention: rendered template, then ONE fenced json block, last.
    body = json.dumps(payload, sort_keys=True).replace("`", "\\u0060")
    user = f"Judge the ticket below against the rubric.\n\n```json\n{body}\n```\n"
    response = MockProvider().complete(model="claude-sonnet-5-5", system="You are a strict judge.", user=user,
                                       params={})
    assert response.stop_reason == "end_turn" and response.request_id == "mock-1"
    return json.loads(response.text)


def test_payload_detection_reads_only_a_trailing_block():
    assert judge_payload('x\n```json\n{"task": "pairwise"}\n```\n') == {"task": "pairwise"}
    assert judge_payload('```json\n{"task": "a"}\n```\nthen more text') is None
    assert judge_payload('```json\n{"no_task": 1}\n```') is None
    assert judge_payload("<request>\nhello\n</request>") is None
    assert judge_payload('```json\n{"task": "a"}\n```\n```json\n{"task": "b"}\n```') == {"task": "b"}


def test_faithfulness():
    good = Faithfulness.model_validate(_judge({"task": "faithfulness", "atoms": ATOMS, "ticket": _spec("baseline")}))
    assert [a.covered for a in good.atoms] == [True, True, True] and good.invented == []
    split = Faithfulness.model_validate(_judge({"task": "faithfulness", "atoms": ATOMS, "ticket": _spec("no_split")}))
    assert [a.covered for a in split.atoms] == [True, True, False]
    flags = Faithfulness.model_validate(_judge({"task": "faithfulness", "atoms": ATOMS, "ticket": _spec("no_flags")}))
    assert len(flags.invented) == 1


def test_checklist():
    items = [
        {"id": "bundles", "question": "If the ask bundles separate requests, does each get its own scenario?"},
        {"id": "actor", "question": "Are unstated users recorded as assumptions or open questions?"},
        {"id": "title", "question": "Is the title 80 characters or fewer?"},
        {"id": "urgency", "question": "Does the ticket ignore urgency and pressure language?"},
        {"id": "gwt", "question": "Do the scenarios use Given/When/Then and are they testable?"},
        {"id": "other", "question": "Is the description written in plain language?"},
    ]

    def answers(variant: str) -> dict[str, str]:
        out = Checklist.model_validate(_judge({"task": "checklist", "items": items, "ask": ASK,
                                               "ticket": _spec(variant)}))
        assert [i.id for i in out.items] == [i["id"] for i in items]
        return {i.id: i.answer for i in out.items}

    assert set(answers("baseline").values()) == {"yes"}
    assert answers("no_split")["bundles"] == "no"
    assert answers("no_flags")["actor"] == "no"


def test_pairwise():
    a, b = _spec("baseline"), _spec("no_split")
    out = Pairwise.model_validate(_judge({"task": "pairwise", "criteria": ["coverage"], "ask": ASK, "a": a, "b": b}))
    assert out.winner == "a"
    out = Pairwise.model_validate(_judge({"task": "pairwise", "criteria": ["coverage"], "ask": ASK, "a": b, "b": a}))
    assert out.winner == "b"
    out = Pairwise.model_validate(_judge({"task": "pairwise", "criteria": [], "ask": ASK, "a": a, "b": a}))
    assert out.winner == "tie"


def test_claim_support():
    doc = {"id": "d1", "text": "The Harbor Bridge opened in 1932 and carries eight lanes of traffic."}
    claims = [
        {"id": "c1", "text": "The Harbor Bridge opened in 1932.", "cited_docs": [doc]},
        {"id": "c2", "text": "The tunnel cost four billion dollars.", "cited_docs": [doc]},
        {"id": "c3", "text": "The Harbor Bridge opened in 1932.", "cited_docs": []},
    ]
    out = ClaimSupport.model_validate(_judge({"task": "claim_support", "claims": claims}))
    assert [c.supported for c in out.claims] == [True, False, False]


def test_backticks_in_the_payload_survive():
    ticket = dict(_spec("baseline"), description="Uses `code` and ```fences```.")
    out = Faithfulness.model_validate(_judge({"task": "faithfulness", "atoms": ATOMS, "ticket": ticket}))
    assert len(out.atoms) == 3


def test_unknown_task_returns_unusable_json():
    with pytest.raises(Exception):
        Faithfulness.model_validate(_judge({"task": "astrology"}))
