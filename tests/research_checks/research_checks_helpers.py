"""Builders and fakes shared by the research-check tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_evals.checks import research
from llm_evals.types import Case, Result, Trace

# Captured at import, before the autouse fixture in conftest.py swaps it out.
REAL_CORPUS_MATCHER = research._corpus_matcher

DOCS = [
    {"id": "d1", "title": "Harbor Town census", "text": "The 2020 census counted 12,500 people in Harbor Town."},
    {"id": "d2", "title": "Harbor Town history", "text": "Harbor Town was founded in 1852 by fishermen."},
    {"id": "d3", "title": "Lighthouse", "text": "The lighthouse is 31 metres tall."},
]


def make_case(
    *,
    behavior: str = "answer",
    answer: str | None = "12,500",
    aliases: list[str] | None = None,
    original_answer: str | None = None,
    op: str = "none",
    perturbation: dict[str, Any] | None = None,
    docs: list[dict[str, Any]] | None = DOCS,
    **expected_extra: Any,
) -> Case:
    pert = {"op": op, **(perturbation or {})}
    fixtures = None if docs is None else {"corpus_id": "harbor", "docs": docs}
    return Case(
        id="rc1",
        input={"question": "How many people live in Harbor Town?"},
        fixtures=fixtures,
        expected={
            "behavior": behavior,
            "answer": answer,
            "aliases": aliases or [],
            "original_answer": original_answer,
            **expected_extra,
        },
        source={"facts": [], "perturbation": pert},
        provenance="constructed",
    )


def make_output(
    *,
    status: str = "answered",
    answer: str = "12,500",
    claims: list[dict[str, Any]] | None = None,
    conflicts: list[str] | None = None,
) -> dict[str, Any]:
    if claims is None:
        claims = [{"text": "Harbor Town has 12,500 people.", "citations": ["d1"]}]
    return {"status": status, "answer": answer, "claims": claims, "conflicts": conflicts or []}


def search(*doc_ids: str, query: str = "harbor town") -> dict[str, Any]:
    return {"name": "search", "input": {"query": query}, "doc_ids": list(doc_ids)}


def fetch(doc_id: str) -> dict[str, Any]:
    return {"name": "fetch", "input": {"doc_id": doc_id}, "doc_ids": [doc_id]}


def make_result(output: dict[str, Any] | None, tool_calls: list[dict[str, Any]] | None = None, **fields: Any) -> Result:
    base: dict[str, Any] = {
        "run_id": "r1",
        "case_id": "rc1",
        "version_id": "v1",
        "sample_idx": 0,
        "output": output,
        "trace": Trace(tool_calls=tool_calls if tool_calls is not None else [search("d1", "d2"), fetch("d1")]),
    }
    base.update(fields)
    return Result(**base)


class FakeContext:
    def __init__(self, suite_dir: Path | str = ".") -> None:
        self.suite_dir = Path(suite_dir)

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:  # pragma: no cover - unused
        raise AssertionError("research checks never rerun")


class JudgeError(Exception):
    """Stands in for llm_evals.judges.client.JudgeError (matched by class name)."""


class FakeJudge:
    """Answers claim_support payloads from a verdict table and records each call."""

    def __init__(self, verdicts: dict[str, bool] | None = None, *, error: Exception | None = None,
                 parsed: Any = None, model: str = "claude-judge-test") -> None:
        self.model = model
        self.verdicts = verdicts or {}
        self.error = error
        self.parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def judge(self, system_template, user_template, variables, payload, response_model):
        self.calls.append({
            "system_template": system_template,
            "user_template": user_template,
            "variables": variables,
            "payload": payload,
            "response_model": response_model,
        })
        if self.error is not None:
            raise self.error
        parsed = self.parsed
        if parsed is None:
            parsed = response_model.model_validate({
                "claims": [
                    {"id": c["id"], "supported": self.verdicts.get(c["id"], True), "reason": f"verdict for {c['id']}"}
                    for c in payload["claims"]
                ]
            })
        return SimpleNamespace(parsed=parsed, raw="{}", prompt_hash="ph", cost_usd=0.002)
