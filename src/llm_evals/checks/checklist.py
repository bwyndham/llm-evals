"""`checklist`: an LLM judge answers a yes/no rubric about the ticket.

One judge call per sample covers every rubric item (task "checklist",
payload `{"task", "items": [{"id", "question"}], "ask", "ticket"}`). Each item is
its own gating unit, `checklist:<id>`, whose prompt hash is
`unit_hash(template, question)` (plus the system template's bytes), so
rewording one question resets only that item's calibration.

Every rubric question is phrased so "yes" is the good answer. The check
passes when every item is answered "yes"; `value` is the share of "yes".

Sub-scores: one per item, in rubric order:
`{"unit": "checklist:<id>", "passed", "answer": "yes"|"no"|None, "reason", "question", "prompt_hash", "call_hash"}`.
An item the judge skipped fails with answer None and reason "judge gave no answer".

Paths (`rubric`, `template`, `system_template`) are relative to the suite dir;
see `llm_evals.judges.base` for how `suite_dir` reaches the check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from llm_evals.checks._base import BaseCheck
from llm_evals.judges.base import JudgeCheckMixin, load_rubric
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, GatingUnit, Score


class ItemAnswer(BaseModel):
    id: str
    answer: Literal["yes", "no"]
    reason: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class ChecklistResponse(BaseModel):
    items: list[ItemAnswer]


class Checklist(JudgeCheckMixin, BaseCheck):
    name = "checklist"
    kind: CheckKind = "judge"

    def __init__(
        self,
        *,
        rubric: str,
        template: str,
        system_template: str | None,
        suite_dir: str | Path | None,
        critical: bool,
    ) -> None:
        BaseCheck.__init__(self, critical=critical)
        self.rubric = rubric
        self.rubric_path: Path | None = None
        self.items: list[dict[str, str]] = []
        self._init_judge(template=template, system_template=system_template, suite_dir=suite_dir)

    def _resolve(self, suite_dir: Path) -> None:
        super()._resolve(suite_dir)
        self.rubric_path = self._path(suite_dir, self.rubric)
        self.items = load_rubric(self.rubric_path)

    @staticmethod
    def unit_key(item_id: str) -> str:
        return f"checklist:{item_id}"

    def unit_keys(self) -> list[str]:
        return [self.unit_key(item["id"]) for item in self.items]

    def gating_units(self) -> list[GatingUnit]:
        self._ensure_resolved()
        model = self.judge_model
        return [
            GatingUnit(key=self.unit_key(item["id"]), judge_model=model, prompt_hash=self.unit_hash(item["question"]))
            for item in self.items
        ]

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        if "ask" in case.input:  # the spec-writing shape; kept byte-identical so its judge cache holds
            payload = {"task": "checklist", "items": self.items, "ask": case.input["ask"], "ticket": output}
        else:  # any other suite: the whole case input and the output under neutral names
            payload = {"task": "checklist", "items": self.items, "input": case.input, "output": output}
        res = self._call(payload, ChecklistResponse)
        answers = {a.id: a for a in res.parsed.items}

        sub_scores: list[dict[str, Any]] = []
        failed: list[str] = []
        for item in self.items:
            a = answers.get(item["id"])
            passed = a is not None and a.answer == "yes"
            if not passed:
                failed.append(item["id"])
            sub_scores.append(
                {
                    "unit": self.unit_key(item["id"]),
                    "passed": passed,
                    "answer": a.answer if a else None,
                    "reason": a.reason if a else "judge gave no answer",
                    "question": item["question"],
                    "prompt_hash": self.unit_hash(item["question"]),
                    "call_hash": res.prompt_hash,
                }
            )
        n = len(self.items)
        n_yes = n - len(failed)
        reason = (
            f"all {n} checklist item(s) pass"
            if not failed
            else f"{len(failed)} of {n} checklist item(s) fail: {', '.join(failed)}"
        )
        return Score(
            check=self.name,
            kind=self.kind,
            passed=not failed,
            value=n_yes / n,
            reason=reason,
            sub_scores=sub_scores,
            judge_model=res.model,
            prompt_hash=self.unit_hash(),  # the template's hash; per-item hashes are in sub_scores
        )


@register_check("checklist")
def checklist(
    *,
    rubric: str = "rubrics/spec_standard.yaml",
    template: str = "judges/checklist.jinja",
    system_template: str | None = "judges/checklist_system.jinja",
    suite_dir: str | Path | None = None,
    critical: bool = False,
) -> Checklist:
    return Checklist(
        rubric=rubric, template=template, system_template=system_template, suite_dir=suite_dir, critical=critical
    )
