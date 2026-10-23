"""`faithfulness`: an LLM judge checks the ticket against the case's requirement atoms.

The judge sees the non-hinted atoms (`case.source["atoms"]` whose `mode` isn't
"hinted", and whose id isn't in `source["hinted_atom_ids"]`), the ask and the
ticket, and answers per atom whether the ticket covers it, plus a list of
invented requirements. The check passes when every atom is covered and
nothing is invented. `value` is the share of atoms covered.

Payload (task "faithfulness"): `{"task", "atoms": [{"id", "text"}], "ask", "ticket"}`
(`ask` is an extra key beyond the contract table; it lets the judge tell an
invented requirement from one the ask stated or hinted at).

Sub-scores, in order:
- `{"unit": "faithfulness", "passed", "prompt_hash", "call_hash", "covered", "total", "invented"}`
- one `{"atom": id, "text", "covered", "evidence"}` per atom
- `{"elaborations": [{"text", "reason"}]}`: additions the judge counted as sensible; never a failure
- `{"invented": [{"text", "reason"}]}`

A judge answer that skips an atom counts that atom as not covered (evidence
"judge gave no verdict"). Unusable judge answers fail with "judge error: ...",
provider outages with "infra: ...". A case with no `source.atoms` fails with
"check error: ...".

Paths (`template`, `system_template`) are relative to the suite dir; see
`llm_evals.judges.base` for how `suite_dir` reaches the check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_evals.checks._base import BaseCheck
from llm_evals.judges.base import JudgeCheckMixin
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, GatingUnit, Score

UNIT_KEY = "faithfulness"


class AtomVerdict(BaseModel):
    id: str
    covered: bool
    evidence: str = ""


class InventedRequirement(BaseModel):
    text: str
    reason: str = ""


class FaithfulnessResponse(BaseModel):
    atoms: list[AtomVerdict]
    elaborations: list[InventedRequirement] = Field(default_factory=list)
    invented: list[InventedRequirement] = Field(default_factory=list)


def judged_atoms(case: Case) -> list[dict[str, str]]:
    """The non-hinted atoms of a case as `{id, text}`. Raises ValueError without atoms."""
    source = case.source or {}
    atoms = source.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError(f"case {case.id!r} has no source.atoms to judge faithfulness against")
    hinted = set(source.get("hinted_atom_ids") or [])
    out: list[dict[str, str]] = []
    for atom in atoms:
        if not isinstance(atom, dict) or "id" not in atom or "text" not in atom:
            raise ValueError(f"case {case.id!r} has a malformed atom: {atom!r}")
        if atom.get("mode") == "hinted" or atom["id"] in hinted:
            continue
        out.append({"id": str(atom["id"]), "text": str(atom["text"])})
    return out


class Faithfulness(JudgeCheckMixin, BaseCheck):
    name = "faithfulness"
    kind: CheckKind = "faithfulness"

    def __init__(
        self, *, template: str, system_template: str | None, suite_dir: str | Path | None, critical: bool
    ) -> None:
        BaseCheck.__init__(self, critical=critical)
        self._init_judge(template=template, system_template=system_template, suite_dir=suite_dir)

    def unit_keys(self) -> list[str]:
        return [UNIT_KEY]

    def gating_units(self) -> list[GatingUnit]:
        return [GatingUnit(key=UNIT_KEY, judge_model=self.judge_model, prompt_hash=self.unit_hash())]

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        atoms = judged_atoms(case)
        payload = {"task": "faithfulness", "atoms": atoms, "ask": case.input.get("ask", ""), "ticket": output}
        res = self._call(payload, FaithfulnessResponse)
        verdicts = {v.id: v for v in res.parsed.atoms}

        details: list[dict[str, Any]] = []
        uncovered: list[str] = []
        for atom in atoms:
            v = verdicts.get(atom["id"])
            covered = bool(v and v.covered)
            evidence = v.evidence if v else "judge gave no verdict"
            details.append({"atom": atom["id"], "text": atom["text"], "covered": covered, "evidence": evidence})
            if not covered:
                uncovered.append(atom["id"])
        invented = [i.model_dump() for i in res.parsed.invented]
        passed = not uncovered and not invented
        n_covered = len(atoms) - len(uncovered)
        unit_hash = self.unit_hash()
        unit = {
            "unit": UNIT_KEY,
            "passed": passed,
            "prompt_hash": unit_hash,
            "call_hash": res.prompt_hash,
            "covered": n_covered,
            "total": len(atoms),
            "invented": len(invented),
        }
        elaborations = [e.model_dump() for e in res.parsed.elaborations]
        sub_scores = [unit, *details, {"elaborations": elaborations}, {"invented": invented}]

        problems: list[str] = []
        if uncovered:
            problems.append(f"{len(uncovered)} of {len(atoms)} atom(s) not covered: {', '.join(uncovered)}")
        if invented:
            problems.append(
                f"{len(invented)} invented requirement(s): " + "; ".join(repr(i["text"]) for i in invented)
            )
        reason = "; ".join(problems) if problems else f"all {len(atoms)} atom(s) covered, nothing invented"
        return Score(
            check=self.name,
            kind=self.kind,
            passed=passed,
            value=n_covered / len(atoms) if atoms else 1.0,
            reason=reason,
            sub_scores=sub_scores,
            judge_model=res.model,
            prompt_hash=unit_hash,
        )


@register_check("faithfulness")
def faithfulness(
    *,
    template: str = "judges/faithfulness.jinja",
    system_template: str | None = "judges/faithfulness_system.jinja",
    suite_dir: str | Path | None = None,
    critical: bool = False,
) -> Faithfulness:
    return Faithfulness(template=template, system_template=system_template, suite_dir=suite_dir, critical=critical)
