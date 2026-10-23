"""`pairwise`: an LLM judge prefers the baseline (a) or candidate (b) ticket.

`compare(case, a, b)` asks the judge twice, once with the outputs in each
order (task "pairwise", payload `{"task", "criteria": [rubric questions], "ask",
"a", "b"}`), and maps the second answer back to the original labels. It
returns that winner only when both orders pick the same underlying output;
otherwise (position bias, a tie in either order) it returns a tie.

Edge cases, all returned as Preferences rather than raised:
- neither result has output: tie; only one has output: that one wins;
- an unusable judge answer: tie with reason "judge error: ...";
- a provider outage: tie with reason "infra: ...".
An unbound judge or a missing suite dir raises (config problems).

There is no CheckContext here, so pass `suite_dir` (the engine does) or use
absolute paths with `suite_dir="/"`. `win_rate` summarises many preferences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from llm_evals.checks._base import no_output_reason
from llm_evals.judges.base import JudgeBinding, JudgeFailure, load_rubric
from llm_evals.registry import register_pairwise
from llm_evals.types import Case, GatingUnit, Preference, Result

_FLIP = {"a": "b", "b": "a", "tie": "tie"}


class PairwiseResponse(BaseModel):
    winner: Literal["a", "b", "tie"]
    reason: str = ""

    @field_validator("winner", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class Pairwise(JudgeBinding):
    name = "pairwise"

    def __init__(
        self, *, rubric: str, template: str, system_template: str | None, suite_dir: str | Path | None
    ) -> None:
        self.rubric = rubric
        self.rubric_path: Path | None = None
        self.criteria: list[str] = []
        self._init_judge(template=template, system_template=system_template, suite_dir=suite_dir)

    def _resolve(self, suite_dir: Path) -> None:
        super()._resolve(suite_dir)
        self.rubric_path = self._path(suite_dir, self.rubric)
        self.criteria = [item["question"] for item in load_rubric(self.rubric_path)]

    def gating_units(self) -> list[GatingUnit]:
        return []  # preferences are reported, never gated

    def _ask(self, case: Case, first: dict[str, Any], second: dict[str, Any]) -> PairwiseResponse:
        payload = {
            "task": "pairwise",
            "criteria": self.criteria,
            "ask": case.input.get("ask", ""),
            "a": first,
            "b": second,
        }
        return self._call(payload, PairwiseResponse).parsed

    def compare(self, case: Case, a: Result, b: Result) -> Preference:
        self._require_judge()
        self._ensure_resolved()
        a_missing, b_missing = no_output_reason(a), no_output_reason(b)
        if a_missing and b_missing:
            return Preference(winner="tie", reason=f"neither output usable (a: {a_missing}; b: {b_missing})")
        if a_missing:
            return Preference(winner="b", reason=f"a has {a_missing}")
        if b_missing:
            return Preference(winner="a", reason=f"b has {b_missing}")
        out_a, out_b = a.output or {}, b.output or {}
        try:
            forward = self._ask(case, out_a, out_b)
            reverse = self._ask(case, out_b, out_a)
        except JudgeFailure as exc:
            return Preference(winner="tie", reason=exc.reason)
        reverse_winner = _FLIP[reverse.winner]  # back to the original labels
        if forward.winner == reverse_winner:
            return Preference(winner=forward.winner, reason=f"both orders agree: {forward.reason}")
        return Preference(
            winner="tie",
            reason=(
                f"orders disagree (a-first picked {forward.winner}, b-first picked {reverse_winner}): "
                f"{forward.reason} | {reverse.reason}"
            ),
        )


def win_rate(prefs: list[Preference]) -> dict[str, Any]:
    """Candidate (b) wins, losses and ties over many preferences.

    `win_share` / `loss_share` / `tie_share` are over all n; `win_rate`
    counts a tie as half a win; `decisive_win_share` is wins / (wins + losses)
    (None with no decisive preferences). Shares are None for an empty list.
    """
    n = len(prefs)
    wins = sum(p.winner == "b" for p in prefs)
    losses = sum(p.winner == "a" for p in prefs)
    ties = n - wins - losses
    decisive = wins + losses
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_share": wins / n if n else None,
        "loss_share": losses / n if n else None,
        "tie_share": ties / n if n else None,
        "win_rate": (wins + 0.5 * ties) / n if n else None,
        "decisive_win_share": wins / decisive if decisive else None,
    }


@register_pairwise("pairwise")
def pairwise(
    *,
    rubric: str = "rubrics/spec_standard.yaml",
    template: str = "judges/pairwise.jinja",
    system_template: str | None = "judges/pairwise_system.jinja",
    suite_dir: str | Path | None = None,
) -> Pairwise:
    return Pairwise(rubric=rubric, template=template, system_template=system_template, suite_dir=suite_dir)
