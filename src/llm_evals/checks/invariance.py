"""`invariance`: a reworded ask should produce a structurally similar ticket.

The check rewords the case's ask with
`llm_evals.cases.renderers.ask_renderer.reword_ask(case, seed)`, reruns the
same version on the new input through `ctx.rerun`, and passes when:
- the rerun returned output;
- both outputs pass `gherkin_parses`;
- their scenario counts differ by at most `max_scenario_delta`;
- both agree on whether `open_questions` is empty.

This is a structural proxy for invariance: it compares shape, not meaning.
Two tickets can have the same scenario count and still cover different
requirements. Semantic invariance (do both outputs recall the same
requirement atoms?) arrives with the Phase 2 atom-recall check.

Only an INFRA error on the rerun fails with a reason starting "infra:", so the
engine can exclude the sample instead of counting it against the version. An
exception from the reworder or the rerun fails with "check error:" and counts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_evals.checks._base import BaseCheck, no_output_reason, truncate
from llm_evals.checks.gherkin import analyze_gherkin
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, ErrorKind, Score

RELATIONS = ("reword_ask",)


def _questions_empty(output: dict[str, Any]) -> bool:
    questions = output.get("open_questions")
    return not questions


def _as_input(rewritten: Any) -> dict[str, Any]:
    """Accept a new input dict, a Case, or a bare ask string from the reworder."""
    if isinstance(rewritten, Case):
        return dict(rewritten.input)
    if isinstance(rewritten, dict):
        return rewritten
    if isinstance(rewritten, str):
        return {"ask": rewritten}
    raise TypeError(f"reword_ask returned {type(rewritten).__name__}, expected a dict input")


def _load_reword_ask() -> Callable[[Case, int], Any]:
    # Imported lazily so this package loads even before the renderer exists.
    from llm_evals.cases.renderers.ask_renderer import reword_ask

    return reword_ask


class Invariance(BaseCheck):
    name = "invariance"
    kind: CheckKind = "metamorphic"
    reruns_per_sample = 1  # one extra model call per scored sample, for the cost estimate

    def __init__(self, *, relation: str, seed: int, max_scenario_delta: int, critical: bool) -> None:
        super().__init__(critical=critical)
        if relation not in RELATIONS:
            raise ValueError(f"invariance: unknown relation {relation!r} (known: {', '.join(RELATIONS)})")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"invariance: seed must be an int, got {seed!r}")
        if not isinstance(max_scenario_delta, int) or isinstance(max_scenario_delta, bool) or max_scenario_delta < 0:
            raise TypeError(f"invariance: max_scenario_delta must be a non-negative int, got {max_scenario_delta!r}")
        self.relation = relation
        self.seed = seed
        self.max_scenario_delta = max_scenario_delta

    def _transform(self, case: Case) -> dict[str, Any]:
        return _as_input(_load_reword_ask()(case, self.seed))

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        # A broken reworder or rerun is a bug in the pipeline, not flaky infra: fail
        # loudly so it shows in the report instead of silently excluding samples.
        try:
            new_input = self._transform(case)
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"check error: could not build the {self.relation} input: {type(exc).__name__}: {exc}")

        try:
            rerun = ctx.rerun(case, new_input)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "FatalRunError":  # config problem: abort the run
                raise
            return self.fail(f"check error: rerun raised {type(exc).__name__}: {exc}")

        debug: dict[str, Any] = {
            "role": "rerun",
            "relation": self.relation,
            "seed": self.seed,
            "input": truncate(new_input, 1000),
            "error_kind": rerun.error_kind.value,
            "output": truncate(rerun.output if rerun.output is not None else rerun.raw_output, 2000),
        }
        if rerun.error_kind == ErrorKind.INFRA:
            return self.fail(f"infra: rerun failed ({rerun.error or 'unspecified'})", sub_scores=[debug])
        missing = no_output_reason(rerun)
        if missing is not None:
            return self.fail(f"rerun has {missing}", sub_scores=[debug])
        rerun_output: dict[str, Any] = rerun.output or {}

        original = analyze_gherkin(output.get("acceptance_criteria"))
        reworded = analyze_gherkin(rerun_output.get("acceptance_criteria"))
        orig_n, new_n = len(original.scenarios), len(reworded.scenarios)
        orig_empty, new_empty = _questions_empty(output), _questions_empty(rerun_output)
        debug.update(scenarios=new_n, open_questions_empty=new_empty, gherkin_ok=reworded.ok)
        summary = {
            "role": "original",
            "scenarios": orig_n,
            "open_questions_empty": orig_empty,
            "gherkin_ok": original.ok,
        }
        sub_scores = [summary, debug]

        problems: list[str] = []
        if not original.ok:
            problems.append(f"original Gherkin invalid ({original.parse_error or '; '.join(original.problems)})")
        if not reworded.ok:
            problems.append(f"reworded Gherkin invalid ({reworded.parse_error or '; '.join(reworded.problems)})")
        if original.ok and reworded.ok and abs(orig_n - new_n) > self.max_scenario_delta:
            problems.append(f"scenario count changed {orig_n} -> {new_n} (max delta {self.max_scenario_delta})")
        if orig_empty != new_empty:
            problems.append(
                "open_questions emptiness changed "
                f"({'empty' if orig_empty else 'non-empty'} -> {'empty' if new_empty else 'non-empty'})"
            )

        if problems:
            return self.fail("; ".join(problems), sub_scores=sub_scores)
        return Score(
            check=self.name,
            kind=self.kind,
            passed=True,
            value=None,
            reason=(
                f"structure held under {self.relation}: scenarios {orig_n} -> {new_n}, "
                f"open_questions {'empty' if orig_empty else 'non-empty'} in both"
            ),
            sub_scores=sub_scores,
        )


@register_check("invariance")
def invariance(
    *,
    relation: str = "reword_ask",
    seed: int = 0,
    max_scenario_delta: int = 1,
    critical: bool = False,
) -> Invariance:
    return Invariance(relation=relation, seed=seed, max_scenario_delta=max_scenario_delta, critical=critical)
