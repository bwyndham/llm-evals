"""A custom LLM judge, as a suite plug-in (docs/adopting.md, "Your own judge").

It asks the judge one question about each output and reports a single gating
unit, `polite_reply`. Like every judge it is scored on every run but gates only
once `llm-evals calibration` has approved that unit against human labels for
this exact judge model and prompt.

Use it from suite.yaml:

    plugins: [plugins/polite_judge.py]
    checks:
      - polite_reply: {}

with the two prompt templates next to the suite at judges/polite.jinja and
judges/polite_system.jinja. The system template must ask for the JSON shape of
`Verdict` below. The payload is appended to the user prompt as a JSON block.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_evals.api import BaseCheck, GatingUnit, JudgeCheckMixin, Score, register_check

UNIT = "polite_reply"


class Verdict(BaseModel):
    polite: bool
    reason: str = ""


class PoliteReply(JudgeCheckMixin, BaseCheck):
    name = "polite_reply"
    kind = "judge"

    def __init__(self, *, template: str, system_template: str | None, suite_dir: str | None, critical: bool) -> None:
        BaseCheck.__init__(self, critical=critical)
        self._init_judge(template=template, system_template=system_template, suite_dir=suite_dir)

    def unit_keys(self) -> list[str]:
        return [UNIT]

    def gating_units(self) -> list[GatingUnit]:
        return [GatingUnit(key=UNIT, judge_model=self.judge_model, prompt_hash=self.unit_hash())]

    def _score(self, case, output, ctx) -> Score:
        res = self._call({"task": "polite", "input": case.input, "output": output}, Verdict)
        prompt_hash = self.unit_hash()
        return Score(
            check=self.name, kind=self.kind, passed=res.parsed.polite, reason=res.parsed.reason,
            sub_scores=[{"unit": UNIT, "passed": res.parsed.polite, "prompt_hash": prompt_hash}],
            judge_model=res.model, prompt_hash=prompt_hash,
        )


@register_check("polite_reply")
def polite_reply(*, template: str = "judges/polite.jinja", system_template: str | None = "judges/polite_system.jinja",
                 suite_dir: str | None = None, critical: bool = False) -> PoliteReply:
    return PoliteReply(template=template, system_template=system_template, suite_dir=suite_dir, critical=critical)
