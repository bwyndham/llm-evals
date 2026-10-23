"""docs/examples/polite_judge.py works as written: loads as a plug-in, calls the judge,
and exposes a gating unit for calibration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_evals import registry
from llm_evals.judges import JudgeClient
from llm_evals.runners.providers.base import FakeProvider
from llm_evals.types import Case, Result

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "polite_judge.py"


class Ctx:
    def __init__(self, d):
        self.suite_dir = d


def test_polite_judge_example(tmp_path):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "judges").mkdir()
    shutil.copy(EXAMPLE, tmp_path / "plugins" / "polite_judge.py")
    (tmp_path / "judges" / "polite.jinja").write_text("Is the reply polite?\n")
    (tmp_path / "judges" / "polite_system.jinja").write_text('Answer with {"polite": true|false, "reason": "..."}\n')
    registry.load_plugins(["plugins/polite_judge.py"], tmp_path)

    check = registry.get("check", "polite_reply")(suite_dir=str(tmp_path))
    check.bind_judge(JudgeClient(FakeProvider.from_texts([json.dumps({"polite": False, "reason": "curt"})])))
    case = Case(id="c", input={"message": "help"}, provenance="constructed")
    score = check.score(case, Result(run_id="r", case_id="c", version_id="v", sample_idx=0,
                                     output={"reply": "No."}), Ctx(tmp_path))
    assert not score.passed and score.reason == "curt"
    [unit] = check.gating_units()
    assert unit.key == "polite_reply" and unit.prompt_hash == score.prompt_hash
