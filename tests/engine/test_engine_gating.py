"""A judge unit gates only once it has an approved calibration (CAL-2, CAL-4)."""

from __future__ import annotations

from pathlib import Path

from llm_evals.engine import gating_pass, run_comparison
from llm_evals.runners.simulated import SimulatedRunner
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Decision, GatingUnit, Score, VersionManifest


class AlwaysFailsJudge:
    """A judge check whose single unit always fails."""

    name = "fake_judge"
    kind = "judge"
    critical = False
    needs_judge = False  # nothing to bind in this test

    def gating_units(self) -> list[GatingUnit]:
        return [GatingUnit(key="fake:unit", judge_model="judge-x", prompt_hash="abc123")]

    def score(self, case, result, ctx) -> Score:
        return Score(check=self.name, kind="judge", passed=False,
                     sub_scores=[{"unit": "fake:unit", "passed": False}])


class PassedCheck:
    name = "simulated_pass"
    kind = "deterministic"
    critical = False

    def score(self, case, result, ctx) -> Score:
        return Score(check=self.name, kind=self.kind,
                     passed=bool(result.output and result.output.get("passed")))


def _setup(tmp_path):
    suite = SuiteConfig(suite="sim", runner={"type": "simulated"}, baseline="x.yaml",
                        cases=[], checks=[], samples=2, gate={"tolerance": 0.2}, dir=tmp_path)
    cases = [Case(id=f"c{i}", input={}, expected={"p_fail": 0.0}, provenance="constructed")
             for i in range(60)]
    version = VersionManifest(name="v", model="sim", prompt_templates={}, id="v")
    return suite, cases, version


def test_uncalibrated_judge_is_report_only(tmp_path):
    suite, cases, version = _setup(tmp_path)
    report = run_comparison(suite, version, version, store=Store(Path(":memory:")),
                            runner=SimulatedRunner(), checks=[PassedCheck(), AlwaysFailsJudge()],
                            cases=cases, n_boot=500, run_id='t-gating')
    # The judge fails everything, but it isn't calibrated, so cases still pass.
    assert report.baseline.pass_rate == 1.0
    assert report.candidate.check_pass_rates["fake_judge"] == 0.0
    assert report.report_only == ["fake:unit"]
    assert report.gating["fake_judge"] == []
    assert report.decision == Decision.PROMOTE


def test_calibrated_judge_unit_gates(tmp_path, monkeypatch):
    suite, cases, version = _setup(tmp_path)
    store = Store(Path(":memory:"))
    monkeypatch.setattr(store, "approved_calibration",
                        lambda key, model, h, **kw: (key, model, h) == ("fake:unit", "judge-x", "abc123"),
                        raising=False)
    report = run_comparison(suite, version, version, store=store, runner=SimulatedRunner(),
                            checks=[PassedCheck(), AlwaysFailsJudge()], cases=cases, n_boot=500, run_id='t-gating')
    assert report.baseline.pass_rate == 0.0  # the calibrated unit now counts
    assert report.gating["fake_judge"] == ["fake:unit"]
    assert report.report_only == []


def test_gating_pass_rules():
    failing = Score(check="j", kind="judge", passed=False,
                    sub_scores=[{"unit": "a", "passed": True}, {"unit": "b", "passed": False}])
    assert gating_pass(failing, None) is False  # deterministic checks: their verdict
    assert gating_pass(failing, set()) is None  # nothing calibrated: report-only
    assert gating_pass(failing, {"a"}) is True  # only calibrated units count
    assert gating_pass(failing, {"a", "b"}) is False
    judge_error = Score(check="j", kind="judge", passed=False, reason="judge error: bad JSON")
    assert gating_pass(judge_error, {"a"}) is False
