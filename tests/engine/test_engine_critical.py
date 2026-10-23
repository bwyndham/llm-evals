"""Critical flips fire only when unanimous AND confirmed by reruns."""

from __future__ import annotations

from pathlib import Path

from llm_evals.engine import run_comparison
from llm_evals.runners.simulated import SimulatedRunner
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Decision, Score, VersionManifest


class PassedCheck:
    name = "simulated_pass"
    kind = "deterministic"
    critical = True

    def score(self, case, result, ctx) -> Score:
        return Score(check=self.name, kind=self.kind,
                     passed=bool(result.output and result.output.get("passed")))


def _suite(tmp_path, confirm_k=8) -> SuiteConfig:
    return SuiteConfig(suite="sim", runner={"type": "simulated"}, baseline="x.yaml", cases=[],
                       checks=[], samples=3, dir=tmp_path,
                       gate={"tolerance": 0.2, "critical_confirm_k": confirm_k})


def _cases(n=60, noisy=0.3):
    return [Case(id=f"c{i:03d}", input={}, provenance="constructed",
                 expected={"p_fail": noisy if i % 10 < 3 else 0.02}) for i in range(n)]


def _v(name):
    return VersionManifest(name=name, model="sim", prompt_templates={}, id=name)


class BreaksOne(SimulatedRunner):
    def run(self, case, version, sample_idx, run_id):
        result = super().run(case, version, sample_idx, run_id)
        if version.name == "cand" and case.id == "c005":
            result = result.model_copy(update={"output": {"passed": False}})
        return result


def test_real_critical_break_is_confirmed(tmp_path):
    report = run_comparison(_suite(tmp_path), _v("base"), _v("cand"), store=Store(Path(":memory:")),
                            runner=BreaksOne(seed=0), checks=[PassedCheck()], cases=_cases(),
                            n_boot=500, run_id="crit-real")
    assert "c005:simulated_pass" in report.critical_new_failures
    confirmation = next(c for c in report.critical_confirmations if c.case_id == "c005")
    assert confirmation.confirmed and confirmation.candidate_passes == 0
    assert confirmation.baseline_n == 8 and confirmation.candidate_n == 8
    assert report.decision == Decision.ROLLBACK


def test_noisy_unanimous_flips_are_cleared_by_reruns(tmp_path):
    # Search seeds for an A/A run where noise flips a case unanimously; the reruns clear it.
    for i in range(200):
        report = run_comparison(_suite(tmp_path), _v("base"), _v("base"),
                                store=Store(Path(":memory:")), runner=SimulatedRunner(seed=0),
                                checks=[PassedCheck()], cases=_cases(noisy=0.5), n_boot=300,
                                run_id=f"crit-noise-{i}")
        if report.critical_confirmations:
            break
    else:
        raise AssertionError("expected noise to flag at least one flip in 200 A/A runs")
    assert not any(c.confirmed for c in report.critical_confirmations)
    assert report.critical_new_failures == []


def test_confirmation_can_be_disabled(tmp_path):
    report = run_comparison(_suite(tmp_path, confirm_k=0), _v("base"), _v("cand"),
                            store=Store(Path(":memory:")), runner=BreaksOne(seed=0),
                            checks=[PassedCheck()], cases=_cases(), n_boot=300, run_id="crit-off")
    assert "c005:simulated_pass" in report.critical_new_failures
    assert report.critical_confirmations == []
