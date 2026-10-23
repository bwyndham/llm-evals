"""End-to-end engine tests on the simulated runner: no model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals.engine import run_comparison
from llm_evals.qa import aa_test
from llm_evals.report.markdown import comment_marker, render_comment, render_markdown
from llm_evals.runners.simulated import SimulatedRunner
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Decision, Score, VersionManifest


class PassedCheck:
    """Reads the simulated runner's output["passed"]."""

    name = "simulated_pass"
    kind = "deterministic"

    def __init__(self, critical: bool = False) -> None:
        self.critical = critical

    def score(self, case, result, ctx) -> Score:
        ok = bool(result.output and result.output.get("passed"))
        return Score(check=self.name, kind=self.kind, passed=ok)


def _suite(tmp_path: Path, tolerance: float = 0.02) -> SuiteConfig:
    # Note: on 200-300 simulated cases with p_fail 2-30%, the A/A CI is about +/-4 points,
    # so a 2-point tolerance gives HOLD on a no-op change. That's the MDE warning working.
    return SuiteConfig(
        suite="sim",
        runner={"type": "simulated"},
        baseline="unused.yaml",
        cases=[],
        checks=[],
        samples=3,
        gate={"tolerance": tolerance},
        dir=tmp_path,
    )


def _cases(n: int = 300) -> list[Case]:
    # Per-case failure probabilities from 2% to 30%, deterministic.
    return [
        Case(id=f"c{i:04d}", input={}, expected={"p_fail": 0.02 + 0.28 * (i % 29) / 28},
             provenance="constructed", strata={"bucket": str(i % 3)})
        for i in range(n)
    ]


def _version(name: str, **params) -> VersionManifest:
    return VersionManifest(name=name, model="sim", prompt_templates={}, params=params, id=name)


@pytest.fixture
def store() -> Store:
    return Store(Path(":memory:"))


def _run(tmp_path, store, baseline, candidate, *, seed=0, cases=None, critical=False):
    return run_comparison(
        _suite(tmp_path), baseline, candidate, store=store,
        runner=SimulatedRunner(seed=seed), checks=[PassedCheck(critical=critical)],
        cases=cases or _cases(), n_boot=2000, seed=seed, max_concurrency=4,
        run_id=f"t-{baseline.name}-{candidate.name}-{seed}",
    )


def test_identical_versions_never_roll_back(tmp_path, store):
    report = _run(tmp_path, store, _version("base"), _version("base"))
    assert report.decision in (Decision.PROMOTE, Decision.HOLD)
    assert report.comparison.ci_low < 0 < report.comparison.ci_high
    assert report.n_cases == 300
    assert report.baseline.version_id != report.candidate.version_id  # A/A arms kept apart
    # With 300 cases the suite can't resolve a 2-point tolerance. A HOLD says so; a
    # PROMOTE was decided anyway and doesn't need the caveat.
    warned = any("could only reliably detect" in w for w in report.warnings)
    assert warned == (report.decision == Decision.HOLD)


def test_identical_versions_promote_with_a_tolerance_the_suite_can_resolve(tmp_path, store):
    report = run_comparison(
        _suite(tmp_path, tolerance=0.08), _version("base"), _version("base"), store=store,
        runner=SimulatedRunner(seed=0), checks=[PassedCheck()], cases=_cases(), n_boot=2000,
        run_id="t-resolve",
    )
    assert report.decision == Decision.PROMOTE


def test_clear_regression_rolls_back(tmp_path, store):
    report = _run(tmp_path, store, _version("base"), _version("worse", shift=0.15))
    assert report.decision == Decision.ROLLBACK
    assert report.comparison.ci_high < 0
    assert any(f.direction == "broke" for f in report.flips)


def test_infra_errors_make_the_run_incomplete(tmp_path, store):
    report = _run(tmp_path, store, _version("base"), _version("flaky", infra_error_rate=0.2))
    assert report.decision == Decision.INCOMPLETE
    assert report.exit_code == 3
    assert report.candidate.infra_share > 0.05


def test_critical_new_failure_rolls_back_despite_small_delta(tmp_path, store):
    cases = _cases(200)
    # One case flips from always passing to always failing.
    cases[0] = cases[0].model_copy(update={"expected": {"p_fail": 0.0}})
    baseline = _version("base")
    candidate = _version("cand")

    class OneCaseBreaks(SimulatedRunner):
        def run(self, case, version, sample_idx, run_id):
            result = super().run(case, version, sample_idx, run_id)
            if version.name == "cand" and case.id == cases[0].id:
                result = result.model_copy(update={"output": {"passed": False}})
            return result

    report = run_comparison(
        _suite(tmp_path), baseline, candidate, store=store, runner=OneCaseBreaks(seed=1),
        checks=[PassedCheck(critical=True)], cases=cases, n_boot=2000, run_id="t-critical",
    )
    assert f"{cases[0].id}:simulated_pass" in report.critical_new_failures
    assert report.decision == Decision.ROLLBACK


def test_reports_render(tmp_path, store):
    report = _run(tmp_path, store, _version("base"), _version("worse", shift=0.1))
    md = render_markdown(report)
    comment = render_comment(report)
    assert "ROLLBACK" in md and "Pass rate by check" in md
    assert comment.startswith(comment_marker(report.suite))
    assert "broke" in comment


def test_aa_rollback_rate_is_low(tmp_path, store):
    result = aa_test(
        _suite(tmp_path), _version("base"), runs=20, store=store,
        runner=SimulatedRunner(seed=0), checks=[PassedCheck()], cases=_cases(200),
        n_boot=1000, max_concurrency=4,
    )
    assert result.runs == 20
    assert sum(result.decisions.values()) == 20
    # Each A/A run draws fresh samples, so decisions vary across runs. The nominal
    # ROLLBACK rate is ~2.75% (see tests/stats/test_simulation.py); P(>= 4 of 20) < 0.5%.
    assert result.decisions.get("ROLLBACK", 0) <= 3
    assert len(result.decisions) > 1 or "PROMOTE" in result.decisions



def test_a_decided_run_carries_no_detectable_drop_warning(tmp_path, store):
    # A clear regression rolls back; its spread may exceed the tolerance, but that's
    # beside the point once the decision is made, so the report doesn't say it.
    report = run_comparison(
        _suite(tmp_path, tolerance=0.001), _version("base"), _version("worse", shift=0.1), store=store,
        runner=SimulatedRunner(seed=0), checks=[PassedCheck()], cases=_cases(), n_boot=2000,
        run_id="t-decided",
    )
    assert report.decision == Decision.ROLLBACK
    assert report.comparison.mde > 0.001
    assert not any("reliably detect" in w for w in report.warnings)


def test_a_hold_explains_what_the_detectable_drop_measures(tmp_path, store):
    for seed in range(20):
        report = run_comparison(
            _suite(tmp_path, tolerance=0.02), _version("base"), _version("base"), store=store,
            runner=SimulatedRunner(seed=seed), checks=[PassedCheck()], cases=_cases(), n_boot=2000,
            run_id=f"t-hold-{seed}",
        )
        if report.decision == Decision.HOLD:
            [w] = [w for w in report.warnings if "reliably detect" in w]
            assert "this run's own spread" in w and "llm-evals aa" in w
            return
    pytest.fail("no HOLD in 20 seeds; the test needs a noisier setup")
