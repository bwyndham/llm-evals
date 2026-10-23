"""STAT-9: a regression in one gating check must not hide behind gains in another."""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_evals.engine import build_checks, run_comparison
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Decision, Result, Score, VersionManifest


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class TwoTraitRunner:
    """Output has two independent traits; versions set each trait's pass rate."""

    def run(self, case, version, sample_idx, run_id):
        out = {
            trait: _u(run_id, case.id, version.id, sample_idx, trait) < version.params[trait]
            for trait in ("a", "b")
        }
        return Result(run_id=run_id, case_id=case.id, version_id=version.id,
                      sample_idx=sample_idx, output=out)


class TraitCheck:
    kind = "deterministic"
    critical = False

    def __init__(self, trait: str) -> None:
        self.name = f"trait_{trait}"
        self.trait = trait

    def score(self, case, result, ctx) -> Score:
        return Score(check=self.name, kind=self.kind, passed=bool(result.output[self.trait]))


def _suite(tmp_path) -> SuiteConfig:
    return SuiteConfig(suite="t", runner={"type": "simulated"}, baseline="x.yaml", cases=[],
                       checks=[], samples=3, gate={"tolerance": 0.05}, dir=tmp_path)


def _cases(n=150):
    return [Case(id=f"c{i:03d}", input={}, provenance="constructed") for i in range(n)]


def _v(name, a, b):
    return VersionManifest(name=name, model="sim", prompt_templates={}, id=name,
                           params={"a": a, "b": b})


def test_masked_regression_rolls_back(tmp_path):
    # Trait a gets much worse, trait b gets much better: the AND pass rate rises,
    # but trait a's own paired test catches the regression.
    baseline, candidate = _v("base", a=0.95, b=0.40), _v("cand", a=0.70, b=0.95)
    report = run_comparison(
        _suite(tmp_path), baseline, candidate, store=Store(Path(":memory:")),
        runner=TwoTraitRunner(), checks=[TraitCheck("a"), TraitCheck("b")], cases=_cases(),
        n_boot=2000, run_id="t-masked",
    )
    assert report.comparison.delta > 0  # the combined metric alone would not catch it
    assert [r.check for r in report.check_regressions] == ["trait_a"]
    assert report.check_regressions[0].alpha == 0.0125  # 0.025 split over two checks
    assert report.decision == Decision.ROLLBACK


def test_no_per_check_regression_when_both_improve(tmp_path):
    report = run_comparison(
        _suite(tmp_path), _v("base", 0.8, 0.8), _v("cand", 0.95, 0.95),
        store=Store(Path(":memory:")), runner=TwoTraitRunner(),
        checks=[TraitCheck("a"), TraitCheck("b")], cases=_cases(), n_boot=2000, run_id="t-both",
    )
    assert report.check_regressions == []
    assert report.decision == Decision.PROMOTE


def test_report_only_check_never_gates(tmp_path):
    report_only = TraitCheck("a")
    report_only.report_only = True
    report = run_comparison(
        _suite(tmp_path), _v("base", 0.95, 0.9), _v("cand", 0.2, 0.9),
        store=Store(Path(":memory:")), runner=TwoTraitRunner(),
        checks=[report_only, TraitCheck("b")], cases=_cases(), n_boot=2000, run_id="t-ro",
    )
    assert report.candidate.check_pass_rates["trait_a"] < 0.3  # still scored and reported
    assert report.check_regressions == []
    assert "trait_a" in report.report_only
    assert report.decision != Decision.ROLLBACK


def test_gate_false_in_yaml_marks_check_report_only(tmp_path):
    suite = SuiteConfig(suite="t", runner={"type": "simulated"}, baseline="x.yaml", cases=[],
                        checks=[{"format_rules": {"gate": False}}, "gherkin_parses"], dir=tmp_path)
    checks = {c.name: c for c in build_checks(suite)}
    assert getattr(checks["format_rules"], "report_only", False) is True
    assert not getattr(checks["gherkin_parses"], "report_only", False)
