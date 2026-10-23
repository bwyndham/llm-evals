"""Offline demo mode end to end: the real spec-writing suite, the engine and the mock model.

No API key and no network. `MockProvider` stands in for the model, so these
tests prove the plumbing (render -> run -> parse -> check -> compare -> gate)
and that the gate catches the seeded regressions. They say nothing about how a
real model would score.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals import registry
from llm_evals.cases.loader import load_cases
from llm_evals.engine import ComparisonReport, run_comparison
from llm_evals.runners.manifest import load_manifest
from llm_evals.runners.prompt import PromptRunner
from llm_evals.runners.providers.mock import MockProvider
from llm_evals.store.sqlite import Store
from llm_evals.suite import load_suite
from llm_evals.types import Decision

SUITE_DIR = Path(__file__).resolve().parents[2] / "suites" / "spec-writing"
N_CASES = 40
K = 3


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def cases():
    return load_cases(SUITE_DIR / "data" / "cases.jsonl")[:N_CASES]


def _checks():
    return [
        registry.get("check", "gherkin_parses")(critical=True),
        registry.get("check", "format_rules")(),
        registry.get("check", "atom_keywords")(),
        registry.get("check", "invariance")(relation="reword_ask", seed=11),
    ]


def _version(name: str):
    return load_manifest(SUITE_DIR / "versions" / f"{name}.yaml", SUITE_DIR)


def _compare(suite, cases, baseline: str, candidate: str) -> ComparisonReport:
    runner = PromptRunner(provider=MockProvider(), output_schema="spec_ticket")
    return run_comparison(
        suite, _version(baseline), _version(candidate), store=Store(Path(":memory:")), runner=runner,
        checks=_checks(), cases=cases, samples=K, n_boot=2000, seed=0,
        max_concurrency=1,  # one call order per (system, user) pair, so the run is reproducible
    )


@pytest.fixture(scope="module")
def no_split(suite, cases):
    return _compare(suite, cases, "baseline", "seeded_no_split_bundles")


@pytest.fixture(scope="module")
def no_flags(suite, cases):
    return _compare(suite, cases, "baseline", "seeded_no_assumption_flags")


@pytest.fixture(scope="module")
def aa(suite, cases):
    return _compare(suite, cases, "baseline", "baseline")


def test_the_case_slice_covers_both_seeded_failure_modes(cases):
    assert len(cases) == N_CASES
    assert 10 <= sum(c.strata["bundling"] == "bundled" for c in cases) <= 30
    assert 10 <= sum(c.strata["missing_actor"] == "yes" for c in cases) <= 30


def test_baseline_mock_passes_the_deterministic_checks(no_split, no_flags, aa):
    for report in (no_split, no_flags, aa):
        rates = report.baseline.check_pass_rates
        assert rates["gherkin_parses"] >= 0.95, rates
        assert rates["format_rules"] >= 0.95, rates
        assert rates["atom_keywords"] >= 0.80, rates
        assert report.baseline.system_errors == 0 and report.baseline.infra_errors == 0
        assert report.baseline.cost_usd > 0  # token counts flow into cost


def test_no_split_bundles_rolls_back(no_split, cases):
    assert no_split.decision == Decision.ROLLBACK, no_split.comparison
    assert no_split.comparison.ci_high < 0
    assert no_split.candidate.check_pass_rates["atom_keywords"] < 0.75
    broke = [f for f in no_split.flips if f.direction == "broke"]
    assert broke and all(f.strata["bundling"] == "bundled" for f in broke)
    assert all("atom_keywords" in f.failed_checks for f in broke)


def test_no_assumption_flags_rolls_back(no_flags):
    assert no_flags.decision == Decision.ROLLBACK, no_flags.comparison
    assert no_flags.comparison.ci_high < 0
    assert no_flags.candidate.check_pass_rates["atom_keywords"] < 0.5
    broke = [f for f in no_flags.flips if f.direction == "broke"]
    assert broke and all("atom_keywords" in f.failed_checks for f in broke)
    # The invented scenarios are named in the stored outputs of the broken cases.
    assert any("Restrict this area" in (f.candidate_output or {}).get("acceptance_criteria", "")
               or "fresh sign-in" in (f.candidate_output or {}).get("acceptance_criteria", "") for f in broke)


def test_aa_does_not_roll_back(aa):
    assert aa.decision != Decision.ROLLBACK, aa.comparison
    assert aa.critical_new_failures == []
    assert aa.baseline.version_id != aa.candidate.version_id
    assert aa.comparison.ci_low <= 0 <= aa.comparison.ci_high


def test_aa_samples_vary(aa):
    # Sampling noise drops a requirement now and then, so not every sample is identical.
    assert aa.baseline.check_pass_rates["atom_keywords"] < 1.0 or aa.candidate.check_pass_rates["atom_keywords"] < 1.0
