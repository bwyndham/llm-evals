"""QA-1 acceptance: the statistics behave as advertised on simulated data.

Every case has a known failure probability (Beta(1, 9), mean 0.1, fixed seed)
and the SimulatedRunner draws pass/fail from it. Each simulated comparison
uses fresh run_ids, so repeated runs of the same version are independent the
way repeated calls to a real model are. All seeds are fixed, so these tests
are deterministic; the thresholds were chosen so that they would also pass for
almost any other seed (see the notes on each test).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_evals.runners.simulated import SimulatedRunner, simulate_outcomes, true_p_fail
from llm_evals.stats import PairedComparison, compare, decide
from llm_evals.types import Case, Decision, VersionManifest

N_CASES = 150
K = 3
N_BOOT = 2000
RUNNER = SimulatedRunner(seed=0)


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    rng = np.random.default_rng(20261019)
    return [
        Case(id=f"c{i:03d}", input={}, expected={"p_fail": float(p)}, provenance="constructed")
        for i, p in enumerate(rng.beta(1, 9, N_CASES))
    ]


def version(name: str, shift: float = 0.0) -> VersionManifest:
    return VersionManifest(name=name, model="m", prompt_templates={}, params={"shift": shift})


BASE = version("base")


def true_delta(cases: list[Case], candidate: VersionManifest) -> float:
    """Mean over cases of the change in pass probability, after clamping."""
    return -float(np.mean([true_p_fail(c, candidate) - true_p_fail(c, BASE) for c in cases]))


def simulate(cases, candidate, sim: int, tag: str) -> PairedComparison:
    base = simulate_outcomes(cases, BASE, K, RUNNER, run_id=f"{tag}-{sim}-base")
    cand = simulate_outcomes(cases, candidate, K, RUNNER, run_id=f"{tag}-{sim}-cand")
    return compare(base, cand, n_boot=N_BOOT, seed=sim)


def test_ci_coverage(cases):
    # 1000 sims: the SE of the coverage estimate is ~0.7pp, so 93-97% is about
    # +/-3 SE around 95%. Observed with these seeds: 94.6%.
    candidate = version("cand", shift=0.03)
    truth = true_delta(cases, candidate)
    assert truth == pytest.approx(-0.03, abs=1e-3)
    n_sims = 1000
    covered = sum(
        c.ci_low <= truth <= c.ci_high
        for c in (simulate(cases, candidate, s, "cov") for s in range(n_sims))
    )
    assert 0.93 <= covered / n_sims <= 0.97


def test_aa_rollback_rate(cases):
    # Nominal one-sided rate is 2.5%; the percentile bootstrap runs slightly
    # hot here (2.75% over 4000 sims in development). 1500 sims gives an SE of
    # ~0.4pp, so the 3.5% bound sits ~1.8 SE above the true rate; with fewer
    # sims (400-1000) the bound fails for a noticeable share of seeds.
    # Observed with these seeds: 3.0%.
    n_sims = 1500
    rollbacks = sum(
        decide(
            simulate(cases, BASE, s, "aa"),
            tolerance=0.02,
            critical_new_failures=[],
            incomplete=False,
        )
        is Decision.ROLLBACK
        for s in range(n_sims)
    )
    rate = rollbacks / n_sims
    assert rate <= 0.035, f"A/A rollback rate {rate:.3%}"


def test_mde_matches_simulated_power(cases):
    """The reported MDE is the |Δ| at which ~80% of runs have ci_high < 0.

    The MDE uses the observed SE, which grows with the size of the regression,
    so first find the fixed point s* where a regression of s* reports MDE = s*.
    Then measure the detection rate (ci_high < 0) at 0.85 s*, s*, 1.15 s*,
    interpolate the |Δ| where it crosses 80%, and require it within 1pp of s*.
    With 500 sims per point the detection rate has an SE of ~1.8pp, which moves
    the interpolated crossing by ~0.1pp of Δ, well inside the 1pp bound.

    Observed with these seeds: MDE 6.1pp, detection 76% at Δ = -MDE, 80% crossing
    at 6.4pp. Power runs a little under nominal because the MDE plugs in an
    estimated SE and the percentile CI is slightly wider than the normal one.
    """
    shift = 0.05
    for it in range(2):  # converges after one step; the second confirms it
        mdes = [simulate(cases, version("cand", shift), s, f"fix{it}").mde for s in range(100)]
        shift = float(np.mean(mdes))
    mde = shift

    n_sims = 500
    deltas, detect = [], []
    for factor in (0.85, 1.0, 1.15):
        candidate = version("cand", mde * factor)
        hits = sum(
            simulate(cases, candidate, s, f"mde{factor}").ci_high < 0 for s in range(n_sims)
        )
        deltas.append(-true_delta(cases, candidate))
        detect.append(hits / n_sims)

    assert detect[0] < detect[1] < detect[2], detect
    assert 0.70 <= detect[1] <= 0.90, detect
    crossing = float(np.interp(0.80, detect, deltas))
    assert abs(crossing - mde) <= 0.01, (crossing, mde, detect)
