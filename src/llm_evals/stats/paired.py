"""Paired baseline-vs-candidate comparison over shared cases.

Δ is the mean over cases of (candidate rate − baseline rate). Its CI comes
from a cluster bootstrap: cases are resampled with replacement and all k
samples of a case move together, so within-case correlation is respected.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field
from scipy.stats import binomtest, norm

Outcomes = dict[str, list[bool]]  # case_id -> pass/fail per sample, infra errors removed

UNDERPOWERED_N = 20  # fewer shared cases than this and a comparison is flagged


class Flip(BaseModel):
    """A case whose majority outcome changed between baseline and candidate."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    direction: Literal["broke", "fixed"]
    baseline_rate: float
    candidate_rate: float


class PairedComparison(BaseModel):
    """Result of `compare`. JSON-serializable via `model_dump_json()`."""

    model_config = ConfigDict(frozen=True)

    delta: float  # candidate − baseline, mean of per-case rate differences
    ci_low: float
    ci_high: float
    n_cases: int
    k_baseline: int  # max samples per case seen on the baseline side
    k_candidate: int
    baseline_rate: float  # mean of per-case pass rates
    candidate_rate: float
    mde: float | None  # minimum detectable |Δ| at the requested power; None if n_cases < 2
    mcnemar_p: float
    flips: list[Flip]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def underpowered(self) -> bool:
        return self.n_cases < UNDERPOWERED_N


def _empty() -> PairedComparison:
    return PairedComparison(
        delta=0.0,
        ci_low=0.0,
        ci_high=0.0,
        n_cases=0,
        k_baseline=0,
        k_candidate=0,
        baseline_rate=0.0,
        candidate_rate=0.0,
        mde=None,
        mcnemar_p=1.0,
        flips=[],
    )


def _majority(rate: float) -> bool:
    # A tie at exactly 0.5 counts as a fail.
    return rate > 0.5


def compare(
    baseline: Outcomes,
    candidate: Outcomes,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PairedComparison:
    """Paired comparison of candidate against baseline on the cases both ran.

    Only case ids present in both, with at least one sample on each side, are
    used. The CI is a percentile interval from a cluster bootstrap over cases.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")

    case_ids = sorted(
        cid for cid in baseline.keys() & candidate.keys() if baseline[cid] and candidate[cid]
    )
    n = len(case_ids)
    if n == 0:
        return _empty()

    # Plain sum/len: np.mean on short Python lists costs more than the bootstrap.
    rate_a = np.array([sum(baseline[c]) / len(baseline[c]) for c in case_ids], dtype=float)
    rate_b = np.array([sum(candidate[c]) / len(candidate[c]) for c in case_ids], dtype=float)
    diffs = rate_b - rate_a
    delta = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32 if n < 2**31 else np.int64)
    boot = diffs[idx].mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]))

    mde: float | None = None
    if n >= 2:
        se = float(diffs.std(ddof=1)) / np.sqrt(n)
        mde = float((norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)) * se)

    flips: list[Flip] = []
    for cid, ra, rb in zip(case_ids, rate_a, rate_b):
        pa, pb = _majority(ra), _majority(rb)
        if pa != pb:
            flips.append(
                Flip(
                    case_id=cid,
                    direction="broke" if pa else "fixed",
                    baseline_rate=float(ra),
                    candidate_rate=float(rb),
                )
            )
    flips.sort(key=lambda f: (f.direction != "broke", f.case_id))
    n_broke = sum(f.direction == "broke" for f in flips)
    n_discordant = len(flips)
    mcnemar_p = 1.0 if n_discordant == 0 else float(binomtest(n_broke, n_discordant, 0.5).pvalue)

    return PairedComparison(
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n_cases=n,
        k_baseline=max(len(baseline[c]) for c in case_ids),
        k_candidate=max(len(candidate[c]) for c in case_ids),
        baseline_rate=float(rate_a.mean()),
        candidate_rate=float(rate_b.mean()),
        mde=mde,
        mcnemar_p=mcnemar_p,
        flips=flips,
    )


def compare_by_stratum(
    baseline: Outcomes,
    candidate: Outcomes,
    strata: dict[str, dict[str, str]],
    **kw: Any,
) -> dict[str, dict[str, PairedComparison]]:
    """Compare within each stratum value: field -> value -> PairedComparison.

    `strata` maps case_id -> {field: value} (i.e. `Case.strata`). A case with
    no value for a field is left out of that field's slices. Keyword arguments
    go to `compare`. Check `.underpowered` on each slice before reading it.
    """
    shared = baseline.keys() & candidate.keys()
    groups: dict[str, dict[str, set[str]]] = {}
    for cid in shared:
        for field, value in strata.get(cid, {}).items():
            groups.setdefault(field, {}).setdefault(value, set()).add(cid)

    out: dict[str, dict[str, PairedComparison]] = {}
    for field in sorted(groups):
        out[field] = {}
        for value in sorted(groups[field]):
            ids = groups[field][value]
            out[field][value] = compare(
                {c: baseline[c] for c in ids}, {c: candidate[c] for c in ids}, **kw
            )
    return out
