import time

import numpy as np
import pytest

from llm_evals.stats import Flip, PairedComparison, compare, compare_by_stratum, decide
from llm_evals.types import Decision

T, F = True, False


def test_delta_on_hand_built_data():
    baseline = {"a": [T, T, F], "b": [T, T], "c": [F], "only_base": [T]}
    candidate = {"a": [T, F, F], "b": [T, T, T], "c": [T], "only_cand": [F]}
    cmp = compare(baseline, candidate, n_boot=500)
    # per-case diffs: a 1/3 - 2/3, b 1 - 1, c 1 - 0
    assert cmp.n_cases == 3
    assert cmp.delta == pytest.approx((-1 / 3 + 0 + 1) / 3)
    assert cmp.baseline_rate == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert cmp.candidate_rate == pytest.approx((1 / 3 + 1 + 1) / 3)
    assert cmp.k_baseline == 3 and cmp.k_candidate == 3
    assert cmp.ci_low <= cmp.delta <= cmp.ci_high


def test_flips_direction_and_order():
    baseline = {"z": [T, T, F], "a": [F, F, T], "m": [T, T, T], "b": [T, T]}
    candidate = {"z": [F, F, T], "a": [T, T, T], "m": [T, T, T], "b": [F, F]}
    cmp = compare(baseline, candidate, n_boot=200)
    assert [(f.case_id, f.direction) for f in cmp.flips] == [
        ("b", "broke"),
        ("z", "broke"),
        ("a", "fixed"),
    ]
    assert cmp.flips[1] == Flip(
        case_id="z", direction="broke", baseline_rate=2 / 3, candidate_rate=1 / 3
    )


def test_tie_at_half_counts_as_fail():
    cmp = compare({"a": [T, F]}, {"a": [T, T]}, n_boot=100)
    assert [(f.case_id, f.direction) for f in cmp.flips] == [("a", "fixed")]
    cmp = compare({"a": [T, T]}, {"a": [T, F]}, n_boot=100)
    assert [f.direction for f in cmp.flips] == ["broke"]


def test_mcnemar_trivial_cases():
    same = {f"c{i}": [T, T, F] for i in range(10)}
    assert compare(same, same, n_boot=100).mcnemar_p == 1.0
    base = {f"c{i}": [T] for i in range(10)}
    cand = {f"c{i}": [F] for i in range(10)}
    assert compare(base, cand, n_boot=100).mcnemar_p == pytest.approx(2 * 0.5**10)
    # one broke, one fixed: perfectly balanced
    assert compare({"a": [T], "b": [F]}, {"a": [F], "b": [T]}, n_boot=100).mcnemar_p == 1.0


def test_cases_with_no_samples_are_skipped():
    cmp = compare({"a": [T], "b": [], "c": [T]}, {"a": [F], "b": [T], "c": []}, n_boot=100)
    assert cmp.n_cases == 1
    assert cmp.delta == -1.0


def test_empty_overlap():
    cmp = compare({"a": [T]}, {"b": [T]})
    assert (cmp.delta, cmp.ci_low, cmp.ci_high, cmp.n_cases) == (0.0, 0.0, 0.0, 0)
    assert cmp.mde is None and cmp.flips == [] and cmp.mcnemar_p == 1.0
    assert decide(cmp, tolerance=0.02, critical_new_failures=[], incomplete=False) is (
        Decision.INCOMPLETE
    )


def test_constant_differences_give_degenerate_ci():
    base = {f"c{i}": [T, T] for i in range(30)}
    cand = {f"c{i}": [T, F] for i in range(30)}
    cmp = compare(base, cand, n_boot=500)
    assert cmp.delta == pytest.approx(-0.5)
    assert cmp.ci_low == pytest.approx(-0.5) and cmp.ci_high == pytest.approx(-0.5)
    assert cmp.mde == pytest.approx(0.0)


def _random_outcomes(n_cases: int, k: int, p: float, seed: int) -> dict[str, list[bool]]:
    rng = np.random.default_rng(seed)
    draws = rng.random((n_cases, k)) < p
    return {f"c{i}": [bool(x) for x in row] for i, row in enumerate(draws)}


def test_bootstrap_seeded_and_sane():
    base = _random_outcomes(200, 3, 0.9, seed=1)
    cand = _random_outcomes(200, 3, 0.85, seed=2)
    a = compare(base, cand, n_boot=2000, seed=7)
    b = compare(base, cand, n_boot=2000, seed=7)
    c = compare(base, cand, n_boot=2000, seed=8)
    assert a == b
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)
    assert a.ci_low < a.delta < a.ci_high
    # The bootstrap CI half-width should be close to the normal one.
    diffs = np.array([np.mean(cand[k]) - np.mean(base[k]) for k in base])
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert (a.ci_high - a.ci_low) / 2 == pytest.approx(1.96 * se, rel=0.1)


def test_mde_formula():
    base = _random_outcomes(120, 3, 0.9, seed=3)
    cand = _random_outcomes(120, 3, 0.9, seed=4)
    cmp = compare(base, cand, n_boot=200, power=0.9, alpha=0.1)
    diffs = np.array([np.mean(cand[k]) - np.mean(base[k]) for k in base])
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert cmp.mde == pytest.approx((1.6448536269514722 + 1.2815515655446004) * se)
    assert compare({"a": [T]}, {"a": [F]}, n_boot=10).mde is None


def test_json_round_trip():
    cmp = compare({"a": [T, T], "b": [F]}, {"a": [F, F], "b": [F]}, n_boot=100)
    data = cmp.model_dump()
    assert data["underpowered"] is True
    assert data["flips"][0]["direction"] == "broke"
    assert PairedComparison.model_validate_json(cmp.model_dump_json()) == cmp


def test_bootstrap_is_fast():
    base = _random_outcomes(500, 5, 0.9, seed=5)
    cand = _random_outcomes(500, 5, 0.88, seed=6)
    compare(base, cand, n_boot=100)  # warm up imports
    start = time.perf_counter()
    compare(base, cand, n_boot=10_000)
    assert time.perf_counter() - start < 1.0


def test_compare_by_stratum():
    base, cand, strata = {}, {}, {}
    for i in range(40):
        cid = f"c{i}"
        base[cid] = [T, T, T]
        lang = "en" if i < 30 else "fr"
        cand[cid] = [T, T, T] if lang == "en" else [F, F, F]
        strata[cid] = {"lang": lang, "len": "short" if i % 2 else "long"}
    strata.pop("c0")  # no strata: left out of every slice
    base["extra"] = [T]
    strata["extra"] = {"lang": "de"}  # not in candidate: no slice at all

    out = compare_by_stratum(base, cand, strata, n_boot=300, seed=1)
    assert set(out) == {"lang", "len"}
    assert set(out["lang"]) == {"en", "fr"}
    assert out["lang"]["en"].n_cases == 29 and out["lang"]["en"].delta == 0.0
    assert out["lang"]["fr"].n_cases == 10 and out["lang"]["fr"].delta == -1.0
    assert out["lang"]["fr"].underpowered and not out["lang"]["en"].underpowered
    assert out["len"]["long"].n_cases + out["len"]["short"].n_cases == 39
