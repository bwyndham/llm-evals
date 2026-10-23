import math

import pytest

from llm_evals.stats import wilson

Z95 = 1.959963984540054


def closed_form(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def test_wilson_129_of_133():
    low, high = wilson(129, 133)
    exp_low, exp_high = closed_form(129, 133)
    assert low == pytest.approx(exp_low, abs=1e-12)
    assert high == pytest.approx(exp_high, abs=1e-12)
    assert low == pytest.approx(0.925, abs=5e-4)
    assert high == pytest.approx(0.988, abs=5e-4)


@pytest.mark.parametrize("x,n", [(1, 2), (5, 10), (37, 50), (500, 1000), (3, 7)])
def test_wilson_matches_closed_form(x, n):
    assert wilson(x, n) == pytest.approx(closed_form(x, n), abs=1e-12)


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-4)


def test_wilson_symmetry_and_alpha():
    low, high = wilson(3, 20)
    low2, high2 = wilson(17, 20)
    assert low == pytest.approx(1 - high2) and high == pytest.approx(1 - low2)
    l90, h90 = wilson(3, 20, alpha=0.10)
    assert low < l90 and h90 < high


@pytest.mark.parametrize("x,n", [(-1, 5), (6, 5), (0, -1)])
def test_wilson_rejects_bad_counts(x, n):
    with pytest.raises(ValueError):
        wilson(x, n)
