"""Confidence intervals for single proportions."""

from __future__ import annotations

import math

from scipy.stats import norm


def wilson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (low, high), clipped to [0, 1]. With n == 0 there is no
    information, so the interval is the whole range (0.0, 1.0).
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n == 0:
        return (0.0, 1.0)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    p = successes / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    # At 0 or n successes the bound is exactly 0 or 1; set it so rounding can't miss.
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return (low, high)
