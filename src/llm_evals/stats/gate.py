"""Gate decision from a paired comparison (PRD, Statistics and gate)."""

from __future__ import annotations

from llm_evals.stats.paired import PairedComparison
from llm_evals.types import Decision


def decide(
    cmp: PairedComparison,
    *,
    tolerance: float,
    critical_new_failures: list[str],
    incomplete: bool,
) -> Decision:
    """Apply the gate rule, in order.

    1. INCOMPLETE if the run is incomplete or no cases could be compared.
    2. ROLLBACK if the whole Δ CI is below zero, or any critical check newly fails.
    3. PROMOTE if the CI's lower bound is above −tolerance.
    4. HOLD otherwise.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    if incomplete or cmp.n_cases == 0:
        return Decision.INCOMPLETE
    if cmp.ci_high < 0 or critical_new_failures:
        return Decision.ROLLBACK
    if cmp.ci_low > -tolerance:
        return Decision.PROMOTE
    return Decision.HOLD
