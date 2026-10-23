"""Statistics and gate: paired Δ with a cluster bootstrap CI, Wilson, McNemar, MDE."""

from llm_evals.stats.gate import decide
from llm_evals.stats.intervals import wilson
from llm_evals.stats.paired import (
    UNDERPOWERED_N,
    Flip,
    Outcomes,
    PairedComparison,
    compare,
    compare_by_stratum,
)

__all__ = [
    "UNDERPOWERED_N",
    "Flip",
    "Outcomes",
    "PairedComparison",
    "compare",
    "compare_by_stratum",
    "decide",
    "wilson",
]
