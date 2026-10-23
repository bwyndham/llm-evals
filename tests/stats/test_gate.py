import pytest

from llm_evals.stats import PairedComparison, decide
from llm_evals.types import Decision


def cmp(ci_low: float, ci_high: float, n_cases: int = 100) -> PairedComparison:
    return PairedComparison(
        delta=(ci_low + ci_high) / 2,
        ci_low=ci_low,
        ci_high=ci_high,
        n_cases=n_cases,
        k_baseline=3,
        k_candidate=3,
        baseline_rate=0.9,
        candidate_rate=0.9,
        mde=0.05,
        mcnemar_p=1.0,
        flips=[],
    )


@pytest.mark.parametrize(
    "c,critical,incomplete,expected",
    [
        (cmp(-0.01, 0.03), [], True, Decision.INCOMPLETE),
        (cmp(-0.09, -0.05), ["c1"], True, Decision.INCOMPLETE),  # incomplete wins
        (cmp(-0.01, 0.03, n_cases=0), [], False, Decision.INCOMPLETE),
        (cmp(-0.09, -0.05), [], False, Decision.ROLLBACK),
        (cmp(0.01, 0.05), ["c1"], False, Decision.ROLLBACK),  # critical blocks a win
        (cmp(-0.01, 0.03), [], False, Decision.PROMOTE),
        (cmp(0.01, 0.05), [], False, Decision.PROMOTE),
        (cmp(-0.05, 0.01), [], False, Decision.HOLD),
        (cmp(-0.02, 0.01), [], False, Decision.HOLD),  # ci_low == -tolerance is not enough
        (cmp(-0.05, 0.0), [], False, Decision.HOLD),  # ci_high == 0 is not a rollback
    ],
)
def test_decide_branches(c, critical, incomplete, expected):
    got = decide(c, tolerance=0.02, critical_new_failures=critical, incomplete=incomplete)
    assert got is expected


def test_exit_codes():
    assert [d.exit_code for d in Decision] == [0, 1, 2, 3]


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        decide(cmp(0.0, 0.1), tolerance=-0.01, critical_new_failures=[], incomplete=False)
