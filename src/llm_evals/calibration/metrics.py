"""Human-vs-judge agreement for one gating unit (CAL-2).

Convention: the POSITIVE class is FAIL. A judge's job is to catch failures, so
- sensitivity = P(judge fails | human fails) = TP / (TP + FN)
- specificity = P(judge passes | human passes) = TN / (TN + FP)
- prevalence  = P(human fails)               = (TP + FN) / n
where TP = both fail, FN = human fail / judge pass, FP = human pass / judge fail,
TN = both pass. The human label is the reference.

Cohen's kappa: po = (TP + TN) / n, pe = p_h_fail * p_j_fail + p_h_pass * p_j_pass,
kappa = (po - pe) / (1 - pe).

Degenerate case: when either rater used only one class, kappa is reported as
None. If both used the same single class the formula is 0/0; if only one rater
is constant, kappa is pinned at exactly 0 whatever the other does, so it says
nothing about judge quality. Either way a None kappa never approves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Label = bool | str  # True / "pass" = pass, False / "fail" = fail


def _passed(value: Label) -> bool:
    if isinstance(value, bool):
        return value
    if value == "pass":
        return True
    if value == "fail":
        return False
    raise ValueError(f"label must be 'pass', 'fail' or a bool, got {value!r}")


@dataclass(frozen=True)
class AgreementMetrics:
    tp: int  # human fail, judge fail
    fn: int  # human fail, judge pass
    fp: int  # human pass, judge fail
    tn: int  # human pass, judge pass
    agreement: float | None
    kappa: float | None
    sensitivity: float | None
    specificity: float | None
    prevalence: float | None

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def from_confusion(*, tp: int, fn: int, fp: int, tn: int) -> AgreementMetrics:
    """Metrics from a confusion matrix with positive = fail (see module docstring)."""
    if min(tp, fn, fp, tn) < 0:
        raise ValueError("confusion counts must be non-negative")
    n = tp + fn + fp + tn
    agreement = _ratio(tp + tn, n)
    kappa: float | None = None
    if n:
        h_fail, j_fail = (tp + fn) / n, (tp + fp) / n
        degenerate = h_fail in (0.0, 1.0) or j_fail in (0.0, 1.0)
        if not degenerate:
            pe = h_fail * j_fail + (1 - h_fail) * (1 - j_fail)
            kappa = (agreement - pe) / (1 - pe)  # type: ignore[operator]
    return AgreementMetrics(
        tp=tp, fn=fn, fp=fp, tn=tn,
        agreement=agreement,
        kappa=kappa,
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        prevalence=_ratio(tp + fn, n),
    )


def agreement_metrics(pairs: Iterable[tuple[Label, Label]]) -> AgreementMetrics:
    """Metrics from (human, judge) label pairs; each label is a bool (True = pass)
    or "pass" / "fail"."""
    tp = fn = fp = tn = 0
    for human, judge in pairs:
        h_pass, j_pass = _passed(human), _passed(judge)
        if not h_pass and not j_pass:
            tp += 1
        elif not h_pass:
            fn += 1
        elif not j_pass:
            fp += 1
        else:
            tn += 1
    return from_confusion(tp=tp, fn=fn, fp=fp, tn=tn)
