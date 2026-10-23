from __future__ import annotations

import pytest

from llm_evals.calibration.metrics import agreement_metrics, from_confusion


def test_hand_computed_confusion_matrix() -> None:
    # TP=20 FN=5 FP=10 TN=65, n=100. po=.85; h_fail=.25, j_fail=.30;
    # pe = .25*.30 + .75*.70 = .60; kappa = (.85-.60)/(1-.60) = .625
    m = from_confusion(tp=20, fn=5, fp=10, tn=65)
    assert m.n == 100
    assert m.agreement == pytest.approx(0.85)
    assert m.kappa == pytest.approx(0.625)
    assert m.sensitivity == pytest.approx(20 / 25)
    assert m.specificity == pytest.approx(65 / 75)
    assert m.prevalence == pytest.approx(0.25)


def test_from_pairs_positive_is_fail() -> None:
    # (human, judge): 2 both-fail, 1 human-fail/judge-pass, 1 human-pass/judge-fail, 6 both-pass
    pairs = ([("fail", "fail")] * 2 + [("fail", "pass")] + [("pass", "fail")]
             + [("pass", "pass")] * 6)
    m = agreement_metrics(pairs)
    assert (m.tp, m.fn, m.fp, m.tn) == (2, 1, 1, 6)
    # po=.8, pe=.3*.3+.7*.7=.58, kappa=.22/.42=11/21
    assert m.agreement == pytest.approx(0.8)
    assert m.kappa == pytest.approx(11 / 21)
    assert m.sensitivity == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(6 / 7)
    assert m.prevalence == pytest.approx(0.3)


def test_bool_labels_true_is_pass() -> None:
    m = agreement_metrics([(False, False), (True, True), (False, True)])
    assert (m.tp, m.tn, m.fn, m.fp) == (1, 1, 1, 0)


def test_perfect_and_inverse_agreement() -> None:
    assert from_confusion(tp=5, fn=0, fp=0, tn=5).kappa == pytest.approx(1.0)
    inverse = from_confusion(tp=0, fn=5, fp=5, tn=0)
    assert inverse.agreement == 0.0
    assert inverse.kappa == pytest.approx(-1.0)  # po=0, pe=.5


def test_chance_level_kappa_is_zero() -> None:
    # Judge independent of human: h_fail=.5, j_fail=.5, po=.5=pe
    assert from_confusion(tp=5, fn=5, fp=5, tn=5).kappa == pytest.approx(0.0)


@pytest.mark.parametrize("counts", [
    dict(tp=0, fn=0, fp=0, tn=10),   # everything passes for both raters: 0/0
    dict(tp=10, fn=0, fp=0, tn=0),   # everything fails for both
    dict(tp=0, fn=0, fp=3, tn=7),    # human all pass, judge mixed: kappa pinned at 0
    dict(tp=4, fn=6, fp=0, tn=0),    # human all fail
    dict(tp=0, fn=3, fp=0, tn=7),    # judge all pass, human mixed
])
def test_degenerate_marginals_give_no_kappa(counts: dict[str, int]) -> None:
    assert from_confusion(**counts).kappa is None


def test_undefined_rates_are_none() -> None:
    m = from_confusion(tp=0, fn=0, fp=0, tn=10)
    assert m.sensitivity is None  # no human failures to catch
    assert m.specificity == 1.0
    assert m.prevalence == 0.0
    empty = agreement_metrics([])
    assert empty.n == 0
    assert (empty.agreement, empty.kappa, empty.sensitivity, empty.specificity,
            empty.prevalence) == (None, None, None, None, None)


def test_rejects_bad_labels() -> None:
    with pytest.raises(ValueError):
        agreement_metrics([("ok", "pass")])
    with pytest.raises(ValueError):
        from_confusion(tp=-1, fn=0, fp=0, tn=0)
