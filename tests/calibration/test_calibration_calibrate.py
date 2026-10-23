from __future__ import annotations

import pytest

from llm_evals.calibration import calibrate, unit_triples
from llm_evals.store.sqlite import Store

from calibration_helpers import HASH_A, HASH_B, JUDGE_MODEL, seed_run


def seed_pairs(store: Store, pairs: list[tuple[str, bool]], *, run_id: str = "r1",
               prompt_hash: str = HASH_A, labeler: str = "ben") -> None:
    """pairs: (human label, judge passed) for unit 'faithfulness'."""
    verdicts = {f"{run_id}-c{i:03d}": {"faithfulness": judge} for i, (_, judge) in
                enumerate(pairs)}
    seed_run(store, verdicts, check="faithfulness", run_id=run_id, per_unit_hash=False,
             prompt_hash=prompt_hash)
    for i, (human, _) in enumerate(pairs):
        store.save_label(run_id, "v1", f"{run_id}-c{i:03d}", 0, "faithfulness", human, labeler)


def test_metrics_joined_across_runs_and_saved(store: Store) -> None:
    # 2 both-fail, 1 FN, 1 FP, 6 both-pass split over two runs -> kappa 11/21
    seed_pairs(store, [("fail", False), ("fail", True), ("pass", False)] + [("pass", True)] * 2)
    seed_pairs(store, [("fail", False)] + [("pass", True)] * 4, run_id="r2")
    rec = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A, min_n=10, min_kappa=0.5)
    assert (rec.n, rec.tp, rec.fn, rec.fp, rec.tn) == (10, 2, 1, 1, 6)
    assert rec.kappa == pytest.approx(11 / 21)
    assert rec.sensitivity == pytest.approx(2 / 3)
    assert rec.approved
    assert rec.label_set == {"run_ids": ["r1", "r2"], "count": 10, "labelers": ["ben"],
                             "ties_dropped": 0}
    assert store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A, min_n=10,
                                      min_kappa=0.5)
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)  # default 50


def test_default_bars(store: Store) -> None:
    seed_pairs(store, [("fail", False)] * 25 + [("pass", True)] * 24)
    rec = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A)
    assert rec.n == 49 and rec.kappa == pytest.approx(1.0) and not rec.approved
    seed_pairs(store, [("pass", True)], run_id="r2")
    assert calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A).approved
    assert store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    assert len(store.calibration_history("faithfulness")) == 2


def test_only_verdicts_from_the_exact_prompt_hash_count(store: Store) -> None:
    seed_pairs(store, [("fail", False)] * 30 + [("pass", True)] * 30, prompt_hash=HASH_B)
    rec_a = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A)
    assert rec_a.n == 0 and rec_a.kappa is None and not rec_a.approved
    assert calibrate(store, "faithfulness", JUDGE_MODEL, HASH_B).approved
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    assert calibrate(store, "faithfulness", "claude-opus-5-5", HASH_B).n == 0


def test_degenerate_labels_never_approve(store: Store) -> None:
    seed_pairs(store, [("pass", True)] * 60)
    rec = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A)
    assert rec.n == 60 and rec.agreement == 1.0 and rec.kappa is None and not rec.approved


def test_multiple_labelers_majority_and_ties(store: Store) -> None:
    seed_pairs(store, [("fail", False), ("pass", True), ("pass", False)])
    store.save_label("r1", "v1", "r1-c000", 0, "faithfulness", "fail", "ana")  # 2-0 fail
    store.save_label("r1", "v1", "r1-c001", 0, "faithfulness", "fail", "ana")  # 1-1 tie
    store.save_label("r1", "v1", "r1-c002", 0, "faithfulness", "fail", "ana")
    store.save_label("r1", "v1", "r1-c002", 0, "faithfulness", "fail", "cy")   # 2-1 fail
    rec = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A, min_n=1)
    assert (rec.n, rec.tp, rec.tn) == (2, 2, 0)
    assert rec.label_set["ties_dropped"] == 1
    assert rec.label_set["labelers"] == ["ana", "ben", "cy"]
    only_ben = calibrate(store, "faithfulness", JUDGE_MODEL, HASH_A, labeler="ben", min_n=1)
    assert only_ben.n == 3 and only_ben.label_set["labelers"] == ["ben"]


def test_unit_triples_use_per_unit_hash_then_score_hash(store: Store) -> None:
    seed_run(store, {"c1": {"checklist:a": True, "checklist:b": False}}, prompt_hash=HASH_B)
    seed_run(store, {"c2": {"faithfulness": True}}, check="faithfulness", per_unit_hash=False)
    assert [(u.key, u.prompt_hash) for u in unit_triples(store, "checklist")] == [
        ("checklist:a", HASH_B), ("checklist:b", HASH_B)]
    assert [(u.key, u.judge_model, u.prompt_hash) for u in unit_triples(store, "faithfulness")
            ] == [("faithfulness", JUDGE_MODEL, HASH_A)]
