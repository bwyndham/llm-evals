from __future__ import annotations

from llm_evals.calibration.sampling import candidate_samples, choose_samples
from llm_evals.store.sqlite import Store

from calibration_helpers import seed_run

UNITS = ("checklist:a", "checklist:b")


def seed(store: Store, n_fail: int, n_pass: int, **kw) -> str:
    verdicts = {}
    for i in range(n_fail):
        verdicts[f"f{i:03d}"] = {"checklist:a": False, "checklist:b": True}
    for i in range(n_pass):
        verdicts[f"p{i:03d}"] = {"checklist:a": True, "checklist:b": True}
    return seed_run(store, verdicts, **kw)


def test_stratified_half_fail(store: Store) -> None:
    run = seed(store, 40, 60)
    picks = choose_samples(store, run, "checklist", labeler="ben")
    assert len(picks) == 50
    assert sum(not p.judge_passed for p in picks) == 25
    assert len({p.key for p in picks}) == 50
    assert all(p.units == UNITS for p in picks)


def test_short_stratum_is_filled_from_the_other(store: Store) -> None:
    run = seed(store, 10, 60)
    picks = choose_samples(store, run, "checklist", labeler="ben")
    assert len(picks) == 50 and sum(not p.judge_passed for p in picks) == 10
    run2 = seed(store, 60, 5, run_id="run2")
    picks2 = choose_samples(store, run2, "checklist", labeler="ben")
    assert len(picks2) == 50 and sum(p.judge_passed for p in picks2) == 5
    assert len(choose_samples(store, seed(store, 3, 4, run_id="run3"), "checklist",
                              labeler="ben")) == 7


def test_deterministic_with_seed_and_shuffled(store: Store) -> None:
    run = seed(store, 40, 60)
    a = choose_samples(store, run, "checklist", labeler="ben", seed=7)
    b = choose_samples(store, run, "checklist", labeler="ben", seed=7)
    c = choose_samples(store, run, "checklist", labeler="ben", seed=8)
    assert a == b
    assert [p.key for p in a] != [p.key for p in c]
    verdicts = [p.judge_passed for p in a]
    assert verdicts != sorted(verdicts) and verdicts != sorted(verdicts, reverse=True)


def test_skips_samples_this_labeler_finished(store: Store) -> None:
    run = seed(store, 2, 2)
    store.save_label(run, "v1", "f000", 0, "checklist:a", "fail", "ben")
    store.save_label(run, "v1", "f000", 0, "checklist:b", "pass", "ben")
    store.save_label(run, "v1", "p000", 0, "checklist:a", "pass", "ben")  # partial
    ids = {p.case_id for p in choose_samples(store, run, "checklist", labeler="ben")}
    assert ids == {"f001", "p000", "p001"}
    assert len(choose_samples(store, run, "checklist", labeler="ana")) == 4


def test_held_out_skipped_only_if_asked(store: Store) -> None:
    run = seed(store, 2, 2, held_out={"f000", "p001"})
    assert len(choose_samples(store, run, "checklist", labeler="ben")) == 4
    ids = {p.case_id for p in choose_samples(store, run, "checklist", labeler="ben",
                                             skip_held_out=True)}
    assert ids == {"f001", "p000"}


def test_held_out_unknown_cases_are_skipped(store: Store) -> None:
    run = seed(store, 2, 2, save_cases=False)
    assert choose_samples(store, run, "checklist", labeler="ben", skip_held_out=True) == []


def test_errored_results_and_other_checks_excluded(store: Store) -> None:
    run = seed(store, 2, 2, errored={"f000"})
    seed_run(store, {"x1": {"faithfulness": False}}, check="faithfulness", run_id=run)
    assert {s.case_id for s in candidate_samples(store, run, "checklist")} == {
        "f001", "p000", "p001"}
    assert choose_samples(store, run, "checklist", labeler="ben", n=0) == []
