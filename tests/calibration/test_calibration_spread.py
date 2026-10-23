"""choose_samples covers distinct cases before repeating any."""

from pathlib import Path

from llm_evals.calibration import choose_samples
from llm_evals.store.sqlite import Store
from llm_evals.types import Result, Score


def test_distinct_cases_come_first(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.create_run(suite="s", kind="rescore", run_id="r")
    for c in range(10):
        for v in ("a", "b"):
            for k in range(3):
                store.save_result(Result(run_id="r", case_id=f"c{c}", version_id=v, sample_idx=k, output={"x": 1}))
                ok = (c + k) % 2 == 0
                store.save_score("r", v, f"c{c}", k, Score(check="faithfulness", kind="faithfulness", passed=ok,
                                 judge_model="j", prompt_hash="h",
                                 sub_scores=[{"unit": "faithfulness", "passed": ok, "judge_model": "j", "prompt_hash": "h"}]))
    picks = choose_samples(store, "r", "faithfulness", labeler="ben", n=16, seed=1)
    ids = [p.case_id for p in picks]
    assert len(ids) == 16
    assert len(set(ids[:10])) == 10  # every case once before any repeats
    assert sum(not p.judge_passed for p in picks) == 8  # verdict balance kept


def test_a_case_in_both_strata_is_not_picked_twice(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.create_run(suite="s", kind="rescore", run_id="r")
    for c in range(40):
        for v, ok in (("a", False), ("b", True)):  # every case has one fail and one pass
            store.save_result(Result(run_id="r", case_id=f"c{c}", version_id=v, sample_idx=0, output={"x": 1}))
            store.save_score("r", v, f"c{c}", 0, Score(check="faithfulness", kind="faithfulness", passed=ok,
                             judge_model="j", prompt_hash="h",
                             sub_scores=[{"unit": "faithfulness", "passed": ok, "judge_model": "j", "prompt_hash": "h"}]))
    picks = choose_samples(store, "r", "faithfulness", labeler="ben", n=30, seed=0)
    assert len({p.case_id for p in picks}) == 30
    assert sum(not p.judge_passed for p in picks) == 15
