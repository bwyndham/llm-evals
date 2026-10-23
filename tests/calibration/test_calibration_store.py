from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llm_evals.store.sqlite import (
    _SCHEMA,
    SCHEMA_VERSION,
    CalibrationRecord,
    Store,
    StoreError,
    unit_entries,
)
from llm_evals.types import Score

from calibration_helpers import HASH_A, HASH_B, JUDGE_MODEL, make_case, seed_run


def record(n: int = 50, kappa: float | None = 0.6, approved: bool = True,
           prompt_hash: str = HASH_A, unit: str = "faithfulness",
           model: str = JUDGE_MODEL) -> CalibrationRecord:
    return CalibrationRecord(unit_key=unit, judge_model=model, prompt_hash=prompt_hash, n=n,
                             agreement=0.9, kappa=kappa, sensitivity=0.8, specificity=0.95,
                             prevalence=0.3, approved=approved,
                             label_set={"run_ids": ["r1"], "count": n})


def test_migrates_a_v1_file(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)  # v1 tables only
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute("INSERT INTO runs (id, suite, kind, started_at, status) "
                 "VALUES ('r1', 'spec-writing', 'compare', '2026-01-01', 'done')")
    conn.commit()
    conn.close()
    with Store(path) as s:
        assert s._read("SELECT version FROM schema_version")[0]["version"] == SCHEMA_VERSION == 2
        assert s.get_run("r1").status == "done"  # existing data intact
        s.save_label("r1", "v1", "c1", 0, "faithfulness", "pass", "ben")
        s.save_calibration(record())
    with Store(path) as s:  # reopening a v2 file is a no-op
        assert len(s.labels_for()) == 1
        assert s.latest_calibration("faithfulness", JUDGE_MODEL, HASH_A).n == 50
        assert len(s._read("SELECT version FROM schema_version")) == 1


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "new.sqlite"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError):
        Store(path)


def test_label_round_trip_and_relabel(store: Store) -> None:
    store.save_label("r1", "v1", "c1", 0, "faithfulness", "pass", "ben", note="fine")
    store.save_label("r1", "v1", "c1", 0, "faithfulness", "fail", "ben", note="missed a2")
    store.save_label("r1", "v1", "c1", 0, "faithfulness", "pass", "ana")
    store.save_label("r2", "v1", "c1", 0, "checklist:testable_ac", "fail", "ben")
    ben = store.labels_for(unit_key="faithfulness", labeler="ben")
    assert len(ben) == 1 and ben[0].label == "fail" and ben[0].note == "missed a2"
    assert ben[0]["labeled_at"]
    assert len(store.labels_for(unit_key="faithfulness")) == 2
    assert [lab.run_id for lab in store.labels_for(run_id="r2")] == ["r2"]
    assert len(store.labels_for()) == 3
    with pytest.raises(ValueError):
        store.save_label("r1", "v1", "c1", 0, "faithfulness", "maybe", "ben")


def test_calibration_round_trip_and_history(store: Store) -> None:
    first = store.save_calibration(record(n=40, approved=False))
    second = store.save_calibration(record(n=60, kappa=0.7))
    store.save_calibration(record(prompt_hash=HASH_B, kappa=0.2, approved=False))
    assert first.seq < second.seq and second.computed_at
    latest = store.latest_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    assert latest.n == 60 and latest.kappa == 0.7 and latest.approved is True
    assert latest.label_set == {"run_ids": ["r1"], "count": 60}
    assert [r.n for r in store.calibration_history("faithfulness")] == [50, 60, 40]
    assert {(r.prompt_hash, r.n) for r in store.latest_calibrations()} == {(HASH_A, 60),
                                                                            (HASH_B, 50)}
    assert store.latest_calibration("faithfulness", "other-model", HASH_A) is None


def test_approved_calibration_bars(store: Store) -> None:
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)  # none yet
    store.save_calibration(record(n=50, kappa=0.6))
    assert store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)  # bars inclusive
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A, min_n=51)
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A, min_kappa=0.61)
    store.save_calibration(record(n=49, kappa=0.9))  # latest wins, and it is short
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)


def test_approved_calibration_is_exact_to_the_triple(store: Store) -> None:
    store.save_calibration(record())
    assert store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_B)
    assert not store.approved_calibration("faithfulness", "claude-opus-5-5", HASH_A)
    assert not store.approved_calibration("checklist:testable_ac", JUDGE_MODEL, HASH_A)


def test_approved_needs_flag_and_kappa(store: Store) -> None:
    store.save_calibration(record(n=80, kappa=0.9, approved=False))
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    store.save_calibration(record(n=80, kappa=None, approved=True))
    assert not store.approved_calibration("faithfulness", JUDGE_MODEL, HASH_A)


def test_unit_entries_fall_back_to_score_fields() -> None:
    score = Score(check="checklist", kind="judge", passed=False, judge_model=JUDGE_MODEL,
                  prompt_hash="score-hash", sub_scores=[
                      {"unit": "checklist:a", "passed": True, "prompt_hash": "unit-hash"},
                      {"unit": "checklist:b", "passed": False},
                      {"claim": "not a unit", "passed": False},
                  ])
    assert unit_entries(score) == [("checklist:a", True, JUDGE_MODEL, "unit-hash"),
                                   ("checklist:b", False, JUDGE_MODEL, "score-hash")]


def test_run_cases_and_unit_verdicts(store: Store) -> None:
    run = seed_run(store, {"c1": {"faithfulness": True}, "c2": {"faithfulness": False}},
                   check="faithfulness", per_unit_hash=False)
    assert set(store.run_cases(run)) == {"c1", "c2"}
    assert store.run_cases(run)["c1"] == make_case("c1")
    verdicts = store.unit_verdicts("faithfulness")
    assert [(v.case_id, v.passed, v.prompt_hash) for v in verdicts] == [
        ("c1", True, HASH_A), ("c2", False, HASH_A)]
    assert store.unit_verdicts("faithfulness", check="other") == []
