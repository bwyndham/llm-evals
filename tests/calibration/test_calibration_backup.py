"""Label backups: one record per distinct labeled output, restorable into an empty store."""

from __future__ import annotations

from llm_evals.calibration.backup import export_labels, read_labels, restore_labels, write_labels
from llm_evals.store.sqlite import Store
from llm_evals.types import Result


def _seed(store: Store) -> None:
    for run in ("r1", "r2"):  # r2 re-uses r1's outputs, as a re-judge does
        store.create_run(suite="s", kind="rescore", run_id=run)
        store.save_result(Result(run_id=run, case_id="c1", version_id="v", sample_idx=0, output={"title": "a"}))
        store.save_result(Result(run_id=run, case_id="c2", version_id="v", sample_idx=0, output={"title": "b"}))
    store.save_label("r1", "v", "c1", 0, "faithfulness", "fail", "ben", note="old", labeled_at="2026-01-01T00:00:00")
    store.save_label("r2", "v", "c1", 0, "faithfulness", "pass", "ben", note="rechecked", labeled_at="2026-01-02T00:00:00")
    store.save_label("r1", "v", "c2", 0, "faithfulness", "fail", "ben", labeled_at="2026-01-01T00:00:00")


def test_export_dedupes_copies_and_keeps_the_latest_verdict(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    _seed(store)
    records = export_labels(store)
    assert [(r["case_id"], r["label"], r["note"]) for r in records] == [("c2", "fail", ""), ("c1", "pass", "rechecked")]
    assert records[1]["output"] == {"title": "a"}


def test_round_trip_into_an_empty_store(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    _seed(store)
    path = write_labels(export_labels(store), tmp_path / "labels.jsonl")
    fresh = Store(tmp_path / "fresh.sqlite")
    run_id, n = restore_labels(fresh, read_labels(path), suite="s")
    assert n == 2
    labels = {l.case_id: l for l in fresh.labels_for(run_id=run_id)}
    assert labels["c1"].label == "pass" and labels["c1"].note == "rechecked"
    assert fresh.get_result(run_id, "v", "c2", 0).output == {"title": "b"}
    assert export_labels(fresh) == read_labels(path)  # restoring loses nothing
