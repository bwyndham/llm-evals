from __future__ import annotations

import threading
from pathlib import Path

import pytest

from llm_evals.store import Store
from llm_evals.store.sqlite import SCHEMA_VERSION
from llm_evals.types import Decision, ErrorKind, Result, Score, Trace, VersionManifest


@pytest.fixture
def store() -> Store:
    return Store(Path(":memory:"))


def result(case_id: str = "c1", sample_idx: int = 0, **kw) -> Result:
    return Result(run_id="run1", case_id=case_id, version_id="v1", sample_idx=sample_idx, **kw)


def manifest(name: str, mid: str) -> VersionManifest:
    return VersionManifest(name=name, model="claude-haiku-5-5",
                           prompt_templates={"user": "/x/user.j2"}, id=mid)


def test_schema_version_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "db" / "evals.sqlite"
    with Store(path) as s:
        run_id = s.create_run(suite="spec-writing", kind="compare")
    with Store(path) as s:
        assert s.get_run(run_id).suite == "spec-writing"
        assert s._read("SELECT version FROM schema_version")[0]["version"] == SCHEMA_VERSION


def test_run_round_trip(store: Store) -> None:
    run_id = store.create_run(suite="spec-writing", kind="compare", config_hash="cfg",
                              case_set_hash="cs", baseline_version_id="b",
                              candidate_version_id="c", meta={"k": 3})
    assert len(run_id) == 32
    run = store.get_run(run_id)
    assert (run.status, run.meta, run.decision) == ("running", {"k": 3}, None)
    store.update_run(run_id, status="done", decision=Decision.PROMOTE,
                     report_path=Path("/tmp/r.md"), meta={"k": 4})
    run = store.get_run(run_id)
    assert (run.status, run.decision, run.report_path, run.meta) == (
        "done", "PROMOTE", "/tmp/r.md", {"k": 4})
    assert store.get_run("missing") is None
    with pytest.raises(ValueError):
        store.update_run(run_id, bogus=1)
    with pytest.raises(KeyError):
        store.update_run("missing", status="x")


def test_result_round_trip_and_cache(store: Store) -> None:
    r = result(output={"title": "t"}, raw_output='{"title": "t"}',
               trace=Trace(latency_ms=12.5, input_tokens=3, cost_usd=0.01, request_ids=["q"]))
    assert store.get_result("run1", "v1", "c1", 0) is None  # miss
    store.save_result(r)
    assert store.get_result("run1", "v1", "c1", 0) == r  # hit: no system call needed
    assert store.get_result("run1", "v1", "c1", 1) is None
    assert store.get_result("run1", "v2", "c1", 0) is None


def test_infra_results_can_be_excluded_from_cache(store: Store) -> None:
    store.save_result(result(error_kind=ErrorKind.INFRA, error="timeout"))
    assert store.get_result("run1", "v1", "c1", 0).error_kind is ErrorKind.INFRA
    assert store.get_result("run1", "v1", "c1", 0, include_infra=False) is None
    store.save_result(result(output={"a": 1}))  # retry replaces it
    assert store.get_result("run1", "v1", "c1", 0, include_infra=False).output == {"a": 1}
    assert len(store.results_for("run1", "v1")) == 1


def test_results_for_ordering(store: Store) -> None:
    for case_id, idx in [("b", 1), ("a", 1), ("b", 0), ("a", 0)]:
        store.save_result(result(case_id, idx))
    got = [(r.case_id, r.sample_idx) for r in store.results_for("run1", "v1")]
    assert got == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert store.results_for("run1", "other") == []


def test_scores_round_trip(store: Store) -> None:
    s1 = Score(check="json_shape", kind="structural", passed=True)
    s2 = Score(check="faithful", kind="faithfulness", passed=False, value=0.4,
               sub_scores=[{"claim": "x", "ok": False}])
    store.save_score("run1", "v1", "c1", 0, s2)
    store.save_score("run1", "v1", "c1", 0, s1)
    store.save_score("run1", "v1", "c1", 0, s1.model_copy(update={"reason": "again"}))  # upsert
    got = store.scores_for("run1", "v1")
    assert [(c, i, s.check) for c, i, s in got] == [("c1", 0, "faithful"), ("c1", 0, "json_shape")]
    assert got[0][2] == s2 and got[1][2].reason == "again"


def test_baseline_history(store: Store) -> None:
    assert store.current_baseline("spec-writing") is None
    store.promote("spec-writing", manifest("v1", "aaa"), "versions/v1.yaml", "run1")
    store.promote("other-suite", manifest("x", "zzz"), None, None)
    store.promote("spec-writing", manifest("v2", "bbb"), Path("versions/v2.yaml"), "run2")
    store.promote("spec-writing", manifest("v1", "aaa"), "versions/v1.yaml", "run3")  # rollback

    current = store.current_baseline("spec-writing")
    assert (current.version_id, current.run_id, current.manifest.name) == ("aaa", "run3", "v1")
    history = store.baseline_history("spec-writing")
    assert [(b.version_id, b.run_id) for b in history] == [
        ("aaa", "run3"), ("bbb", "run2"), ("aaa", "run1")]
    assert history[1].manifest_path == history[1]["manifest_path"] == "versions/v2.yaml"
    assert history[0].promoted_at >= history[1].promoted_at >= history[2].promoted_at
    assert store.current_baseline("other-suite").version_id == "zzz"


def test_promote_requires_id(store: Store) -> None:
    with pytest.raises(ValueError):
        store.promote("s", manifest("v", ""), None, None)


def test_concurrent_writes(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.sqlite")

    def write(t: int) -> None:
        for i in range(50):
            store.save_result(result(f"c{t}", i))

    threads = [threading.Thread(target=write, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(store.results_for("run1", "v1")) == 400
