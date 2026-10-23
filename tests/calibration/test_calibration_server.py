"""The local labeling app's API: blind samples, saved labels, calibration."""

from __future__ import annotations

import json
import threading
import urllib.request

from llm_evals.calibration.server import LabelSession, serve
from llm_evals.types import Case, Result, Score


def _seed(store):
    store.create_run(suite="s", kind="rescore", run_id="r1")
    cases = {}
    for i in range(6):
        case = Case(id=f"c{i}", input={"ask": f"ask {i}"}, provenance="constructed",
                    source={"atoms": [{"id": "s/a1", "text": "do x", "keywords": ["x"], "mode": "explicit"},
                                      {"id": "s/a2", "text": "maybe y", "mode": "hinted"}],
                            "hinted_atom_ids": ["s/a2"]})
        cases[case.id] = case
        store.save_result(Result(run_id="r1", case_id=case.id, version_id="v", sample_idx=0,
                                 output={"title": "t", "acceptance_criteria": "Feature: f"}))
        store.save_score("r1", "v", case.id, 0, Score(
            check="faithfulness", kind="faithfulness", passed=i % 2 == 0, reason="SECRET REASON",
            judge_model="judge-x", prompt_hash="h1",
            sub_scores=[{"unit": "faithfulness", "passed": i % 2 == 0, "prompt_hash": "h1", "judge_model": "judge-x"}]))
    return cases


def _call(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()


def test_label_app_round_trip(tmp_path):
    from llm_evals.store.sqlite import Store

    store = Store(tmp_path / "s.sqlite")
    cases = _seed(store)
    session = LabelSession(store=store, run_id="r1", check="faithfulness", labeler="ben",
                           cases=cases, target=6, criteria="the rule")
    server = serve(session, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = _call(port, "/")
        assert "Judge calibration" in page
        raw = _call(port, "/api/session")
        state = json.loads(raw)
        assert len(state["samples"]) == 6 and state["criteria"] == "the rule"
        assert "SECRET REASON" not in raw and '"passed"' not in raw  # the judge's verdict never reaches the page
        assert any(a["hinted"] for a in state["samples"][0]["case"]["atoms"])
        for s in state["samples"]:
            judged_pass = int(s["key"][2][1:]) % 2 == 0
            _call(port, "/api/label", {"key": s["key"], "labels": {"faithfulness": "pass" if judged_pass else "fail"}})
        assert len(store.labels_for(run_id="r1", labeler="ben")) == 6
        results = json.loads(_call(port, "/api/calibrate", {}))["results"]
        assert results[0]["unit"] == "faithfulness" and results[0]["agreement"] == 1.0
    finally:
        server.shutdown()


def test_reload_drops_labeled_and_skipped_samples(tmp_path):
    from llm_evals.store.sqlite import Store

    store = Store(tmp_path / "s.sqlite")
    cases = _seed(store)
    session = LabelSession(store=store, run_id="r1", check="faithfulness", labeler="ben",
                           cases=cases, target=6)
    server = serve(session, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        first = json.loads(_call(port, "/api/session"))["samples"]
        _call(port, "/api/label", {"key": first[0]["key"], "labels": {"faithfulness": "pass"}})
        _call(port, "/api/skip", {"key": first[1]["key"]})
        again = json.loads(_call(port, "/api/session"))
        keys = [s["key"] for s in again["samples"]]
        assert first[0]["key"] not in keys and first[1]["key"] not in keys
        assert len(keys) == 4 and again["already_done"] == 1 and again["skipped"] == 1
        assert again["seen_cases"] == [first[0]["key"][2]]
    finally:
        server.shutdown()


def test_export_is_blind_and_import_round_trips(tmp_path):
    from llm_evals.calibration.server import export_blind, import_labels
    from llm_evals.store.sqlite import Store

    store = Store(tmp_path / "s.sqlite")
    cases = _seed(store)
    store.save_label("r1", "v", "c0", 0, "faithfulness", "pass", "ben")  # labeled earlier: left out
    session = LabelSession(store=store, run_id="r1", check="faithfulness", labeler="ben",
                           cases=cases, target=6)
    data, keys = export_blind(session)
    raw = json.dumps(data)
    assert len(data["samples"]) == 5 and "c0" not in {s["case"]["id"] for s in data["samples"]}
    assert "SECRET REASON" not in raw and '"passed"' not in raw and '"key"' not in raw and "r1" not in raw
    assert sorted(keys) == [s["id"] for s in data["samples"]]
    s1, s2, s3 = (s["id"] for s in data["samples"][:3])
    counts = import_labels(store, keys, {s1: {"labels": {"faithfulness": "fail"}, "note": "invented"},
                                         s2: {"skipped": True}, "zz": {"labels": {"faithfulness": "pass"}},
                                         s3: {"labels": {"faithfulness": "maybe"}}}, "ben")
    assert counts == {"saved": 1, "skipped": 2, "unknown": 1}
    labels = store.labels_for(run_id="r1", labeler="ben")
    assert len(labels) == 2 and any(l.label == "fail" and l.note == "invented" for l in labels)


def test_copy_labels_only_onto_identical_outputs(tmp_path):
    from llm_evals.calibration.server import copy_labels
    from llm_evals.store.sqlite import Store

    store = Store(tmp_path / "s.sqlite")
    _seed(store)
    store.create_run(suite="s", kind="rescore", run_id="r2")
    store.save_result(Result(run_id="r2", case_id="c0", version_id="v", sample_idx=0,
                             output={"title": "t", "acceptance_criteria": "Feature: f"}))
    store.save_result(Result(run_id="r2", case_id="c1", version_id="v", sample_idx=0,
                             output={"title": "changed"}))
    for c in ("c0", "c1", "c2"):
        store.save_label("r1", "v", c, 0, "faithfulness", "fail", "ben", note="n")
    assert copy_labels(store, "r1", "r2", "ben") == {"copied": 1, "different_output": 1, "missing": 1}
    [lab] = store.labels_for(run_id="r2", labeler="ben")
    assert (lab.case_id, lab.label, lab.note) == ("c0", "fail", "n")
