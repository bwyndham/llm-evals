import shutil
from collections import Counter
from pathlib import Path

import pytest

from llm_evals.cases import case_set_hash, load_cases
from llm_evals.cases.corpus import (
    answer_matches,
    build_research_cases,
    generate_cases,
    load_world,
    text_mentions,
)
from llm_evals.types import Case

SUITE_DIR = Path(__file__).resolve().parents[2] / "suites" / "research-agent"
CASES_PATH = SUITE_DIR / "data" / "cases.jsonl"


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return load_cases(CASES_PATH)


@pytest.fixture(scope="module")
def world():
    return load_world(SUITE_DIR, seed=7)


def _by_op(cases, op):
    out = [c for c in cases if c.source["perturbation"]["op"] == op]
    assert out, op
    return out


def _all_text(case: Case) -> list[str]:
    return [d["title"] + "\n" + d["text"] for d in case.fixtures["docs"]]


def test_checked_in_cases_are_current(tmp_path: Path, cases) -> None:
    copy = tmp_path / "suite"
    shutil.copytree(SUITE_DIR, copy, ignore=shutil.ignore_patterns("cases.jsonl"))
    rebuilt = build_research_cases(copy)
    assert rebuilt.read_bytes() == CASES_PATH.read_bytes()
    assert case_set_hash(load_cases(rebuilt)) == case_set_hash(cases)


def test_same_seed_same_cases(tmp_path: Path, world) -> None:
    a = generate_cases(world, n_base=40, seed=7)
    b = generate_cases(load_world(SUITE_DIR, seed=7), n_base=40, seed=7)
    assert case_set_hash(a) == case_set_hash(b)
    other = generate_cases(load_world(SUITE_DIR, seed=3), n_base=40, seed=3)
    assert case_set_hash(other) != case_set_hash(a)


def test_cases_validate_and_follow_contract(cases) -> None:
    assert 90 <= len(cases) <= 110
    for c in cases:
        Case.model_validate(c.model_dump())
        assert set(c.input) == {"question"} and c.input["question"].endswith("?")
        assert c.provenance == "constructed"
        assert set(c.fixtures) == {"corpus_id", "docs"}
        docs = c.fixtures["docs"]
        assert 8 <= len(docs) <= 12, c.id
        assert len({d["id"] for d in docs}) == len(docs)
        assert all(set(d) == {"id", "title", "text"} for d in docs)
        exp = c.expected
        assert {"behavior", "answer", "aliases", "original_answer"} <= set(exp)
        assert exp["behavior"] in {"answer", "abstain", "flag_conflict"}
        assert (exp["answer"] is None) == (exp["behavior"] != "answer")
        assert set(c.strata) == {"qtype", "perturbation", "behavior"}
        assert c.strata["behavior"] == exp["behavior"]
        assert c.strata["perturbation"] == c.source["perturbation"]["op"]
        assert isinstance(c.source["facts"], list) and c.source["facts"]


def test_ids_and_suffixes(cases) -> None:
    ids = {c.id for c in cases}
    suffix = {"plant_fact": "-plant", "remove_answer_doc": "-remove", "change_fact": "-change",
              "add_conflict": "-conflict"}
    for c in cases:
        op = c.source["perturbation"]["op"]
        if op == "none":
            assert not any(c.id.endswith(s) for s in suffix.values())
        else:
            assert c.id.endswith(suffix[op])
            assert c.source["perturbation"]["base_case_id"] in ids


def test_answer_cases_have_the_answer_in_the_corpus(cases) -> None:
    for c in cases:
        if c.expected["behavior"] != "answer":
            continue
        exp = c.expected
        if c.strata["qtype"] == "comparison":
            subjects = [f["subject"] for f in c.source["facts"]]
            assert exp["answer"] in subjects
        else:
            assert any(text_mentions(t, exp["answer"], exp["aliases"]) for t in _all_text(c)), c.id
        texts = {d["id"]: d["text"] for d in c.fixtures["docs"]}
        for f in c.source["facts"]:
            assert f["doc_ids"], (c.id, f["id"])
            assert all(text_mentions(texts[i], f["value"], f["aliases"]) for i in f["doc_ids"])
        assert answer_matches(exp["answer"], exp)


def test_remove_answer_doc_lacks_the_answer(cases) -> None:
    for c in _by_op(cases, "remove_answer_doc"):
        p = c.source["perturbation"]
        assert c.expected["answer"] is None and c.expected["behavior"] == "abstain"
        assert c.expected["original_answer"] == p["removed_value"]
        doc_ids = {d["id"] for d in c.fixtures["docs"]}
        assert not set(p["removed_doc_ids"]) & doc_ids
        for text in _all_text(c):
            assert p["removed_value"].lower() not in text.lower()
            for alias in p["removed_aliases"]:
                assert alias.lower() not in text.lower()
            assert not text_mentions(text, p["removed_value"], p["removed_aliases"])


def test_change_fact_corpus_has_only_the_new_value(cases) -> None:
    for c in _by_op(cases, "change_fact"):
        p = c.source["perturbation"]
        assert p["new_value"] != p["old_value"]
        texts = _all_text(c)
        for t in texts:
            assert not text_mentions(t, p["old_value"], p["old_aliases"]), c.id
        assert any(p["new_value"] in t for t in texts)
        assert c.expected["behavior"] == "answer"
        assert c.expected["original_answer"] is not None
        assert c.expected["answer"] != c.expected["original_answer"]
        assert not answer_matches(c.expected["original_answer"], c.expected)
        if c.strata["qtype"] != "comparison":
            assert c.expected["answer"] == p["new_value"]


def test_change_fact_flips_comparisons(cases) -> None:
    by_id = {c.id: c for c in cases}
    flips = [c for c in _by_op(cases, "change_fact") if c.strata["qtype"] == "comparison"]
    assert flips
    for c in flips:
        base = by_id[c.source["perturbation"]["base_case_id"]]
        assert c.expected["original_answer"] == base.expected["answer"]
        assert c.expected["answer"] != base.expected["answer"]


def test_add_conflict_has_both_values(cases) -> None:
    for c in _by_op(cases, "add_conflict"):
        p = c.source["perturbation"]
        old, new = p["values"]
        assert old != new and c.expected["conflicting_values"] == [old, new]
        assert c.expected["answer"] is None and c.expected["behavior"] == "flag_conflict"
        texts = {d["id"]: d["text"] for d in c.fixtures["docs"]}
        assert p["doc_ids"]["original"] and all(old in texts[i] for i in p["doc_ids"]["original"])
        (conflict_id,) = p["doc_ids"]["conflict"]
        assert new in texts[conflict_id] and old not in texts[conflict_id]
        assert all(not text_mentions(texts[i], new) for i in texts if i != conflict_id)


def test_plant_fact_cases(cases, world) -> None:
    world_values = {f.value for f in world.facts.values()}
    for c in _by_op(cases, "plant_fact"):
        p = c.source["perturbation"]
        assert c.expected["answer"] == p["fact"]["value"] and c.expected["original_answer"] is None
        assert c.expected["answer"] not in world_values
        hits = [d["id"] for d in c.fixtures["docs"]
                if text_mentions(d["text"], p["fact"]["value"], p["fact"]["aliases"])]
        assert hits == [p["doc_id"]]


def test_distribution(cases) -> None:
    by = {k: Counter(c.strata[k] for c in cases) for k in ("qtype", "perturbation", "behavior")}
    assert set(by["qtype"]) == {"lookup", "two_hop", "comparison"}
    assert min(by["qtype"].values()) >= 0.25 * len(cases)
    assert set(by["perturbation"]) == {"none", "plant_fact", "remove_answer_doc", "change_fact", "add_conflict"}
    assert by["perturbation"]["none"] == 40
    assert by["behavior"]["abstain"] >= 0.2 * len(cases)
    assert by["behavior"]["flag_conflict"] >= 0.2 * len(cases)
    assert by["behavior"]["answer"] <= 0.6 * len(cases)
    for qtype in ("lookup", "two_hop", "comparison"):
        ops = {c.strata["perturbation"] for c in cases if c.strata["qtype"] == qtype}
        assert {"none", "change_fact", "add_conflict"} <= ops


def test_two_hop_needs_two_documents(cases) -> None:
    for c in cases:
        if c.strata["qtype"] == "two_hop" and c.strata["perturbation"] == "none":
            bridge, final = c.source["facts"]
            assert bridge["value"] == final["subject"]
            assert not set(bridge["doc_ids"]) & set(final["doc_ids"])


def test_lookalike_distractors_are_included(cases) -> None:
    c = next(c for c in cases if "Drennick" in c.input["question"] and c.strata["perturbation"] == "none")
    titles = " ".join(d["title"] for d in c.fixtures["docs"])
    assert "New Drennick" in titles
