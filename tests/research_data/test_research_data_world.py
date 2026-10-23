import json
import re
from pathlib import Path

import pytest

from llm_evals.cases.corpus import (
    DOC_SPECS,
    ENTITY_ATTRS,
    PLANT_FACTS,
    build_world,
    is_numeric,
    load_world_facts,
    normalize_answer,
    text_mentions,
)

SUITE_DIR = Path(__file__).resolve().parents[2] / "suites" / "research-agent"
FACTS_PATH = SUITE_DIR / "data" / "world_facts.jsonl"


@pytest.fixture(scope="module")
def facts():
    return load_world_facts(FACTS_PATH)


@pytest.fixture(scope="module")
def world(facts):
    return build_world(facts, seed=7)


def test_world_facts_shape(facts) -> None:
    assert 60 <= len(facts) <= 80
    assert len({f.id for f in facts}) == len(facts)
    for line in FACTS_PATH.read_text().splitlines():
        assert set(json.loads(line)) == {"id", "subject", "attribute", "value", "aliases", "sentence"}
    for f in facts:
        assert f.value in f.sentence and f.subject in f.sentence, f.id
        assert f.value not in f.aliases


def test_numeric_values_are_unique(facts) -> None:
    seen: dict[str, str] = {}
    for f in facts:
        if not is_numeric(f.attribute):
            continue
        for v in [f.value, *f.aliases]:
            num = normalize_answer(v).split()[0]
            assert seen.setdefault(num, f.id) == f.id, (num, f.id, seen[num])


def test_entity_values_name_entities(facts) -> None:
    subjects = {f.subject for f in facts}
    for f in facts:
        if f.attribute in {"headquarters", "province", "capital", "flagship_product"}:
            assert f.value in subjects, f.id
        if f.attribute in ENTITY_ATTRS:
            assert not re.search(r"\d", f.value)


def test_world_documents(world, facts) -> None:
    assert 30 <= len(world.docs) <= 40
    assert len({d.id for d in world.docs}) == len(world.docs)
    assert {fid for d in world.docs for fid in d.fact_ids} == {f.id for f in facts}
    for d in world.docs:
        assert 100 <= len(d.text.split()) <= 300, d.slug
        assert not re.search(r"\d", d.title)
        rest = d.text
        for fid in d.fact_ids:
            assert world.facts[fid].sentence in rest
            rest = rest.replace(world.facts[fid].sentence, "")
        assert not re.search(r"\d", rest), d.slug  # filler never carries numbers


def test_numeric_values_only_in_their_documents(world) -> None:
    for f in world.facts.values():
        if is_numeric(f.attribute):
            mentioning = {d.id for d in world.docs if text_mentions(d.text, f.value, f.aliases)}
            assert mentioning == {d.id for d in world.docs_for_fact(f.id)}, f.id


def test_lookalike_distractor_docs_exist(world) -> None:
    subjects = {d.subject for d in world.docs}
    for name in ["New Drennick", "Ilvane Junction", "Upper Kestrin", "Kestrin Lantern Works",
                 "Velmouth Ferry Society", "Caddow Hill", "Greywater Viaduct"]:
        assert name in subjects


def test_planted_facts_are_not_world_facts(world) -> None:
    keys = {(f.subject, f.attribute) for f in world.facts.values()}
    assert len({p.id for p in PLANT_FACTS}) == len(PLANT_FACTS)
    for p in PLANT_FACTS:
        assert (p.subject, p.attribute) not in keys
        assert p.value in p.sentence
        assert not any(text_mentions(d.text, p.value, p.aliases) for d in world.docs)


def test_world_is_deterministic(facts) -> None:
    a, b = build_world(facts, seed=7), build_world(facts, seed=7)
    assert [(d.id, d.text) for d in a.docs] == [(d.id, d.text) for d in b.docs]
    c = build_world(facts, seed=8)
    assert [d.id for d in a.docs] != [d.id for d in c.docs]
    assert len(DOC_SPECS) == len(a.docs)
