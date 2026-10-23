import itertools
from collections import Counter

import pytest

from llm_evals.cases.renderers.ask_renderer import (
    FEATURE_VALUES,
    AskRenderer,
    case_id_for,
    reword_ask,
)
from llm_evals.types import Case

ALL_COMBOS = [dict(zip(FEATURE_VALUES, c)) for c in itertools.product(*FEATURE_VALUES.values())]


@pytest.fixture(scope="module")
def suite(renderer: AskRenderer) -> list[Case]:
    return renderer.render_suite(120, seed=7)


@pytest.fixture(scope="module")
def grid_cases(renderer: AskRenderer) -> list[Case]:
    # every feature combination, several seeds each
    return [renderer.render(f, seed) for f in ALL_COMBOS for seed in (0, 1, 2)]


def _all_actor_forms(renderer: AskRenderer) -> set[str]:
    forms = set()
    for s in renderer.atom_sets:
        forms |= {s.actor.lower(), s.actor.lower() + "s"}
    return forms


def test_registered() -> None:
    from llm_evals import registry

    import llm_evals.cases.renderers  # noqa: F401

    assert registry._REGISTRIES["renderer"]["ask_renderer"] is AskRenderer


def test_determinism(renderer: AskRenderer) -> None:
    f = {"vagueness": "vague", "bundling": "bundled", "tone": "rambling",
         "missing_actor": "yes", "source_format": "email"}
    assert renderer.render(f, 42) == renderer.render(f, 42)
    assert renderer.render(f, 42).input != renderer.render(f, 43).input
    a = renderer.render_suite(30, seed=5)
    b = renderer.render_suite(30, seed=5)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    assert [c.id for c in renderer.render_suite(30, seed=6)] != [c.id for c in a]


def test_ids_stable(renderer: AskRenderer) -> None:
    f = {"vagueness": "explicit", "bundling": "single", "tone": "neutral",
         "missing_actor": "no", "source_format": "slack"}
    case = renderer.render(f, 3)
    assert case.id == case_id_for(case.source["atom_set_ids"], case.strata, 3)
    assert case.id.startswith("ask-") and len(case.id) == 16
    # a fresh renderer gives the same id
    assert AskRenderer(renderer.atoms_path).render(f, 3).id == case.id
    # id depends on features
    assert renderer.render({**f, "tone": "urgent"}, 3).id != case.id


def test_case_shape(grid_cases: list[Case]) -> None:
    for c in grid_cases:
        assert c.provenance == "constructed"
        assert set(c.input) == {"ask"} and c.input["ask"].strip()
        assert set(c.strata) == set(FEATURE_VALUES)
        assert all(isinstance(v, str) for v in c.strata.values())
        src = c.source
        for key in ("atoms", "actor", "atom_set_ids", "hinted_atom_ids"):
            assert key in src
        ids = [a["id"] for a in src["atoms"]]
        assert len(ids) == len(set(ids))
        assert {a["set_id"] for a in src["atoms"]} == set(src["atom_set_ids"])
        assert set(src["hinted_atom_ids"]) <= set(ids)


def test_render_suite_coverage(suite: list[Case]) -> None:
    assert len(suite) == 120
    assert len({c.id for c in suite}) == 120
    for name, values in FEATURE_VALUES.items():
        counts = Counter(c.strata[name] for c in suite)
        assert set(counts) == set(values)
        assert max(counts.values()) - min(counts.values()) <= 1, (name, counts)
    combos = {tuple(c.strata[k] for k in FEATURE_VALUES) for c in suite}
    assert len(combos) == len(ALL_COMBOS)  # 120 >= 108, so the full grid is covered


def test_render_suite_small_and_restricted(renderer: AskRenderer) -> None:
    cases = renderer.render_suite(10, seed=1, features={"source_format": ["email"], "tone": ["urgent", "neutral"]})
    assert len(cases) == 10
    assert {c.strata["source_format"] for c in cases} == {"email"}
    assert Counter(c.strata["tone"] for c in cases) == {"urgent": 5, "neutral": 5}
    with pytest.raises(ValueError):
        renderer.render_suite(3, seed=1, features={"tone": ["shouty"]})


def test_atom_sets_used_evenly(renderer: AskRenderer, suite: list[Case]) -> None:
    uses = Counter(s for c in suite for s in c.source["atom_set_ids"])
    assert set(uses) == {s.id for s in renderer.atom_sets}
    assert max(uses.values()) - min(uses.values()) <= 1


def test_non_hinted_atoms_present_in_text(grid_cases: list[Case]) -> None:
    for c in grid_cases:
        text = c.input["ask"].lower()
        for atom in c.source["atoms"]:
            if atom["id"] in c.source["hinted_atom_ids"]:
                assert atom["mode"] == "hinted"
                continue
            for kw in atom["keywords"]:
                assert kw in text, (c.id, atom["id"], kw)
            # the exact rendered phrase is in the text
            assert atom["phrase"].lower() in text, (c.id, atom["id"])


def test_vagueness_modes(grid_cases: list[Case]) -> None:
    for c in grid_cases:
        modes = Counter(a["mode"] for a in c.source["atoms"])
        n_sets = len(c.source["atom_set_ids"])
        v = c.strata["vagueness"]
        if v == "hinted":
            assert modes["hinted"] == n_sets  # exactly one implied atom per set
            for sid in c.source["atom_set_ids"]:
                per_set = [a for a in c.source["atoms"] if a["set_id"] == sid and a["mode"] == "hinted"]
                assert len(per_set) == 1
            assert set(modes) <= {"explicit", "hinted"}
        else:
            assert set(modes) == {v} and not c.source["hinted_atom_ids"]


def test_bundled_cases_carry_both_sets(renderer: AskRenderer, grid_cases: list[Case]) -> None:
    by_id = {s.id: s for s in renderer.atom_sets}
    for c in grid_cases:
        set_ids = c.source["atom_set_ids"]
        assert len(set_ids) == (2 if c.strata["bundling"] == "bundled" else 1)
        assert len(set(set_ids)) == len(set_ids)
        expected = {f"{sid}/{a.id}" for sid in set_ids for a in by_id[sid].atoms}
        assert {a["id"] for a in c.source["atoms"]} == expected
        for a in c.source["atoms"]:
            assert a["set_id"] in set_ids and a["id"] == f"{a['set_id']}/{a['local_id']}"


def test_missing_actor(renderer: AskRenderer, grid_cases: list[Case]) -> None:
    actor_forms = _all_actor_forms(renderer)
    for c in grid_cases:
        text = c.input["ask"].lower()
        if c.strata["missing_actor"] == "yes":
            assert not c.source["actor_stated"]
            assert not [f for f in actor_forms if f in text], c.id
        else:
            for actor in c.source["actors"].values():
                assert actor.lower() in text


def test_tone_and_format(grid_cases: list[Case]) -> None:
    for c in grid_cases:
        text = c.input["ask"]
        if c.strata["tone"] == "urgent":
            assert "URGENT" in text or "ASAP" in text
        if c.strata["tone"] == "rambling":
            for aside in c.source["asides"]:
                assert aside in text
        else:
            assert not c.source["asides"]
        fmt = c.strata["source_format"]
        if fmt == "email":
            assert text.startswith("Subject: ")
        elif fmt == "meeting_notes":
            assert "Attendees:" in text and "\n- " in text
        else:
            assert not text.startswith("Subject:") and "Attendees:" not in text


def test_render_rejects_unknown_features(renderer: AskRenderer) -> None:
    with pytest.raises(ValueError):
        renderer.render({"tone": "angry"}, 1)
    with pytest.raises(ValueError):
        renderer.render({"colour": "blue"}, 1)


def test_reword_ask_changes_text_keeps_content(grid_cases: list[Case]) -> None:
    for c in grid_cases:
        new_input = reword_ask(c, seed=11)
        assert set(new_input) == set(c.input)
        new = new_input["ask"]
        assert new != c.input["ask"], c.id
        assert reword_ask(c, seed=11) == new_input  # deterministic
        low = new.lower()
        for atom in c.source["atoms"]:
            assert atom["phrase"].lower() in low, (c.id, atom["id"])
            if atom["mode"] != "hinted":
                for kw in atom["keywords"]:
                    assert kw in low
        for aside in c.source["asides"]:
            assert aside in new
        if c.strata["missing_actor"] == "no":
            for actor in c.source["actors"].values():
                assert actor.lower() in low
        if c.strata["source_format"] == "email":
            assert new.startswith("Subject: ")
        # the original case is not modified
        assert c.input["ask"] != new


def test_reword_ask_without_plan() -> None:
    case = Case(id="x", input={"ask": "hey team, please add CSV export. thanks!"}, provenance="real")
    new = reword_ask(case, seed=1)["ask"]
    assert new != case.input["ask"] and "please add CSV export" in new
    plain = Case(id="y", input={"ask": "Add CSV export."}, provenance="real")
    assert "Add CSV export." in reword_ask(plain, seed=1)["ask"]
