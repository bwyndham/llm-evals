from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_helpers import DOCS, QUESTION, VARIANTS, make_case, system_prompt  # type: ignore[import-not-found]

from llm_evals.agents.research import run
from llm_evals.agents.tools import ToolSession
from llm_evals.runners.providers.mock_research import (
    MockResearchModel,
    analyse_question,
    detect_traits,
    extract_question,
    instruction_text,
    plan_question,
)
from llm_evals.runners.providers.tools import ToolProvider

REPO = Path(__file__).resolve().parents[2]
SUITE_PROMPTS = REPO / "suites" / "research-agent" / "prompts"
WORLD_FACTS = REPO / "suites" / "research-agent" / "data" / "world_facts.jsonl"


def answer(model, version, docs=None, question: str = QUESTION):
    session = ToolSession(make_case(docs, question=question))
    outcome = run({"question": question}, session, model, version)
    assert outcome.error is None, outcome.error
    return outcome


def test_is_a_tool_provider() -> None:
    assert isinstance(MockResearchModel(), ToolProvider)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_trait_detection_on_fixture_prompts(variant) -> None:
    traits = detect_traits(system_prompt(**VARIANTS[variant]))
    assert traits.docs_only is (variant != "ignore_corpus")
    assert traits.cite is (variant != "drop_citations")
    assert traits.abstain is (variant != "no_abstain")
    assert traits.conflicts


def test_schema_blocks_dont_count_as_instructions() -> None:
    prompt = 'Find answers.\n```json\n{"status": "not_found", "claims": [{"citations": []}]}\n```\n'
    assert "not_found" not in instruction_text(prompt)
    traits = detect_traits(prompt)
    assert not traits.abstain and not traits.cite


def test_phrase_lists_can_be_overridden() -> None:
    assert detect_traits("Stick to the sources.", {"docs_only": ["stick to the sources"]}).docs_only


def test_question_analysis() -> None:
    assert extract_question("Context here.\nQuestion: Who runs it?\nThanks") == "Who runs it?"
    assert extract_question("Please answer: When was X built?") == "Please answer: When was X built?"
    q = analyse_question("How far is Drennick from Port Ilvane by road?")
    assert q.kind == "number" and q.subjects == ["drennick", "port ilvane"]
    assert analyse_question(QUESTION).kind == "year"
    q = analyse_question("Who is the CEO of Kestrin Lamp Company?")
    assert q.kind == "name" and q.subjects == ["kestrin lamp company"] and "ceo" in q.attribute


def test_three_turn_tool_loop(version) -> None:
    out = answer(MockResearchModel(noise=0), version)
    assert out.turns == 3
    assert [c["name"] for c in out.tool_calls] == ["search", "fetch", "fetch", "fetch"]
    assert out.tool_calls[0]["input"]["query"] == "year Halvorsen Bridge completed"
    assert out.tool_calls[1]["input"] == {"doc_id": "d1"}
    assert out.output == {
        "status": "answered",
        "answer": "1931",
        "claims": [{"text": "The Halvorsen Bridge was completed in 1931 after six years of work.",
                    "citations": ["d1"]}],
        "conflicts": [],
    }


def test_deterministic(version) -> None:
    a = answer(MockResearchModel(), version)
    b = answer(MockResearchModel(), version)
    assert a.output == b.output and a.tool_calls == b.tool_calls and a.request_ids == b.request_ids


def test_removed_answer_doc_abstains_unless_no_abstain(make_version) -> None:
    model = MockResearchModel(noise=0)
    docs = DOCS[1:]
    assert answer(model, make_version("baseline"), docs).output == {
        "status": "not_found", "answer": "", "claims": [], "conflicts": []}
    guessed = answer(model, make_version("no_abstain"), docs).output
    assert guessed["status"] == "answered" and guessed["answer"].isdigit()


def test_empty_search_abstains(version) -> None:
    out = answer(MockResearchModel(noise=0), version, docs=[DOCS[3]], question="Who is Zyx Qorp?")
    assert out.output["status"] == "not_found"
    assert [c["name"] for c in out.tool_calls] == ["search"]


def test_drop_citations(make_version) -> None:
    out = answer(MockResearchModel(noise=0), make_version("drop_citations")).output
    assert out["answer"] == "1931" and all(c["citations"] == [] for c in out["claims"])


def test_conflicting_docs(make_version) -> None:
    rival = {"id": "d5", "title": "Ostmark Gazette archive",
             "text": "The Halvorsen Bridge was completed in 1934, according to the city ledger."}
    out = answer(MockResearchModel(noise=0), make_version("baseline"), DOCS + [rival]).output
    assert out["status"] == "answered" and out["answer"] == ""
    assert out["conflicts"] and "1931" in out["conflicts"][0] and "1934" in out["conflicts"][0]
    assert sorted(c["citations"][0] for c in out["claims"]) == ["d1", "d5"]


def test_sibling_entity_is_not_a_match(version) -> None:
    sibling = [{"id": "s1", "title": "Upper Halvorsen Bridge",
                "text": "The Upper Halvorsen Bridge was completed in 1960."}]
    assert answer(MockResearchModel(noise=0), version, sibling).output["status"] == "not_found"


def test_change_fact_follows_corpus_unless_ignore_corpus(make_version, memory_path) -> None:
    changed = [dict(DOCS[0], text=DOCS[0]["text"].replace("1931", "1937"))] + DOCS[1:]
    model = MockResearchModel(memory_path=memory_path, noise=0)
    assert answer(model, make_version("baseline"), changed).output["answer"] == "1937"
    assert answer(model, make_version("ignore_corpus"), changed).output["answer"] == "1931"


def test_noise_one_degrades_the_answer(make_version) -> None:
    clean = answer(MockResearchModel(noise=0), make_version("baseline")).output
    noisy = answer(MockResearchModel(noise=1.0), make_version("baseline")).output
    assert clean != noisy


@pytest.mark.skipif(not WORLD_FACTS.exists(), reason="world facts not written yet")
def test_memory_lookup_on_suite_world_facts() -> None:
    model = MockResearchModel(memory_path=WORLD_FACTS)
    facts = [json.loads(line) for line in WORLD_FACTS.read_text().splitlines() if line.strip()]
    assert model.memory
    fact = next(f for f in facts if f["attribute"] == "population")
    q = analyse_question(f"What is the population of {fact['subject']}?")
    assert model._memory_value(q) == fact["value"]


def _suite_system_prompts() -> list[Path]:
    if not SUITE_PROMPTS.exists():
        return []
    return sorted(p for p in SUITE_PROMPTS.rglob("*.jinja") if "user" not in p.stem)


@pytest.mark.skipif(not _suite_system_prompts(), reason="suite prompts not written yet")
@pytest.mark.parametrize("path", _suite_system_prompts(), ids=lambda p: p.stem)
def test_trait_detection_on_suite_prompts(path: Path) -> None:
    traits = detect_traits(path.read_text())
    name = path.stem
    assert traits.docs_only is ("ignore_corpus" not in name)
    assert traits.cite is ("drop_citations" not in name)
    assert traits.abstain is ("no_abstain" not in name)


TOWNS = [
    {"id": "t1", "title": "Ashby", "text": "Ashby had a population of 4,770 at the last census. "
     "Ashby was granted its charter in 1788."},
    {"id": "t2", "title": "North Ashby", "text": "North Ashby had a population of 90,000 at the last census."},
    {"id": "t3", "title": "Brill", "text": "Brill had a population of 30,150 at the last census."},
    {"id": "t4", "title": "Kell Province", "text": "The capital of Kell Province is Brill. "
     "Kell Province covers an area of 960 km²."},
]


def test_plans() -> None:
    assert plan_question("Which had the larger population, Ashby or Brill?").kind == "comparison"
    p = plan_question("What was the population of the capital of Kell Province?")
    assert p.kind == "two_hop" and p.span == "the capital of Kell Province"
    assert plan_question("What is the capital of Kell Province?").kind == "lookup"
    assert plan_question("Who is the mayor of Ashby?").kind == "lookup"


def test_comparison(version) -> None:
    model = MockResearchModel(noise=0)
    q = "Which had the larger population at the last census, Ashby or Brill?"
    out = answer(model, version, TOWNS, question=q)
    assert out.output["answer"] == "Brill"  # North Ashby's 90,000 is a look-alike, not Ashby
    assert {c["citations"][0] for c in out.output["claims"]} == {"t1", "t3"}
    assert out.tool_calls[0]["name"] == "search" and out.tool_calls[1]["name"] == "search"
    q = "Which had the smaller population at the last census, the Brill or Ashby?"
    assert answer(model, version, TOWNS, question=q).output["answer"] == "Ashby"


def test_two_hop_searches_again_for_the_bridge_entity(version) -> None:
    q = "What was the population of the capital of Kell Province at the last census?"
    out = answer(MockResearchModel(noise=0), version, TOWNS, question=q)
    assert out.output["answer"] == "30,150"
    assert {c["citations"][0] for c in out.output["claims"]} == {"t4", "t3"}
    assert [c["name"] for c in out.tool_calls].count("search") == 2


def test_value_with_unit(version) -> None:
    out = answer(MockResearchModel(noise=0), version, TOWNS, question="What is the area of Kell Province?")
    assert out.output["answer"] == "960 km²"


SUITE_CASES = REPO / "suites" / "research-agent" / "data" / "cases.jsonl"
SUITE_VERSIONS = REPO / "suites" / "research-agent" / "versions"


def _suite_outcomes(version_file: str):
    from llm_evals.runners.agent_replay import AgentReplayRunner
    from llm_evals.runners.manifest import load_manifest
    from llm_evals.types import Case

    suite = SUITE_CASES.parents[1]
    v = load_manifest(SUITE_VERSIONS / version_file, suite)
    runner = AgentReplayRunner(tool_provider=MockResearchModel(memory_path=WORLD_FACTS, noise=0))
    cases = [Case.model_validate_json(x) for x in SUITE_CASES.read_text().splitlines() if x.strip()]
    return [(c, runner.run(c, v, 0, "r")) for c in cases]


def _behaves(case, result) -> bool:
    e, o = case.expected, result.output
    if o is None:
        return False
    if e["behavior"] == "abstain":
        return o["status"] == "not_found"
    if e["behavior"] == "flag_conflict":
        return bool(o["conflicts"]) and o["answer"] == ""
    norm = lambda s: (s or "").lower().replace(",", "").strip()  # noqa: E731
    return norm(o["answer"]) in {norm(x) for x in [e["answer"], *e["aliases"]]}


needs_suite = pytest.mark.skipif(
    not (SUITE_CASES.exists() and (SUITE_VERSIONS / "baseline.yaml").exists()),
    reason="research-agent suite not written yet",
)


@needs_suite
def test_suite_baseline_mostly_behaves() -> None:
    outcomes = _suite_outcomes("baseline.yaml")
    good = sum(_behaves(c, r) for c, r in outcomes)
    assert good >= 0.9 * len(outcomes)
    assert all(r.error_kind.value == "none" for _, r in outcomes)


@needs_suite
def test_suite_seeded_variants_show_their_failure() -> None:
    by_op = lambda outs, op: [(c, r) for c, r in outs if c.source["perturbation"]["op"] == op]  # noqa: E731
    ignore = _suite_outcomes("seeded_ignore_corpus.yaml")
    changed = by_op(ignore, "change_fact")
    assert changed and sum(_behaves(c, r) for c, r in changed) <= len(changed) // 2
    no_abstain = _suite_outcomes("seeded_no_abstain.yaml")
    removed = by_op(no_abstain, "remove_answer_doc")
    assert removed and all(r.output["status"] == "answered" for _, r in removed)
    dropped = _suite_outcomes("seeded_drop_citations.yaml")
    assert all(not cl["citations"] for _, r in dropped for cl in r.output["claims"])
