"""Known-facts cases: real facts, contradicted in the corpus to test source-following."""

from pathlib import Path

from llm_evals.cases.corpus import answer_matches, text_mentions
from llm_evals.cases.known_facts import FACTS, build_cases
from llm_evals.cases.loader import case_set_hash, load_cases

SUITE = Path(__file__).resolve().parents[2] / "suites" / "research-agent"


def test_checked_in_file_matches_a_rebuild():
    checked_in = load_cases(SUITE / "data" / "known_facts_cases.jsonl")
    assert case_set_hash(checked_in) == case_set_hash(build_cases(seed=7))


def test_two_cases_per_fact():
    cases = build_cases()
    assert len(cases) == 2 * len(FACTS)
    assert {c.strata["perturbation"] for c in cases} == {"none", "change_fact"}
    assert all(c.strata["qtype"] == "known_fact" for c in cases)


def test_changed_corpus_states_only_the_changed_value():
    for case in build_cases():
        if case.strata["perturbation"] != "change_fact":
            continue
        text = " ".join(d["text"] for d in case.fixtures["docs"])
        assert text_mentions(text, case.expected["answer"])
        assert not text_mentions(text, case.expected["original_answer"])  # token-level, so "8" isn't found in "1889"
        assert not answer_matches(case.expected["original_answer"], case.expected)
        assert "deliberately states a false" in case.source["note"]


def test_true_variant_answers_match():
    for case in build_cases():
        assert answer_matches(case.expected["answer"], case.expected)
