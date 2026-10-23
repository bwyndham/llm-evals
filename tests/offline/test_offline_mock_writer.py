"""The offline mock spec writer: prompt-trait detection, output validity, behaviour, noise.

The mock is rule-based. These tests prove it is a usable stand-in for the
plumbing (valid JSON, valid Gherkin, prompt-dependent behaviour), not that it
writes good specs.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from llm_evals import registry
from llm_evals.cases.loader import load_cases
from llm_evals.cases.renderers.ask_renderer import reword_ask
from llm_evals.checks.gherkin import analyze_gherkin
from llm_evals.runners.prompt import parse_output, render_template
from llm_evals.runners.providers import MockProvider as ExportedMockProvider
from llm_evals.runners.providers.base import Provider
from llm_evals.runners.providers.mock import (
    INVENTED_SCENARIOS,
    MockProvider,
    PromptTraits,
    content_words,
    parse_ask,
)
from llm_evals.types import Result

SUITE = Path(__file__).resolve().parents[2] / "suites" / "spec-writing"
PROMPTS = {
    "baseline": SUITE / "prompts" / "spec_system.jinja",
    "no_split_bundles": SUITE / "prompts" / "seeded" / "no_split_bundles.jinja",
    "no_assumption_flags": SUITE / "prompts" / "seeded" / "no_assumption_flags.jinja",
}
USER = SUITE / "prompts" / "spec_user.jinja"


@pytest.fixture(scope="module")
def cases():
    return load_cases(SUITE / "data" / "cases.jsonl")


def _prompts(name: str, ask: str) -> tuple[str, str]:
    return render_template(str(PROMPTS[name]), {"ask": ask}), render_template(str(USER), {"ask": ask})


def _ticket(provider: MockProvider, name: str, ask: str) -> dict:
    system, user = _prompts(name, ask)
    response = provider.complete(model="claude-haiku-5-5", system=system, user=user, params={})
    return parse_output(response.text, "spec_ticket")


# --- Trait detection -----------------------------------------------------------


@pytest.mark.parametrize(
    "name, split, flag",
    [("baseline", True, True), ("no_split_bundles", False, True), ("no_assumption_flags", True, False)],
)
def test_detects_instructions_in_the_real_prompts(name, split, flag):
    traits = PromptTraits.detect(PROMPTS[name].read_text())
    assert (traits.split_bundles, traits.flag_assumptions) == (split, flag), traits.evidence


def test_detection_survives_small_edits():
    text = PROMPTS["baseline"].read_text()
    # Reflowed, re-cased and with one of the split phrases reworded: still the same traits.
    edited = re.sub(r"\s+", "  ", text).upper().replace("NEVER MERGE TWO DIFFERENT ASKS", "DO NOT COMBINE ASKS")
    traits = PromptTraits.detect(edited)
    assert traits.split_bundles and traits.flag_assumptions
    # Dropping a single flag sentence leaves enough other cues.
    assert PromptTraits.detect(text.replace("Write an assumption instead of quietly adding a requirement.", "")
                               ).flag_assumptions
    # An explicit opposite instruction wins over the remaining cues.
    assert not PromptTraits.detect(text + "\nKeep it to one scenario.").split_bundles
    assert not PromptTraits.detect(text + "\nFill the gaps yourself.").flag_assumptions
    assert PromptTraits.detect(None) == PromptTraits(split_bundles=False, flag_assumptions=False)


# --- Output validity -------------------------------------------------------------


def test_is_a_provider_and_exported():
    assert ExportedMockProvider is MockProvider
    assert isinstance(MockProvider(), Provider)


@pytest.mark.parametrize("name", list(PROMPTS))
def test_every_case_gives_schema_valid_json_and_gherkin(cases, name):
    provider = MockProvider(noise=0.0)
    fmt = registry.get("check", "format_rules")()
    for case in cases:
        ticket = _ticket(provider, name, case.input["ask"])  # raises on a schema violation
        report = analyze_gherkin(ticket["acceptance_criteria"])
        assert report.ok, (case.id, report.parse_error, report.problems)
        result = Result(run_id="r", case_id=case.id, version_id="v", sample_idx=0, output=ticket)
        assert fmt.score(case, result, None).passed, case.id


def test_response_metadata():
    provider = MockProvider()
    system, user = _prompts("baseline", "hey team,\nWe need to be able to export invoices as CSV.\nthanks!")
    first = provider.complete(model="m", system=system, user=user, params={})
    second = provider.complete(model="m", system=system, user=user, params={})
    assert first.request_id == "mock-1" and second.request_id == "mock-2"
    assert first.stop_reason == "end_turn"
    assert first.input_tokens == (len(system) + len(user)) // 4
    assert first.output_tokens == len(first.text) // 4


# --- Extraction ------------------------------------------------------------------


def test_extraction_matches_the_constructed_answer(cases):
    """On every rendered (and reworded) ask: one block per atom set, one requirement per
    stated atom, one hint per hinted atom, and the actor found exactly when stated."""
    for case in cases:
        src = case.source
        for ask in (case.input["ask"], reword_ask(case, 11)["ask"]):
            blocks = parse_ask(ask)
            assert len(blocks) == len(src["sets"]), (case.id, ask)
            stated = len(src["atoms"]) - len(src["hinted_atom_ids"])
            assert sum(len(b.requirements) for b in blocks) == stated, (case.id, ask)
            assert sum(len(b.hints) for b in blocks) == len(src["hinted_atom_ids"]), (case.id, ask)
            assert all((b.actor is not None) == src["actor_stated"] for b in blocks), (case.id, ask)


def test_requirement_words_are_kept_and_noise_is_dropped():
    ask = (
        "Subject: URGENT: Request: invoice search\n\nHi all,\n\nThis is extremely urgent and needs to go out ASAP.\n\n"
        "This comes from the office managers. We need the office managers to be able to search invoices by "
        "invoice number. Heads up, the parking garage is closed for repairs next week. FWIW, the current search "
        "is so slow that people have gone back to scrolling.\n\nThanks,\nGrace"
    )
    (block,) = parse_ask(ask)
    assert block.topic == "invoice search" and block.actor == "office managers"
    assert [r.phrase for r in block.requirements] == ["search invoices by invoice number"]
    assert block.hints == ["the current search is so slow that people have gone back to scrolling"]
    ticket = _ticket(MockProvider(noise=0.0), "baseline", ask)
    ac = ticket["acceptance_criteria"].lower()
    assert "search invoices by invoice number" in ac
    assert "parking" not in ac and "urgent" not in ac and "scrolling" not in ac
    assert any("scrolling" in a for a in ticket["assumptions"])


# --- Prompt-dependent behaviour ----------------------------------------------------

BUNDLED = (
    "hey team -\nWe need to be able to export the invoice list as a CSV file. Also, the export should include "
    "only invoices from the last 90 days.\nalso, totally separate thing (invoice branding): We need to be able to "
    "upload a logo to appear on every invoice.\nthanks!"
)


def test_baseline_splits_bundles_and_flags_the_missing_actor():
    ticket = _ticket(MockProvider(noise=0.0), "baseline", BUNDLED)
    report = analyze_gherkin(ticket["acceptance_criteria"])
    assert len(report.scenarios) == 3
    assert "logo" in ticket["acceptance_criteria"]
    assert ticket["open_questions"] and ticket["assumptions"]


def test_no_split_variant_keeps_only_the_first_topic_in_one_scenario():
    ticket = _ticket(MockProvider(noise=0.0), "no_split_bundles", BUNDLED)
    report = analyze_gherkin(ticket["acceptance_criteria"])
    assert len(report.scenarios) == 1
    assert "90 days" in ticket["acceptance_criteria"]
    text = " ".join(ticket[k] for k in ("title", "description", "acceptance_criteria")).lower()
    assert "logo" not in text and "branding" not in text


def test_no_flag_variant_invents_a_scenario_and_flags_nothing():
    ticket = _ticket(MockProvider(noise=0.0), "no_assumption_flags", BUNDLED)
    names = [s.name for s in analyze_gherkin(ticket["acceptance_criteria"]).scenarios]
    assert any(name in names for name, _ in INVENTED_SCENARIOS)
    assert ticket["assumptions"] == [] and ticket["open_questions"] == []


def test_invented_scenarios_share_nothing_with_any_atom():
    atom_sets = [json.loads(line) for line in (SUITE / "data" / "atoms.jsonl").read_text().splitlines() if line]
    keywords = {k.lower() for s in atom_sets for a in s["atoms"] for k in a["keywords"]}
    atom_words = set().union(*(content_words(a["text"]) for s in atom_sets for a in s["atoms"]))
    for name, steps in INVENTED_SCENARIOS:
        text = " ".join([name, *(re.sub(r"^\w+\s+", "", s) for s in steps)]).lower()
        assert not [k for k in keywords if k in text], name
        assert not content_words(text) & atom_words, (name, content_words(text) & atom_words)


# --- Sampling noise ------------------------------------------------------------------


def _n_scenarios(provider: MockProvider, ask: str) -> list[int]:
    out = []
    for _ in range(40):
        ticket = _ticket(provider, "baseline", ask)
        out.append(len(analyze_gherkin(ticket["acceptance_criteria"]).scenarios))
    return out


def test_noise_drops_a_requirement_reproducibly():
    noisy = _n_scenarios(MockProvider(noise=0.5), BUNDLED)
    assert set(noisy) == {2, 3}  # sometimes one of the three requirements is dropped
    assert _n_scenarios(MockProvider(noise=0.5), BUNDLED) == noisy  # same call order, same outputs
    assert set(_n_scenarios(MockProvider(noise=0.0), BUNDLED)) == {3}
    assert _n_scenarios(MockProvider(noise=0.5, seed=1), BUNDLED) != noisy


def test_noise_never_drops_the_only_requirement():
    ask = "hey team -\nWe need to be able to export the invoice list as a CSV file.\nthanks!"
    assert set(_n_scenarios(MockProvider(noise=1.0), ask)) == {1}


def test_call_counter_is_thread_safe():
    provider = MockProvider(noise=0.3)
    system, user = _prompts("baseline", BUNDLED)

    def work() -> None:
        for _ in range(50):
            provider.complete(model="m", system=system, user=user, params={})

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(provider._pair_calls.values()) == 200 and provider._requests == 200


def test_rejects_bad_noise():
    with pytest.raises(ValueError):
        MockProvider(noise=1.5)
