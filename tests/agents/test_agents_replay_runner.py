from __future__ import annotations

import json
from typing import Any

import pytest

from agents_helpers import DOCS, GOOD_OUTPUT, make_case  # type: ignore[import-not-found]

from llm_evals import registry
from llm_evals.agents.research import AgentOutcome
from llm_evals.runners.agent_replay import AgentReplayRunner
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.anthropic_tools import AnthropicToolProvider
from llm_evals.runners.providers.mock_research import MockResearchModel
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.runners.providers.tools import FakeToolProvider, text_response, tool_use_response
from llm_evals.types import ErrorKind


def test_registered() -> None:
    assert registry.get("runner", "agent_replay") is AgentReplayRunner


def test_default_provider_is_anthropic_and_strings_are_accepted() -> None:
    assert isinstance(AgentReplayRunner().provider, AnthropicToolProvider)  # no API call made
    assert isinstance(AgentReplayRunner(tool_provider="mock").provider, MockResearchModel)
    with pytest.raises(ValueError, match="tool_provider"):
        AgentReplayRunner(tool_provider="openai")


@pytest.mark.parametrize(
    "kwargs",
    [{"entry": "nomodule"}, {"entry": "llm_evals.agents.research:nope"},
     {"entry": "no.such.module:run"}, {"on_unrecorded_call": "skip"}, {"max_turns": 0}],
)
def test_bad_config_fails_fast(kwargs) -> None:
    with pytest.raises(ValueError):
        AgentReplayRunner(tool_provider=FakeToolProvider(), **kwargs)


def test_end_to_end_with_mock(version) -> None:
    runner = AgentReplayRunner(tool_provider=MockResearchModel(noise=0))
    result = runner.run(make_case(), version, 2, "run-1")
    assert result.error_kind is ErrorKind.NONE, result.error
    assert (result.run_id, result.case_id, result.version_id, result.sample_idx) == (
        "run-1", "c1", "v-baseline", 2)
    assert result.output["answer"] == "1931"
    assert result.output["claims"][0]["citations"] == ["d1"]
    assert json.loads(result.raw_output) == result.output
    t = result.trace
    assert [c["name"] for c in t.tool_calls] == ["search", "fetch", "fetch", "fetch"]
    assert t.tool_calls[1] == {"name": "fetch", "input": {"doc_id": "d1"}, "doc_ids": ["d1"]}
    assert t.input_tokens > 0 and t.output_tokens > 0 and t.latency_ms >= 0
    assert len(t.request_ids) == 3 and t.cost_usd > 0


def test_remove_answer_case_is_not_found(version) -> None:
    runner = AgentReplayRunner(tool_provider=MockResearchModel(noise=0))
    result = runner.run(make_case(DOCS[1:], op="remove_answer_doc"), version, 0, "r")
    assert result.error_kind is ErrorKind.NONE
    assert result.output["status"] == "not_found"


def test_change_fact_with_memory(make_version, memory_path) -> None:
    changed = [dict(DOCS[0], text=DOCS[0]["text"].replace("1931", "1937"))] + DOCS[1:]
    case = make_case(changed, op="change_fact")
    runner = AgentReplayRunner(tool_provider=MockResearchModel(memory_path=memory_path, noise=0))
    assert runner.run(case, make_version("baseline"), 0, "r").output["answer"] == "1937"
    assert runner.run(case, make_version("ignore_corpus"), 0, "r").output["answer"] == "1931"


def test_unrecorded_call_under_fail_is_system(version) -> None:
    provider = FakeToolProvider(script=[tool_use_response([("fetch", {"doc_id": "ghost"})]),
                                        json.dumps(GOOD_OUTPUT)])
    result = AgentReplayRunner(tool_provider=provider).run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.SYSTEM
    assert "unrecorded tool call" in result.error and "ghost" in result.error
    assert result.output is None
    assert result.trace.tool_calls[0]["doc_ids"] == []


def test_unrecorded_call_under_error_result_continues(version) -> None:
    provider = FakeToolProvider(script=[tool_use_response([("fetch", {"doc_id": "ghost"})]),
                                        json.dumps(GOOD_OUTPUT)])
    runner = AgentReplayRunner(tool_provider=provider, on_unrecorded_call="error_result")
    result = runner.run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.NONE and result.output == GOOD_OUTPUT


def test_session_failure_wins_even_if_a_custom_entry_ignores_it(version, monkeypatch) -> None:
    import llm_evals.agents.research as research

    def sloppy(case_input, tools, provider, version):  # no max_turns parameter
        tools.call("fetch", {"doc_id": "ghost"})
        return AgentOutcome(output=GOOD_OUTPUT, raw_text="{}", tool_calls=list(tools.calls))

    monkeypatch.setattr(research, "sloppy_entry", sloppy, raising=False)
    runner = AgentReplayRunner(tool_provider=FakeToolProvider(),
                               entry="llm_evals.agents.research:sloppy_entry")
    result = runner.run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.SYSTEM and "ghost" in result.error
    assert result.output is None


def test_max_turns_is_passed_through(version) -> None:
    provider = FakeToolProvider(default=lambda call: tool_use_response([("search", {"query": "x"})]))
    result = AgentReplayRunner(tool_provider=provider, max_turns=2).run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.SYSTEM and "max_turns=2" in result.error
    assert len(provider.calls) == 2


def test_infra_error_is_infra_and_keeps_trace(version) -> None:
    provider = FakeToolProvider(script=[
        tool_use_response([("search", {"query": "bridge"})], input_tokens=10, output_tokens=5,
                          request_id="req_a"),
        InfraError("529 overloaded", request_id="req_b"),
    ])
    result = AgentReplayRunner(tool_provider=provider).run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.INFRA
    assert result.trace.request_ids == ["req_a", "req_b"]
    assert result.trace.input_tokens == 10 and len(result.trace.tool_calls) == 1


def test_exception_from_entry_is_classified(version, monkeypatch) -> None:
    import llm_evals.agents.research as research

    def boom(*args: Any, **kwargs: Any):
        raise TimeoutError("slow")

    monkeypatch.setattr(research, "boom_entry", boom, raising=False)
    runner = AgentReplayRunner(tool_provider=FakeToolProvider(),
                               entry="llm_evals.agents.research:boom_entry")
    result = runner.run(make_case(), version, 0, "r")
    assert result.error_kind is ErrorKind.INFRA and "slow" in result.error


def test_fatal_errors_propagate(version) -> None:
    runner = AgentReplayRunner(tool_provider=FakeToolProvider(script=[FatalRunError("bad key")]))
    with pytest.raises(FatalRunError):
        runner.run(make_case(), version, 0, "r")
    with pytest.raises(FatalRunError, match="fixtures.docs"):
        runner.run(make_case().model_copy(update={"fixtures": None}), version, 0, "r")


def test_cost_is_summed_per_call(version) -> None:
    provider = FakeToolProvider(script=[
        tool_use_response([("search", {"query": "bridge"})], input_tokens=90_000, output_tokens=100),
        text_response(json.dumps(GOOD_OUTPUT), input_tokens=110_000, output_tokens=200),
    ])
    result = AgentReplayRunner(tool_provider=provider).run(make_case(), version, 0, "r")
    expected = cost_usd("claude-haiku-5-5", 90_000, 100) + cost_usd("claude-haiku-5-5", 110_000, 200)
    assert result.trace.cost_usd == pytest.approx(expected)
    assert result.trace.input_tokens == 200_000
