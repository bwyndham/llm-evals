from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from agents_helpers import GOOD_OUTPUT, QUESTION, make_case  # type: ignore[import-not-found]

from llm_evals.agents.research import AgentOutcome, parse_research_output, run
from llm_evals.agents.tools import ToolSession
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.tools import (
    FakeToolProvider,
    ToolResponse,
    text_response,
    tool_use_response,
)
from llm_evals.types import ErrorKind

FINAL = json.dumps(GOOD_OUTPUT)


def search(query: str = "Halvorsen Bridge completed", **kw) -> ToolResponse:
    return tool_use_response([("search", {"query": query})], input_tokens=100, output_tokens=10,
                             request_id="req_1", **kw)


def final(text: str = FINAL, **kw) -> ToolResponse:
    return text_response(text, input_tokens=300, output_tokens=50, request_id="req_2", **kw)


def go(script, version, *, on_unrecorded: str = "fail", **kw) -> tuple[AgentOutcome, FakeToolProvider, ToolSession]:
    provider = FakeToolProvider(script=list(script))
    session = ToolSession(make_case(), on_unrecorded_call=on_unrecorded)  # type: ignore[arg-type]
    outcome = run({"question": QUESTION}, session, provider, version, **kw)
    return outcome, provider, session


def test_tool_use_then_final_answer(version) -> None:
    outcome, provider, _ = go([search(), final()], version)
    assert outcome.error_kind is ErrorKind.NONE, outcome.error
    assert outcome.output == GOOD_OUTPUT
    assert outcome.raw_text == FINAL
    assert (outcome.input_tokens, outcome.output_tokens) == (400, 60)
    assert outcome.call_usage == [(100, 10), (300, 50)]
    assert outcome.request_ids == ["req_1", "req_2"]
    assert outcome.turns == 2
    assert [c["name"] for c in outcome.tool_calls] == ["search"]
    assert outcome.tool_calls[0]["doc_ids"][0] == "d1"

    first, second = provider.calls
    assert first["system"].startswith("You are a research assistant")
    assert first["messages"] == [{"role": "user", "content": f"Question: {QUESTION}\n"}]
    assert [t["name"] for t in first["tools"]] == ["search", "fetch"]
    assert first["model"] == "claude-haiku-5-5"
    user, assistant, results = second["messages"]
    assert assistant["role"] == "assistant" and results["role"] == "user"
    (tr,) = results["content"]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == assistant["content"][0]["id"]
    assert "is_error" not in tr
    assert json.loads(tr["content"])[0]["doc_id"] == "d1"


def test_assistant_content_is_appended_unchanged(version) -> None:
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig-opaque")
    tool_block = SimpleNamespace(type="tool_use", id="toolu_x", name="search",
                                 input={"query": "bridge"})
    raw = [thinking, tool_block]
    resp = ToolResponse(
        content=[{"type": "thinking", "thinking": "", "signature": "sig-opaque"},
                 {"type": "tool_use", "id": "toolu_x", "name": "search", "input": {"query": "bridge"}}],
        stop_reason="tool_use",
        raw_content=raw,
    )
    outcome, provider, _ = go([resp, final()], version)
    assert outcome.error_kind is ErrorKind.NONE
    sent = provider.calls[1]["messages"][1]["content"]
    assert sent is raw and sent[0] is thinking and sent[1] is tool_block


def test_parallel_tool_uses_return_all_results_in_one_user_message(version) -> None:
    fetches = tool_use_response([("fetch", {"doc_id": "d1"}), ("fetch", {"doc_id": "d2"}),
                                 ("search", {"query": "Vell"})])
    outcome, provider, _ = go([fetches, final()], version)
    assert outcome.error_kind is ErrorKind.NONE
    msgs = provider.calls[1]["messages"]
    assert len(msgs) == 3
    results = msgs[2]["content"]
    assert [r["tool_use_id"] for r in results] == [b["id"] for b in fetches.tool_uses]
    assert [json.loads(r["content"]).get("doc_id") for r in results[:2]] == ["d1", "d2"]
    assert [c["name"] for c in outcome.tool_calls] == ["fetch", "fetch", "search"]


def test_failed_tool_gives_is_error_result_and_loop_continues(version) -> None:
    bad = tool_use_response([("fetch", {"doc_id": "missing"}), ("fetch", {"doc_id": "d1"})])
    outcome, provider, session = go([bad, final()], version, on_unrecorded="error_result")
    assert outcome.error_kind is ErrorKind.NONE
    results = provider.calls[1]["messages"][2]["content"]
    assert results[0]["is_error"] is True and "missing" in results[0]["content"]
    assert "is_error" not in results[1]
    assert outcome.tool_calls[0]["doc_ids"] == [] and outcome.tool_calls[1]["doc_ids"] == ["d1"]


def test_unrecorded_call_under_fail_stops_the_loop(version) -> None:
    bad = tool_use_response([("fetch", {"doc_id": "missing"})])
    outcome, provider, _ = go([bad, final()], version, on_unrecorded="fail")
    assert outcome.error_kind is ErrorKind.SYSTEM
    assert "unrecorded tool call" in (outcome.error or "") and "missing" in outcome.error
    assert len(provider.calls) == 1 and outcome.output is None


@pytest.mark.parametrize(
    "text",
    ["not json at all", "[1, 2]", json.dumps({**GOOD_OUTPUT, "status": "maybe"}),
     json.dumps({"status": "answered", "answer": "x"}),
     json.dumps({**GOOD_OUTPUT, "claims": [{"citations": ["d1"]}]})],
)
def test_invalid_final_output_is_system(version, text) -> None:
    outcome, _, _ = go([search(), final(text)], version)
    assert outcome.error_kind is ErrorKind.SYSTEM
    assert outcome.output is None and outcome.raw_text == text


def test_fenced_json_and_defaults_are_accepted() -> None:
    body = {"status": "not_found", "answer": "", "claims": [{"text": "nothing"}]}
    out = parse_research_output("```json\n" + json.dumps(body) + "\n```")
    assert out == {"status": "not_found", "answer": "", "claims": [{"text": "nothing", "citations": []}],
                   "conflicts": []}


def test_exceeding_max_turns_is_system(version) -> None:
    provider = FakeToolProvider(default=lambda call: search())
    session = ToolSession(make_case())
    outcome = run({"question": QUESTION}, session, provider, version, max_turns=3)
    assert outcome.error_kind is ErrorKind.SYSTEM and "max_turns=3" in (outcome.error or "")
    assert len(provider.calls) == 3 and outcome.turns == 3
    assert len(outcome.tool_calls) == 3


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", "stop_sequence"])
def test_bad_stop_reasons_are_system(version, stop) -> None:
    outcome, _, _ = go([search(), final(FINAL, stop_reason=stop)], version)
    assert outcome.error_kind is ErrorKind.SYSTEM and outcome.error == f"stop_reason={stop}"
    assert outcome.output is None
    assert outcome.request_ids == ["req_1", "req_2"]


def test_tool_use_stop_without_blocks_is_system(version) -> None:
    empty = ToolResponse(content=[{"type": "text", "text": "hmm"}], stop_reason="tool_use")
    outcome, _, _ = go([empty], version)
    assert outcome.error_kind is ErrorKind.SYSTEM


def test_pause_turn_resends_with_assistant_content(version) -> None:
    paused = ToolResponse(content=[{"type": "text", "text": "thinking..."}], stop_reason="pause_turn")
    outcome, provider, _ = go([paused, final()], version)
    assert outcome.error_kind is ErrorKind.NONE
    assert provider.calls[1]["messages"][-1] == {"role": "assistant", "content": paused.content}


def test_infra_error_keeps_partial_trace(version) -> None:
    outcome, _, _ = go([search(), InfraError("overloaded", request_id="req_err")], version)
    assert outcome.error_kind is ErrorKind.INFRA
    assert outcome.request_ids == ["req_1", "req_err"]
    assert outcome.input_tokens == 100 and len(outcome.tool_calls) == 1


def test_request_too_large_is_system(version) -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.RequestTooLargeError(
        "too big", response=httpx2.Response(413, request=req), body=None
    )
    outcome, _, _ = go([exc], version)
    assert outcome.error_kind is ErrorKind.SYSTEM


def test_config_errors_propagate(version, make_version) -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    bad = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body=None)
    with pytest.raises(FatalRunError):
        go([bad], version)
    with pytest.raises(FatalRunError, match="needs a variable"):
        run({"q": "x"}, ToolSession(make_case()), FakeToolProvider(), version)


def test_params_are_passed_to_the_provider(make_version) -> None:
    v = make_version(max_tokens=2000, effort="low")
    _, provider, _ = go([final()], v)
    assert provider.calls[0]["params"] == {"max_tokens": 2000, "effort": "low"}
