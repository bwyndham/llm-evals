from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from llm_evals.agents.tools import tool_schemas
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.anthropic_tools import AnthropicToolProvider, block_to_dict

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
MESSAGES = [{"role": "user", "content": "Question: x?"}]


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=REQUEST, headers={"request-id": "req_err"})
    return cls("boom", response=response, body={"error": {"type": "x", "message": "boom"}})


def sdk_message(**overrides: Any) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=[
            ThinkingBlock(type="thinking", thinking="", signature="sig123"),
            TextBlock(type="text", text="Let me search."),
            ToolUseBlock(type="tool_use", id="toolu_1", name="search", input={"query": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=21, output_tokens=8),
        stop_reason="tool_use",
        _request_id="req_abc",
    )
    for k, v in overrides.items():
        setattr(msg, k, v)
    return msg


class FakeMessages:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def provider_with(outcome: Any) -> tuple[AnthropicToolProvider, FakeMessages]:
    messages = FakeMessages(outcome)
    return AnthropicToolProvider(client=SimpleNamespace(messages=messages)), messages


def test_request_construction() -> None:
    provider, fake = provider_with(sdk_message())
    tools = tool_schemas()
    provider.create(model="claude-haiku-5-5", system="sys", messages=MESSAGES, tools=tools,
                    params={"effort": "low"})
    (kwargs,) = fake.kwargs
    assert kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 16000,
        "system": "sys",
        "messages": MESSAGES,
        "tools": tools,
        "output_config": {"effort": "low"},
    }
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert banned not in kwargs


def test_optional_fields_omitted_and_max_tokens_passed() -> None:
    provider, fake = provider_with(sdk_message())
    provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={"max_tokens": 900})
    (kwargs,) = fake.kwargs
    assert "system" not in kwargs and "output_config" not in kwargs
    assert kwargs["max_tokens"] == 900


@pytest.mark.parametrize("param", ["temperature", "top_p", "top_k", "thinking"])
def test_rejected_params_abort(param) -> None:
    provider, fake = provider_with(sdk_message())
    with pytest.raises(FatalRunError):
        provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={param: 1})
    assert fake.kwargs == []


def test_response_conversion_keeps_raw_blocks() -> None:
    msg = sdk_message()
    provider, _ = provider_with(msg)
    out = provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={})
    assert out.stop_reason == "tool_use"
    assert (out.input_tokens, out.output_tokens, out.request_id) == (21, 8, "req_abc")
    assert out.content[0] == {"type": "thinking", "thinking": "", "signature": "sig123"}
    assert out.text == "Let me search."
    assert out.tool_uses == [
        {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"query": "x"}}
    ]
    assert all(a is b for a, b in zip(out.assistant_content, msg.content, strict=True))


def test_block_to_dict_handles_stubs_and_dicts() -> None:
    assert block_to_dict({"type": "text", "text": "a"}) == {"type": "text", "text": "a"}
    assert block_to_dict(SimpleNamespace(type="text", text="b")) == {"type": "text", "text": "b"}


@pytest.mark.parametrize(
    "exc",
    [status_error(anthropic.InternalServerError, 500), status_error(anthropic.RateLimitError, 429),
     anthropic.APITimeoutError(request=REQUEST)],
)
def test_infra_errors_become_infra_error(exc) -> None:
    provider, _ = provider_with(exc)
    with pytest.raises(InfraError):
        provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={})


@pytest.mark.parametrize(
    "exc",
    [status_error(anthropic.BadRequestError, 400), status_error(anthropic.AuthenticationError, 401),
     status_error(anthropic.NotFoundError, 404)],
)
def test_config_errors_are_fatal(exc) -> None:
    provider, _ = provider_with(exc)
    with pytest.raises(FatalRunError):
        provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={})


def test_request_too_large_is_reraised() -> None:
    provider, _ = provider_with(status_error(anthropic.RequestTooLargeError, 413))
    with pytest.raises(anthropic.RequestTooLargeError):
        provider.create(model="m", system=None, messages=MESSAGES, tools=[], params={})
