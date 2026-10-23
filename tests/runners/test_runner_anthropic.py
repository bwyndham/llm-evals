from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from llm_evals.runners.errors import FatalRunError, InfraError, classify_exception
from llm_evals.runners.providers.anthropic import AnthropicProvider
from llm_evals.types import ErrorKind

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=REQUEST, headers={"request-id": "req_err"})
    return cls("boom", response=response, body={"error": {"type": "x", "message": "boom"}})


def fake_message(**overrides: Any) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text='{"a": '),
            SimpleNamespace(type="text", text="1}"),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        stop_reason="end_turn",
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


def provider_with(outcome: Any) -> tuple[AnthropicProvider, FakeMessages]:
    messages = FakeMessages(outcome)
    return AnthropicProvider(client=SimpleNamespace(messages=messages)), messages


def test_request_construction_with_system_and_effort() -> None:
    provider, messages = provider_with(fake_message())
    out = provider.complete(
        model="claude-haiku-5-5", system="sys", user="hi", params={"effort": "low"}
    )
    (kwargs,) = messages.kwargs
    assert kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 16000,
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "low"},
    }
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in kwargs
    assert out.text == '{"a": 1}'  # thinking block skipped
    assert (out.input_tokens, out.output_tokens) == (11, 7)
    assert out.request_id == "req_abc" and out.stop_reason == "end_turn"


def test_system_and_output_config_omitted_when_absent() -> None:
    provider, messages = provider_with(fake_message())
    provider.complete(model="claude-haiku-5-5", system=None, user="hi", params={"max_tokens": 500})
    (kwargs,) = messages.kwargs
    assert "system" not in kwargs and "output_config" not in kwargs
    assert kwargs["max_tokens"] == 500
    assert kwargs["messages"][-1]["role"] == "user"  # no assistant prefill


@pytest.mark.parametrize("param", ["temperature", "top_p", "top_k", "thinking"])
def test_rejected_params_are_fatal_before_any_call(param: str) -> None:
    provider, messages = provider_with(fake_message())
    with pytest.raises(FatalRunError, match=param):
        provider.complete(model="claude-sonnet-5-5", system=None, user="hi", params={param: 0})
    assert messages.kwargs == []


def test_refusal_stop_reason_is_returned() -> None:
    provider, _ = provider_with(fake_message(stop_reason="refusal", content=[]))
    out = provider.complete(model="claude-haiku-5-5", system=None, user="hi", params={})
    assert out.stop_reason == "refusal" and out.text == ""


def test_client_options_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)  # a local .env may set it
    seen: dict[str, Any] = {}

    def fake_ctor(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(messages=FakeMessages(fake_message()))

    monkeypatch.setattr(anthropic, "Anthropic", fake_ctor)
    provider = AnthropicProvider(max_retries=5, timeout=30.0)
    provider.complete(model="claude-haiku-5-5", system=None, user="hi", params={})
    assert seen == {"max_retries": 5, "timeout": 30.0}


INFRA_ERRORS = [
    anthropic.APIConnectionError(request=REQUEST),
    anthropic.APITimeoutError(request=REQUEST),
    status_error(anthropic.RateLimitError, 429),
    status_error(anthropic.InternalServerError, 500),
    status_error(anthropic.ServiceUnavailableError, 503),
    status_error(anthropic.OverloadedError, 529),
    status_error(anthropic.APIStatusError, 502),
    status_error(anthropic.ConflictError, 409),
]
FATAL_ERRORS = [
    status_error(anthropic.AuthenticationError, 401),
    status_error(anthropic.PermissionDeniedError, 403),
    status_error(anthropic.NotFoundError, 404),
    status_error(anthropic.BadRequestError, 400),
    status_error(anthropic.UnprocessableEntityError, 422),
]


@pytest.mark.parametrize("exc", INFRA_ERRORS, ids=lambda e: type(e).__name__)
def test_infra_classification(exc: Exception) -> None:
    assert classify_exception(exc) is ErrorKind.INFRA
    provider, _ = provider_with(exc)
    with pytest.raises(InfraError) as info:
        provider.complete(model="claude-haiku-5-5", system=None, user="hi", params={})
    assert info.value.__cause__ is exc


@pytest.mark.parametrize("exc", FATAL_ERRORS, ids=lambda e: type(e).__name__)
def test_fatal_classification(exc: Exception) -> None:
    with pytest.raises(FatalRunError):
        classify_exception(exc)
    provider, _ = provider_with(exc)
    with pytest.raises(FatalRunError):
        provider.complete(model="claude-haiku-5-5", system=None, user="hi", params={})


def test_request_too_large_is_system() -> None:
    assert classify_exception(status_error(anthropic.RequestTooLargeError, 413)) is ErrorKind.SYSTEM


def test_generic_exceptions() -> None:
    assert classify_exception(TimeoutError()) is ErrorKind.INFRA
    assert classify_exception(InfraError("x")) is ErrorKind.INFRA
    with pytest.raises(FatalRunError):
        classify_exception(FatalRunError("x"))
