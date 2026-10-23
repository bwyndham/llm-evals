"""RUN-7: the agent replay runner.

For each sample it builds a `ToolSession` over the case's frozen corpus
(`case.fixtures["docs"]`), imports the agent entry point (`"module:function"`,
signature `run(case_input, tools, provider, version) -> AgentOutcome`, with an
optional keyword `max_turns`) and turns the outcome into a `Result`.

- An unrecorded call under `on_unrecorded_call="fail"` makes the sample a
  SYSTEM error with the reason, whatever the agent returned.
- Exceptions from the agent are classified with `classify_exception`
  (INFRA recorded; FatalRunError propagates and aborts the run).
- A case without `fixtures.docs` is a data problem: FatalRunError.
- Cost is summed per model call so Haiku's long-prompt tier applies per call.
"""

from __future__ import annotations

import importlib
import inspect
import time
from collections.abc import Callable
from typing import Any

from llm_evals.agents.tools import ON_UNRECORDED_CALL_VALUES, ToolSession
from llm_evals.registry import register_runner
from llm_evals.runners.errors import FatalRunError, classify_exception
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.runners.providers.tools import ToolProvider
from llm_evals.types import Case, ErrorKind, Result, Trace, VersionManifest

DEFAULT_ENTRY = "llm_evals.agents.research:run"


def resolve_entry(entry: str) -> Callable[..., Any]:
    module_name, sep, attr = entry.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"entry must be 'module:function', got {entry!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"entry {entry!r}: cannot import {module_name!r}: {exc}") from exc
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"entry {entry!r}: {module_name} has no callable {attr!r}")
    return fn


def _accepts_max_turns(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "max_turns" or p.kind is p.VAR_KEYWORD for p in params)


def _make_provider(tool_provider: ToolProvider | str | None) -> ToolProvider:
    if tool_provider is None or tool_provider == "anthropic":
        from llm_evals.runners.providers.anthropic_tools import AnthropicToolProvider

        return AnthropicToolProvider()
    if tool_provider == "mock":
        from llm_evals.runners.providers.mock_research import MockResearchModel

        return MockResearchModel()
    if isinstance(tool_provider, str):
        raise ValueError(f"unknown tool_provider {tool_provider!r} (use 'anthropic' or 'mock')")
    return tool_provider


@register_runner("agent_replay")
class AgentReplayRunner:
    """`tool_provider`: a ToolProvider, "anthropic" or None (the real API), or "mock"."""

    def __init__(
        self,
        tool_provider: ToolProvider | str | None = None,
        entry: str = DEFAULT_ENTRY,
        on_unrecorded_call: str = "fail",
        max_turns: int = 8,
    ) -> None:
        if on_unrecorded_call not in ON_UNRECORDED_CALL_VALUES:
            raise ValueError(
                f"on_unrecorded_call must be one of {ON_UNRECORDED_CALL_VALUES}, "
                f"not {on_unrecorded_call!r}"
            )
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        self.provider = _make_provider(tool_provider)
        self.entry = entry
        self._entry_fn = resolve_entry(entry)  # fail fast on a typo
        self._pass_max_turns = _accepts_max_turns(self._entry_fn)
        self.on_unrecorded_call = on_unrecorded_call
        self.max_turns = max_turns

    def run(self, case: Case, version: VersionManifest, sample_idx: int, run_id: str) -> Result:
        try:
            session = ToolSession(case, self.on_unrecorded_call)  # type: ignore[arg-type]
        except ValueError as exc:
            raise FatalRunError(str(exc)) from exc
        result = Result(
            run_id=run_id, case_id=case.id, version_id=version.id, sample_idx=sample_idx
        )
        kwargs = {"max_turns": self.max_turns} if self._pass_max_turns else {}

        start = time.perf_counter()
        try:
            outcome = self._entry_fn(case.input, session, self.provider, version, **kwargs)
        except FatalRunError:
            raise
        except Exception as exc:
            result.trace.latency_ms = (time.perf_counter() - start) * 1000
            result.trace.tool_calls = list(session.calls)
            kind = classify_exception(exc)  # raises FatalRunError for config problems
            request_id = getattr(exc, "request_id", None)
            if request_id:
                result.trace.request_ids.append(request_id)
            result.error_kind = kind
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = int(getattr(outcome, "input_tokens", 0))
        output_tokens = int(getattr(outcome, "output_tokens", 0))
        call_usage = list(getattr(outcome, "call_usage", None) or [])
        if call_usage:
            cost = sum(cost_usd(version.model, i, o) for i, o in call_usage)
        else:
            cost = cost_usd(version.model, input_tokens, output_tokens)
        tool_calls = getattr(outcome, "tool_calls", None)
        result.trace = Trace(
            model=version.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            tool_calls=list(tool_calls if tool_calls is not None else session.calls),
            request_ids=list(getattr(outcome, "request_ids", [])),
        )
        result.raw_output = getattr(outcome, "raw_text", None)
        kind = getattr(outcome, "error_kind", ErrorKind.NONE)
        error = getattr(outcome, "error", None)

        if session.failed and kind is not ErrorKind.SYSTEM:
            kind, error = ErrorKind.SYSTEM, session.failure_reason
        result.error_kind = kind
        result.error = error
        if kind is ErrorKind.NONE:
            output = getattr(outcome, "output", None)
            if output is None:
                result.error_kind = ErrorKind.SYSTEM
                result.error = "agent returned no output"
            else:
                result.output = output
        return result


__all__ = ["DEFAULT_ENTRY", "AgentReplayRunner", "resolve_entry"]
