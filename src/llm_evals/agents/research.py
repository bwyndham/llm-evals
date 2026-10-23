"""The research agent: a manual tool-use loop over `search` and `fetch`.

Each iteration sends the conversation, then by `stop_reason`:

- `tool_use`: append the assistant's full content unchanged, run every
  tool_use block, and send all their tool_result blocks back in ONE user
  message (a failed tool gives a result with `is_error: true`).
- `pause_turn`: append the assistant content and send again.
- `end_turn`: parse the final text as the output JSON (fences stripped) and
  validate its shape; invalid output is a SYSTEM error.
- `max_tokens`, `refusal` or anything else: SYSTEM error.

More than `max_turns` model calls without a final answer is a SYSTEM error.
Provider exceptions are classified with `classify_exception` (INFRA is
recorded, FatalRunError propagates). Tokens, request ids and tool calls made
before an error are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import jinja2
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_evals.agents.tools import ToolSession
from llm_evals.runners.errors import FatalRunError, classify_exception
from llm_evals.runners.prompt import render_template, strip_json_fence
from llm_evals.runners.providers.tools import ToolProvider, ToolResponse
from llm_evals.types import ErrorKind, VersionManifest

DEFAULT_MAX_TURNS = 8


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    citations: list[str] = []


class ResearchOutput(BaseModel):
    """The agent output contract (docs/contracts.md, "Research agent")."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["answered", "not_found"]
    answer: str
    claims: list[Claim]
    conflicts: list[str] = []


@dataclass
class AgentOutcome:
    output: dict[str, Any] | None = None
    raw_text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    request_ids: list[str] = field(default_factory=list)
    error_kind: ErrorKind = ErrorKind.NONE
    error: str | None = None
    turns: int = 0  # model calls made
    # (input, output) tokens per model call, so cost can use per-call price tiers.
    call_usage: list[tuple[int, int]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)  # final transcript


def parse_research_output(text: str) -> dict[str, Any]:
    """Parse and validate the final answer. Raises ValueError with a clear message."""
    body = strip_json_fence(text)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"output JSON must be an object, got {type(data).__name__}")
    try:
        return ResearchOutput.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(f"output does not match the research output shape: {exc}") from exc


def render_prompts(version: VersionManifest, case_input: dict[str, Any]) -> tuple[str | None, str]:
    templates = version.prompt_templates
    if "user" not in templates:
        raise FatalRunError(f"version {version.name!r} has no 'user' prompt template")
    unknown = set(templates) - {"system", "user"}
    if unknown:
        raise FatalRunError(
            f"version {version.name!r}: the research agent supports roles system and user, "
            f"not {sorted(unknown)}"
        )
    try:
        system = render_template(templates["system"], case_input) if "system" in templates else None
        user = render_template(templates["user"], case_input)
    except jinja2.UndefinedError as exc:
        raise FatalRunError(
            f"template for version {version.name!r} needs a variable the case input lacks: {exc}"
        ) from exc
    return system, user


def run(
    case_input: dict[str, Any],
    tools: ToolSession,
    provider: ToolProvider,
    version: VersionManifest,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AgentOutcome:
    system, user = render_prompts(version, case_input)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    schemas = tools.tool_schemas()
    outcome = AgentOutcome(messages=messages)

    def finish(kind: ErrorKind = ErrorKind.NONE, error: str | None = None) -> AgentOutcome:
        outcome.error_kind = kind
        outcome.error = error
        outcome.tool_calls = list(tools.calls)
        if kind is not ErrorKind.NONE:
            outcome.output = None
        return outcome

    for _ in range(max_turns):
        try:
            response: ToolResponse = provider.create(
                model=version.model,
                system=system,
                messages=messages,
                tools=schemas,
                params=dict(version.params),
            )
        except FatalRunError:
            raise
        except Exception as exc:
            kind = classify_exception(exc)  # raises FatalRunError for config problems
            request_id = getattr(exc, "request_id", None)
            if request_id:
                outcome.request_ids.append(request_id)
            return finish(kind, f"{type(exc).__name__}: {exc}")

        outcome.turns += 1
        outcome.input_tokens += response.input_tokens
        outcome.output_tokens += response.output_tokens
        outcome.call_usage.append((response.input_tokens, response.output_tokens))
        if response.request_id:
            outcome.request_ids.append(response.request_id)
        stop = response.stop_reason

        if stop == "tool_use":
            tool_uses = response.tool_uses
            if not tool_uses:
                outcome.raw_text = response.text or None
                return finish(ErrorKind.SYSTEM, "stop_reason=tool_use but no tool_use blocks")
            messages.append({"role": "assistant", "content": response.assistant_content})
            results: list[dict[str, Any]] = []
            for block in tool_uses:
                result = tools.call(block.get("name", ""), block.get("input"))
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.content,
                }
                if result.is_error:
                    item["is_error"] = True
                results.append(item)
            messages.append({"role": "user", "content": results})
            if tools.failed:
                return finish(ErrorKind.SYSTEM, tools.failure_reason)
            continue

        if stop == "pause_turn":
            messages.append({"role": "assistant", "content": response.assistant_content})
            continue

        outcome.raw_text = response.text
        if stop == "end_turn":
            messages.append({"role": "assistant", "content": response.assistant_content})
            if not response.text.strip():
                return finish(ErrorKind.SYSTEM, "final response has no text")
            try:
                outcome.output = parse_research_output(response.text)
            except ValueError as exc:
                return finish(ErrorKind.SYSTEM, str(exc))
            return finish()
        return finish(ErrorKind.SYSTEM, f"stop_reason={stop}")

    return finish(ErrorKind.SYSTEM, f"no final answer within max_turns={max_turns}")


__all__ = [
    "DEFAULT_MAX_TURNS",
    "AgentOutcome",
    "ResearchOutput",
    "parse_research_output",
    "render_prompts",
    "run",
]
