"""The tool-use provider protocol the research agent's loop calls, plus a scripted fake.

`ToolResponse.content` holds the response's blocks as plain dicts the loop can
read: `{"type": "text", "text"}`, `{"type": "tool_use", "id", "name", "input"}`,
and anything else (thinking, redacted_thinking, ...) passed through opaquely.
`raw_content` is what the loop appends back as the assistant turn: for the real
API, the SDK's own block objects, unchanged (thinking blocks must go back
exactly as received). When it's None, `content` is appended.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResponse:
    content: list[dict[str, Any]]
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None
    raw_content: list[Any] | None = None

    @property
    def assistant_content(self) -> list[Any]:
        """The blocks to append back as the assistant turn, unchanged."""
        return self.raw_content if self.raw_content is not None else self.content

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


@runtime_checkable
class ToolProvider(Protocol):
    def create(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> ToolResponse: ...


def tool_use_id(*parts: Any) -> str:
    """A stable tool_use id derived from its parts (for fakes and mocks)."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return "toolu_" + digest[:20]


def text_response(text: str, stop_reason: str = "end_turn", **kwargs: Any) -> ToolResponse:
    return ToolResponse(content=[{"type": "text", "text": text}], stop_reason=stop_reason, **kwargs)


def tool_use_response(
    calls: list[tuple[str, dict[str, Any]]], *, text: str | None = None, **kwargs: Any
) -> ToolResponse:
    """A `stop_reason="tool_use"` response with one tool_use block per (name, input)."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for i, (name, tool_input) in enumerate(calls):
        blocks.append(
            {"type": "tool_use", "id": tool_use_id(name, tool_input, i), "name": name,
             "input": tool_input}
        )
    return ToolResponse(content=blocks, stop_reason="tool_use", **kwargs)


ToolScriptItem = ToolResponse | BaseException | Callable[[dict[str, Any]], Any]


@dataclass
class FakeToolProvider:
    """Returns scripted `ToolResponse`s in order, for tests.

    - A script item is a `ToolResponse` (returned), an exception (raised), a
      `str` (a final end_turn text response) or a callable that receives the
      call kwargs and returns one of those.
    - When the script runs out, `default` is used; with no default a call
      raises `AssertionError`.
    - `calls` records each call's kwargs. `messages` is a shallow copy of the
      list as it was at call time, so the block objects inside are the very
      ones the loop sent (identity checks work). Thread-safe.
    """

    script: list[ToolScriptItem | str] = field(default_factory=list)
    default: ToolScriptItem | str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> ToolResponse:
        call = {
            "model": model,
            "system": system,
            "messages": list(messages),
            "tools": tools,
            "params": dict(params),
        }
        with self._lock:
            self.calls.append(call)
            if self.script:
                item: Any = self.script.pop(0)
            elif self.default is not None:
                item = self.default
            else:
                raise AssertionError("FakeToolProvider script exhausted")
        if callable(item) and not isinstance(item, BaseException):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return text_response(item)
        if isinstance(item, ToolResponse):
            return item
        raise TypeError(f"bad FakeToolProvider script item: {item!r}")


__all__ = [
    "FakeToolProvider",
    "ToolProvider",
    "ToolResponse",
    "text_response",
    "tool_use_id",
    "tool_use_response",
]
