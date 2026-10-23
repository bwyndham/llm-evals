"""The provider protocol the prompt runner calls, plus a scripted fake."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None
    stop_reason: str | None = None


@runtime_checkable
class Provider(Protocol):
    def complete(
        self, *, model: str, system: str | None, user: str, params: dict[str, Any]
    ) -> ProviderResponse: ...


# A script item is a response, an exception to raise, or a callable that
# receives the call kwargs and returns either.
ScriptItem = ProviderResponse | BaseException | Callable[[dict[str, Any]], Any]


@dataclass
class FakeProvider:
    """Returns scripted responses in order, for tests and dry runs.

    - `script`: consumed one item per call. A `ProviderResponse` is returned,
      an exception is raised, a callable is called with the call kwargs and its
      result is handled the same way (a `str` becomes a response).
    - When the script runs out, `default` is used; with no default, a call
      raises `AssertionError`.
    - `calls` records the kwargs of every call. Thread-safe.
    """

    script: list[ScriptItem] = field(default_factory=list)
    default: ScriptItem | str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs: Any) -> FakeProvider:
        return cls(script=[ProviderResponse(text=t, input_tokens=10, output_tokens=20) for t in texts], **kwargs)

    def complete(
        self, *, model: str, system: str | None, user: str, params: dict[str, Any]
    ) -> ProviderResponse:
        call = {"model": model, "system": system, "user": user, "params": dict(params)}
        with self._lock:
            self.calls.append(call)
            if self.script:
                item: Any = self.script.pop(0)
            elif self.default is not None:
                item = self.default
            else:
                raise AssertionError("FakeProvider script exhausted")
        if callable(item) and not isinstance(item, BaseException):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return ProviderResponse(text=item)
        if isinstance(item, ProviderResponse):
            return item
        raise TypeError(f"bad FakeProvider script item: {item!r}")
