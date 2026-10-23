"""Shared plumbing for the spec-writing checks.

Every check returns a Score and never raises from `score`: a result with no
output fails with a "no output (...)" reason, and an unexpected exception
inside a check fails with a "check error: ..." reason.
"""

from __future__ import annotations

import json
from typing import Any

from llm_evals.types import Case, CheckContext, CheckKind, ErrorKind, Result, Score

SPEC_FIELDS: tuple[str, ...] = (
    "title",
    "description",
    "acceptance_criteria",
    "assumptions",
    "open_questions",
)


def no_output_reason(result: Result) -> str | None:
    """Why `result` has no usable output, or None if it has one."""
    if result.error_kind != ErrorKind.NONE:
        detail = result.error or "unspecified"
        return f"no output ({result.error_kind.value} error: {detail})"
    if result.output is None:
        return "no output (system error: output is None)"
    if not isinstance(result.output, dict):
        return f"no output (system error: output is {type(result.output).__name__}, not an object)"
    return None


def truncate(value: Any, limit: int = 2000) -> str:
    """A JSON (or str) rendering of `value`, cut to `limit` characters."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class BaseCheck:
    """Handles error results and unexpected exceptions; subclasses implement `_score`."""

    name: str
    kind: CheckKind
    critical: bool

    def __init__(self, *, critical: bool) -> None:
        if not isinstance(critical, bool):
            raise TypeError(f"{self.name}: critical must be a bool, got {critical!r}")
        self.critical = critical

    def applies(self, case: Case) -> bool:
        """Whether this check measures this case at all. Decided before the output is
        looked at, so an errored sample counts only against the checks it applies to."""
        return True

    def not_applicable_score(self, case: Case) -> Score:
        return self.not_applicable()

    def not_applicable(self, why: str = "not applicable") -> Score:
        return Score(check=self.name, kind=self.kind, passed=True, reason=why,
                     sub_scores=[{"applicable": False}])

    def score(self, case: Case, result: Result, ctx: CheckContext) -> Score:
        if not self.applies(case):
            return self.not_applicable_score(case)
        missing = no_output_reason(result)
        if missing is not None:
            return self.fail(missing)
        try:
            return self._score(case, result.output or {}, ctx)
        except Exception as exc:  # noqa: BLE001 - a check must never raise
            return self.fail(f"check error: {type(exc).__name__}: {exc}")

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        raise NotImplementedError

    def fail(self, reason: str, **fields: Any) -> Score:
        return Score(check=self.name, kind=self.kind, passed=False, reason=reason, **fields)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} critical={self.critical}>"
