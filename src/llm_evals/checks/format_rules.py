"""`format_rules`: the ticket JSON has the fields and shapes the contract asks for.

Violations (each listed in the reason and as a sub-score):
- a required field is missing, None, or a blank string (an empty list is
  allowed: a ticket may have no assumptions or no open questions);
- title, description or acceptance_criteria is not a string;
- the title is longer than `title_max_chars`, ends with a period, or contains
  a line break;
- assumptions or open_questions (when present) is not a list of non-empty
  strings;
- open_questions has more than `max_open_questions` items (when set).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_evals.checks._base import SPEC_FIELDS, BaseCheck
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score

_STRING_FIELDS = ("title", "description", "acceptance_criteria")
_LIST_FIELDS = ("assumptions", "open_questions")


class FormatRules(BaseCheck):
    name = "format_rules"
    kind: CheckKind = "deterministic"

    def __init__(
        self,
        *,
        required_fields: Sequence[str],
        title_max_chars: int,
        max_open_questions: int | None,
        critical: bool,
    ) -> None:
        super().__init__(critical=critical)
        if isinstance(required_fields, str) or not all(isinstance(f, str) for f in required_fields):
            raise TypeError(f"format_rules: required_fields must be a list of strings, got {required_fields!r}")
        if not isinstance(title_max_chars, int) or isinstance(title_max_chars, bool) or title_max_chars < 1:
            raise TypeError(f"format_rules: title_max_chars must be a positive int, got {title_max_chars!r}")
        if max_open_questions is not None and (
            not isinstance(max_open_questions, int) or isinstance(max_open_questions, bool) or max_open_questions < 0
        ):
            raise TypeError(
                f"format_rules: max_open_questions must be a non-negative int or None, got {max_open_questions!r}"
            )
        self.required_fields = list(required_fields)
        self.title_max_chars = title_max_chars
        self.max_open_questions = max_open_questions

    def violations(self, output: dict[str, Any]) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []

        def add(field: str, rule: str, detail: str) -> None:
            found.append({"field": field, "rule": rule, "detail": detail})

        for name in self.required_fields:
            value = output.get(name)
            if name not in output or value is None:
                add(name, "required", f"{name} is missing")
            elif isinstance(value, str) and not value.strip():
                add(name, "required", f"{name} is empty")

        for name in _STRING_FIELDS:
            value = output.get(name)
            if value is not None and not isinstance(value, str):
                add(name, "type", f"{name} must be a string, got {type(value).__name__}")

        title = output.get("title")
        if isinstance(title, str) and title.strip():
            if len(title) > self.title_max_chars:
                add("title", "length", f"title is {len(title)} chars (max {self.title_max_chars})")
            if "\n" in title or "\r" in title:
                add("title", "line_break", "title contains a line break")
            if title.rstrip().endswith("."):
                add("title", "trailing_period", "title ends with a period")

        for name in _LIST_FIELDS:
            if name not in output or output[name] is None:
                continue  # reported above if required
            value = output[name]
            if not isinstance(value, list):
                add(name, "type", f"{name} must be a list of strings, got {type(value).__name__}")
                continue
            bad = [i for i, item in enumerate(value) if not isinstance(item, str) or not item.strip()]
            if bad:
                add(name, "items", f"{name} items {bad} are not non-empty strings")

        questions = output.get("open_questions")
        limit = self.max_open_questions
        if limit is not None and isinstance(questions, list) and len(questions) > limit:
            add("open_questions", "count", f"open_questions has {len(questions)} items (max {limit})")
        return found

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        found = self.violations(output)
        if found:
            return self.fail(
                f"{len(found)} format violation(s): " + "; ".join(v["detail"] for v in found),
                sub_scores=found,
            )
        return Score(check=self.name, kind=self.kind, passed=True, reason="all format rules pass")


@register_check("format_rules")
def format_rules(
    *,
    required_fields: Sequence[str] = SPEC_FIELDS,
    title_max_chars: int = 80,
    max_open_questions: int | None = None,
    critical: bool = False,
) -> FormatRules:
    return FormatRules(
        required_fields=required_fields,
        title_max_chars=title_max_chars,
        max_open_questions=max_open_questions,
        critical=critical,
    )
