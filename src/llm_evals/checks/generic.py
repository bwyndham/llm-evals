"""Generic deterministic checks for any structured output.

Each reads one field of the output, named by `field` (dots reach into nested
objects: `answer.label`), and compares it with the case or a fixed rule:

- `field_matches`: equals `case.expected[expected_key]` (default: the field's
  own name). The expected value may be a list of acceptable values. Text is
  compared after trimming, collapsing spaces and, unless `ignore_case: false`,
  ignoring case. Cases without that expected key pass as not applicable.
- `regex`: matches `pattern` (or, with `must_match: false`, doesn't). `flags`
  takes any of "i" (ignore case), "m" (multiline) and "s" (dot matches newline).
- `keywords_present`: contains at least `min_share` of the keywords listed in
  `case.expected[expected_key]` (default "keywords"), ignoring case. Cases
  without that key pass as not applicable.
- `max_length`: is a string of at most `max_chars` characters.

A missing field fails, with the reason saying which field.
"""

from __future__ import annotations

import re
from typing import Any

from llm_evals.checks._base import BaseCheck, truncate
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score

_MISSING = object()


def get_field(output: Any, path: str) -> Any:
    value = output
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return _MISSING
    return value


def _norm(value: Any, ignore_case: bool) -> Any:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text.casefold() if ignore_case else text
    return value


class _FieldCheck(BaseCheck):
    kind: CheckKind = "deterministic"

    def __init__(self, *, name: str, field: str, critical: bool) -> None:
        super().__init__(critical=critical)
        if not isinstance(field, str) or not field:
            raise ValueError(f"{name}: `field` must name an output field")
        self.name = name
        self.field = field

    def ok(self, reason: str, **detail: Any) -> Score:
        return Score(check=self.name, kind=self.kind, passed=True, reason=reason,
                     sub_scores=[{"field": self.field, **detail}])

    def value(self, output: dict[str, Any]) -> Any:
        return get_field(output, self.field)


class FieldMatches(_FieldCheck):
    def __init__(self, *, field: str, expected_key: str | None, ignore_case: bool, critical: bool) -> None:
        super().__init__(name="field_matches", field=field, critical=critical)
        self.expected_key = expected_key or field
        self.ignore_case = ignore_case

    def applies(self, case: Case) -> bool:
        return get_field(case.expected or {}, self.expected_key) is not _MISSING

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        expected = get_field(case.expected or {}, self.expected_key)
        accepted = expected if isinstance(expected, list) else [expected]
        got = self.value(output)
        if got is _MISSING:
            return self.fail(f"output has no field {self.field!r}", sub_scores=[{"field": self.field}])
        if any(_norm(got, self.ignore_case) == _norm(a, self.ignore_case) for a in accepted):
            return self.ok(f"{self.field} is {truncate(str(got), 80)!r} as expected", got=got)
        want = accepted[0] if len(accepted) == 1 else accepted
        return self.fail(f"{self.field} is {truncate(str(got), 80)!r}, expected {want!r}",
                         sub_scores=[{"field": self.field, "got": got, "expected": expected}])


class Regex(_FieldCheck):
    def __init__(self, *, field: str, pattern: str, must_match: bool, flags: str, critical: bool) -> None:
        super().__init__(name="regex", field=field, critical=critical)
        bits = 0
        for flag in flags:
            bits |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}.get(flag, 0)
            if flag not in "ims":
                raise ValueError(f"regex: unknown flag {flag!r} (use i, m or s)")
        self.pattern = re.compile(pattern, bits)
        self.must_match = must_match

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        got = self.value(output)
        if not isinstance(got, str):
            return self.fail(f"output field {self.field!r} is missing or not text", sub_scores=[{"field": self.field}])
        found = self.pattern.search(got) is not None
        if found == self.must_match:
            verb = "matches" if found else "does not match"
            return self.ok(f"{self.field} {verb} /{self.pattern.pattern}/")
        verb = "does not match" if self.must_match else "matches"
        return self.fail(f"{self.field} {verb} /{self.pattern.pattern}/", sub_scores=[{"field": self.field}])


class KeywordsPresent(_FieldCheck):
    def __init__(self, *, field: str, expected_key: str, min_share: float, critical: bool) -> None:
        super().__init__(name="keywords_present", field=field, critical=critical)
        if not 0 < min_share <= 1:
            raise ValueError("keywords_present: min_share must be in (0, 1]")
        self.expected_key = expected_key
        self.min_share = min_share

    def applies(self, case: Case) -> bool:
        keywords = get_field(case.expected or {}, self.expected_key)
        return isinstance(keywords, list) and bool(keywords)

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        keywords = [str(k) for k in get_field(case.expected or {}, self.expected_key)]
        got = self.value(output)
        if not isinstance(got, str):
            return self.fail(f"output field {self.field!r} is missing or not text", sub_scores=[{"field": self.field}])
        text = got.casefold()
        missing = [k for k in keywords if k.casefold() not in text]
        share = 1 - len(missing) / len(keywords)
        if share >= self.min_share:
            return self.ok(f"{len(keywords) - len(missing)} of {len(keywords)} keywords present", missing=missing)
        return self.fail(f"missing {', '.join(repr(k) for k in missing)} ({len(keywords) - len(missing)} of "
                         f"{len(keywords)} present)", sub_scores=[{"field": self.field, "missing": missing}])


class MaxLength(_FieldCheck):
    def __init__(self, *, field: str, max_chars: int, critical: bool) -> None:
        super().__init__(name="max_length", field=field, critical=critical)
        self.max_chars = max_chars

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        got = self.value(output)
        if not isinstance(got, str):
            return self.fail(f"output field {self.field!r} is missing or not text", sub_scores=[{"field": self.field}])
        if len(got) <= self.max_chars:
            return self.ok(f"{self.field} is {len(got)} characters", length=len(got))
        return self.fail(f"{self.field} is {len(got)} characters (max {self.max_chars})",
                         sub_scores=[{"field": self.field, "length": len(got)}])


@register_check("field_matches")
def field_matches(*, field: str, expected_key: str | None = None, ignore_case: bool = True,
                  critical: bool = False) -> FieldMatches:
    return FieldMatches(field=field, expected_key=expected_key, ignore_case=ignore_case, critical=critical)


@register_check("regex")
def regex(*, field: str, pattern: str, must_match: bool = True, flags: str = "", critical: bool = False) -> Regex:
    return Regex(field=field, pattern=pattern, must_match=must_match, flags=flags, critical=critical)


@register_check("keywords_present")
def keywords_present(*, field: str, expected_key: str = "keywords", min_share: float = 1.0,
                     critical: bool = False) -> KeywordsPresent:
    return KeywordsPresent(field=field, expected_key=expected_key, min_share=min_share, critical=critical)


@register_check("max_length")
def max_length(*, field: str, max_chars: int, critical: bool = False) -> MaxLength:
    return MaxLength(field=field, max_chars=max_chars, critical=critical)
