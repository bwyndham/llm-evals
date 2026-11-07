"""Keyword matching shared by the coverage checks.

Two modes:

- `literal`: the keyword is a case-insensitive substring of the text. Simple,
  but brittle on real model output: "api keys" misses "an API key", and
  "wcag 2.1 aa" misses "WCAG 2.1 level AA".
- `words`: every word of the keyword appears in the text, in any order and any
  form. A text word matches a keyword word when their crude stems are equal
  ("rates"/"rate", "named"/"name", "syncs"/"sync") or the text word starts with
  it (keywords written as prefixes, like "approv", still work). Hyphenated words
  also count as their parts ("overdue-invoice" covers "overdue"), except number
  ranges like "0-30". Small filler words in a keyword ("a", "the", "level", ...)
  are ignored.

`words` mode is meant to be applied per *place*: one acceptance criterion, one
list item, one field. A keyword whose words are spread across two different
criteria isn't covered.
"""

from __future__ import annotations

import re
from typing import Literal

MatchMode = Literal["literal", "words"]
MATCH_MODES: tuple[str, ...] = ("literal", "words")

_TOKEN = re.compile(r"[a-z0-9]+(?:[.%+-][a-z0-9%+]+)*%?\+?")
_NUMERIC = re.compile(r"[\d.%+-]+")
FILLER = frozenset({"a", "an", "the", "of", "to", "for", "and", "level", "by", "in", "on"})


def stem(word: str) -> str:
    """Crude and consistent: applied to both sides, so it only has to agree with itself."""
    if not word.isalpha() or len(word) <= 3:
        return word
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def words(text: str) -> list[str]:
    """Lowercase tokens, plus the parts of hyphenated words (number ranges stay whole)."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "-" in tok and not _NUMERIC.fullmatch(tok):
            out += [p for p in tok.split("-") if p]
    return out


def _words_match(keyword: str, text: str) -> bool:
    kw = [w for w in words(keyword) if w not in FILLER] or words(keyword)
    if not kw:
        return False
    found = words(text)
    stems = {stem(w) for w in found}
    return all(stem(k) in stems or any(w.startswith(k) for w in found) for k in kw)


def keyword_in(keyword: str, text: str, mode: MatchMode = "literal") -> bool:
    """Whether `keyword` occurs in one piece of text."""
    if mode == "literal":
        return keyword.lower() in text.lower()
    if mode == "words":
        return _words_match(keyword, text)
    raise ValueError(f"match must be one of {', '.join(MATCH_MODES)}, got {mode!r}")


def keyword_in_any(keyword: str, places: list[str], mode: MatchMode = "literal") -> bool:
    """Whether `keyword` occurs within any single place."""
    return any(keyword_in(keyword, p, mode) for p in places)


__all__ = ["FILLER", "MATCH_MODES", "MatchMode", "keyword_in", "keyword_in_any", "stem", "words"]
