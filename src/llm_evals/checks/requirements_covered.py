"""`requirements_covered`: every requirement the input states shows up in the output.

A format-agnostic sibling of `atom_keywords`. You name the output fields where a
requirement counts as covered (say, a ticket's acceptance criteria, but not its
summary or assumptions), and each requirement's keywords must appear there.

Where the requirements come from, per case:
- `case.expected[requirements_key]` (default key "requirements"): a list of
  `{"id": ..., "keywords": [...], "hinted": false}` objects, or plain keyword lists;
- otherwise `case.source["atoms"]`, as cases built by `ask_renderer` carry
  (hinted atoms come from `source["hinted_atom_ids"]` or `mode == "hinted"`).
Cases with neither pass as not applicable. Hinted requirements are only implied
by the input, so they're reported but not required.

Where it looks (`fields`, dotted paths into the output): each field is split
into *places*. A list field gives one place per item; a string field is one
place, or one per line / paragraph with `split: lines|paragraphs`. A keyword
counts only if it's found within a single place, so a criterion has to state
the requirement, not just mention its words in passing across the ticket.

How it matches (`match`, see `llm_evals.checks.keywords`): `words` (default:
every word of the keyword, any form and order) or `literal` (substring). A
requirement is covered when at least ceil(`min_keyword_share` x its keywords)
are found. The check fails if any required requirement isn't; `value` is the
share of required requirements covered.

    checks:
      - requirements_covered: {fields: [acceptance_criteria]}
      - requirements_covered: {fields: [answer.steps], split: lines, min_keyword_share: 1.0}
"""

from __future__ import annotations

import math
import re
from typing import Any

from llm_evals.checks._base import BaseCheck
from llm_evals.checks.generic import _MISSING, get_field
from llm_evals.checks.keywords import MATCH_MODES, MatchMode, keyword_in_any
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score

SPLITS = ("none", "lines", "paragraphs")


def requirements_of(case: Case, key: str) -> list[dict[str, Any]]:
    """[{id, keywords, hinted, text}] for the case, or [] when it has none."""
    raw = (case.expected or {}).get(key)
    out: list[dict[str, Any]] = []
    if isinstance(raw, list) and raw:
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                kws = [str(k) for k in item.get("keywords") or []]
                out.append({"id": str(item.get("id", i)), "keywords": kws, "hinted": bool(item.get("hinted")),
                            "text": str(item.get("text") or item.get("id") or i)})
            elif isinstance(item, list):
                out.append({"id": str(i), "keywords": [str(k) for k in item], "hinted": False,
                            "text": ", ".join(map(str, item))})
        return [r for r in out if r["keywords"]]
    source = case.source or {}
    atoms = [a for a in source.get("atoms") or [] if isinstance(a, dict) and a.get("keywords")]
    hinted = set(source.get("hinted_atom_ids") or []) | {str(a.get("id")) for a in atoms if a.get("mode") == "hinted"}
    return [{"id": str(a.get("id")), "keywords": [str(k) for k in a["keywords"]], "hinted": str(a.get("id")) in hinted,
             "text": str(a.get("text") or a.get("id"))} for a in atoms]


class RequirementsCovered(BaseCheck):
    name = "requirements_covered"
    kind: CheckKind = "deterministic"

    def __init__(self, *, fields: list[str], min_keyword_share: float, match: MatchMode, split: str,
                 requirements_key: str, critical: bool) -> None:
        super().__init__(critical=critical)
        if isinstance(fields, str):
            fields = [fields]
        if not fields or not all(isinstance(f, str) and f for f in fields):
            raise TypeError(f"requirements_covered: fields must be a non-empty list of field paths, got {fields!r}")
        if isinstance(min_keyword_share, bool) or not isinstance(min_keyword_share, (int, float)) or not (
                0.0 < float(min_keyword_share) <= 1.0):
            raise TypeError(f"requirements_covered: min_keyword_share must be in (0, 1], got {min_keyword_share!r}")
        if match not in MATCH_MODES:
            raise TypeError(f"requirements_covered: match must be one of {', '.join(MATCH_MODES)}, got {match!r}")
        if split not in SPLITS:
            raise TypeError(f"requirements_covered: split must be one of {', '.join(SPLITS)}, got {split!r}")
        self.fields = list(fields)
        self.min_share = float(min_keyword_share)
        self.match: MatchMode = match
        self.split = split
        self.requirements_key = requirements_key

    def applies(self, case: Case) -> bool:
        return bool(requirements_of(case, self.requirements_key))

    def needed(self, n_keywords: int) -> int:
        return max(1, math.ceil(self.min_share * n_keywords - 1e-9))

    def places(self, output: dict[str, Any]) -> tuple[list[str], list[str]]:
        """(places to search, fields that were missing)."""
        places: list[str] = []
        missing: list[str] = []
        for path in self.fields:
            value = get_field(output, path)
            if value is _MISSING or value is None:
                missing.append(path)
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                text = item if isinstance(item, str) else str(item)
                if self.split == "lines":
                    places += [line for line in text.splitlines() if line.strip()]
                elif self.split == "paragraphs":
                    places += [p for p in re.split(r"\n\s*\n", text) if p.strip()]
                elif text.strip():
                    places.append(text)
        return places, missing

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        reqs = requirements_of(case, self.requirements_key)
        places, missing_fields = self.places(output)
        if len(missing_fields) == len(self.fields):
            return self.fail(f"output has none of the fields {', '.join(self.fields)}")
        sub_scores: list[dict[str, Any]] = []
        uncovered: list[str] = []
        required = covered_required = 0
        for r in reqs:
            found = [k for k in r["keywords"] if keyword_in_any(k, places, self.match)]
            need = self.needed(len(r["keywords"]))
            covered = len(found) >= need
            sub_scores.append({"requirement": r["id"], "covered": covered, "hinted": r["hinted"],
                               "found": found, "missing": [k for k in r["keywords"] if k not in found],
                               "needed": need})
            if not r["hinted"]:
                required += 1
                covered_required += covered
                if not covered:
                    uncovered.append(r["text"])
        value = covered_required / required if required else 1.0
        where = ", ".join(self.fields)
        if uncovered:
            return self.fail(f"{len(uncovered)} of {required} requirement(s) not covered in {where}: "
                             + "; ".join(uncovered[:3]) + (" …" if len(uncovered) > 3 else ""),
                             value=value, sub_scores=sub_scores)
        n_hinted = len(reqs) - required
        return Score(check=self.name, kind=self.kind, passed=True, value=value,
                     reason=f"all {required} requirement(s) covered in {where}"
                            + (f" ({n_hinted} hinted, not required)" if n_hinted else ""),
                     sub_scores=sub_scores)


@register_check("requirements_covered")
def requirements_covered(*, fields: list[str], min_keyword_share: float = 0.6, match: MatchMode = "words",
                         split: str = "none", requirements_key: str = "requirements",
                         critical: bool = False) -> RequirementsCovered:
    return RequirementsCovered(fields=fields, min_keyword_share=min_keyword_share, match=match, split=split,
                               requirements_key=requirements_key, critical=critical)


__all__ = ["RequirementsCovered", "requirements_of"]
