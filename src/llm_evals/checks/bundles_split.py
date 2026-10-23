"""`bundles_split`: each request in a bundled ask gets its own scenario.

`atom_keywords` can't see a merged ticket: a single scenario can mention every
request's keywords and still cover them all. This check looks at structure. On
a bundled case (`source["atom_set_ids"]` names two or more atom sets), each set
must have a scenario that matches one of its required atoms, and those
scenarios must be different ones, so one merged scenario covering both fails.

An atom matches a scenario when at least ceil(`min_keyword_share` x keywords)
of its keywords appear in the scenario's name and step text (the same rule as
`atom_keywords`, scoped to one scenario). Hinted atoms are not required.

Non-bundled cases pass as not applicable. Gherkin that doesn't parse also
passes as not applicable here; `gherkin_parses` reports that failure.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any

from llm_evals.checks._base import BaseCheck
from llm_evals.checks.atom_keywords import scenario_texts
from llm_evals.checks.gherkin import analyze_gherkin
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score


class BundlesSplit(BaseCheck):
    name = "bundles_split"
    kind: CheckKind = "deterministic"

    def __init__(self, *, min_keyword_share: float, critical: bool) -> None:
        super().__init__(critical=critical)
        if not 0 < min_keyword_share <= 1:
            raise TypeError(f"bundles_split: min_keyword_share must be in (0, 1], got {min_keyword_share!r}")
        self.min_keyword_share = min_keyword_share

    def _matches(self, atom: dict[str, Any], text: str) -> bool:
        keywords = [str(k).lower() for k in atom.get("keywords") or []]
        if not keywords:
            return False
        needed = max(1, math.ceil(self.min_keyword_share * len(keywords)))
        return sum(k in text for k in keywords) >= needed

    def applies(self, case: Case) -> bool:
        return len((case.source or {}).get("atom_set_ids") or []) >= 2

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        source = case.source or {}
        set_ids = list(source.get("atom_set_ids") or [])
        if len(set_ids) < 2:
            return Score(check=self.name, kind=self.kind, passed=True, reason="not applicable: not bundled",
                         sub_scores=[{"applicable": False}])
        text = str(output.get("acceptance_criteria") or "")
        if not analyze_gherkin(text).ok:  # same verdict as gherkin_parses
            return Score(check=self.name, kind=self.kind, passed=True,
                         reason="not applicable: Gherkin doesn't parse (gherkin_parses reports it)",
                         sub_scores=[{"applicable": False}])
        scenarios = [body.lower() for _, body in scenario_texts(text)]
        hinted = set(source.get("hinted_atom_ids") or [])
        atoms = [a for a in source.get("atoms") or []
                 if a.get("id") not in hinted and a.get("mode") != "hinted"]
        by_set: dict[str, set[int]] = {}
        for set_id in set_ids:
            set_atoms = [a for a in atoms if a.get("set_id") == set_id]
            by_set[set_id] = {i for i, body in enumerate(scenarios)
                              if any(self._matches(a, body) for a in set_atoms)}
        missing = [s for s, idx in by_set.items() if not idx]
        detail = [{"set": s, "scenarios": sorted(idx)} for s, idx in by_set.items()]
        if missing:
            return self.fail(f"no scenario for request(s): {', '.join(missing)}",
                             sub_scores=[{"applicable": True, "n_scenarios": len(scenarios)}, *detail])
        # Each set needs its own scenario: look for an assignment of distinct scenarios.
        options = [sorted(by_set[s]) for s in set_ids]
        if not _distinct_choice(options):
            return self.fail(f"{len(set_ids)} requests share {len(set().union(*by_set.values()))} "
                             "scenario(s): bundled requests were merged",
                             sub_scores=[{"applicable": True, "n_scenarios": len(scenarios)}, *detail])
        return Score(check=self.name, kind=self.kind, passed=True,
                     reason=f"each of {len(set_ids)} requests has its own scenario",
                     sub_scores=[{"applicable": True, "n_scenarios": len(scenarios)}, *detail])


def _distinct_choice(options: list[list[int]]) -> bool:
    """True if one scenario can be picked per request with no scenario picked twice."""
    pool = sorted(set().union(*options))
    if len(pool) < len(options):
        return False
    for combo in permutations(pool, len(options)):
        if all(c in opts for c, opts in zip(combo, options)):
            return True
    return False


@register_check("bundles_split")
def bundles_split(*, min_keyword_share: float = 0.6, critical: bool = False) -> BundlesSplit:
    return BundlesSplit(min_keyword_share=min_keyword_share, critical=critical)
