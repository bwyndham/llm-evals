"""`gaps_flagged`: when the ask leaves a gap, the ticket says so.

Cases built from known answers record where the ask is incomplete: the actor is
left out (`strata.missing_actor == "yes"`, or `source.actor_stated` is false),
or a requirement is only hinted at (`source.hinted_atom_ids`). On those cases
the spec writer should record what it assumed or ask about it, so the ticket's
`assumptions` or `open_questions` must be non-empty.

This checks that something was flagged, not that the flag is about the right
gap; the faithfulness judge covers that once it's calibrated. Cases with no gap
pass as not applicable.
"""

from __future__ import annotations

from typing import Any

from llm_evals.checks._base import BaseCheck
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score


def _gaps(case: Case) -> list[str]:
    source = case.source or {}
    gaps = []
    if case.strata.get("missing_actor") == "yes" or source.get("actor_stated") is False:
        gaps.append("actor not stated")
    hinted = source.get("hinted_atom_ids") or []
    if hinted:
        gaps.append(f"{len(hinted)} requirement(s) only hinted")
    return gaps


def _non_empty(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(i) for i in items if isinstance(i, str) and i.strip()]


class GapsFlagged(BaseCheck):
    name = "gaps_flagged"
    kind: CheckKind = "deterministic"

    def applies(self, case: Case) -> bool:
        return bool(_gaps(case))

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        gaps = _gaps(case)
        if not gaps:
            return Score(check=self.name, kind=self.kind, passed=True,
                         reason="not applicable: the ask has no known gap",
                         sub_scores=[{"applicable": False}])
        flags = _non_empty(output.get("assumptions")) + _non_empty(output.get("open_questions"))
        detail = {"applicable": True, "gaps": gaps, "n_flags": len(flags)}
        if not flags:
            return self.fail(f"the ask leaves gaps ({'; '.join(gaps)}) but the ticket lists no "
                             "assumptions or open questions", sub_scores=[detail])
        return Score(check=self.name, kind=self.kind, passed=True,
                     reason=f"{len(flags)} assumption(s) or question(s) flag the gaps",
                     sub_scores=[detail])


@register_check("gaps_flagged")
def gaps_flagged(*, critical: bool = False) -> GapsFlagged:
    return GapsFlagged(critical=critical)
