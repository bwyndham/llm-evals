"""`atom_keywords`: deterministic keyword faithfulness for constructed spec-writing cases.

Implements the rule in docs/contracts.md ("Keyword faithfulness"):

- Ticket text = title + description + acceptance_criteria (assumptions and
  open questions are excluded, so flagging a requirement as an assumption does
  not count as covering it).
- An atom is covered when at least ceil(`min_keyword_share` x len(keywords)) of
  its `keywords` appear in the ticket text (case-insensitive substring).
- Hinted atoms (`source["hinted_atom_ids"]`, or `mode == "hinted"`) are only
  implied by the ask, so they are reported but not required.
- A Scenario whose name and step text (step keywords excluded; doc strings and
  table cells included) contain none of the keywords of any atom in the case,
  hinted ones included, is an invented requirement.
- The check fails when any non-hinted atom is uncovered or any scenario is
  invented. `value` is the share of required atoms covered.

The case must carry `source["atoms"]` with `keywords` (cases built by
`ask_renderer` do); without them the check fails with a clear reason.
"""

from __future__ import annotations

import math
import re
from typing import Any, cast

from gherkin import Parser
from gherkin.errors import ParserError

from llm_evals.checks._base import BaseCheck
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score

_SCENARIO_LINE = re.compile(r"^\s*(?:Scenario(?: Outline| Template)?|Example)\s*:\s*(.*)$")
_STEP_LINE = re.compile(r"^\s*(?:Given|When|Then|And|But|\*)\s+(.*)$")


def _step_text(step: dict[str, Any]) -> list[str]:
    parts = [str(step.get("text", ""))]
    doc = step.get("docString")
    if doc:
        parts.append(str(doc.get("content", "")))
    table = step.get("dataTable")
    if table:
        parts += [str(c.get("value", "")) for row in table.get("rows", []) for c in row.get("cells", [])]
    return parts


def scenario_texts(gherkin_text: str) -> list[tuple[str, str]]:
    """(name, name + step text) for every scenario, via the Gherkin parser.

    If the text doesn't parse, fall back to a line scan so invented scenarios are
    still found (gherkin_parses reports the syntax problem separately).
    """
    try:
        doc = cast(dict[str, Any], Parser().parse(gherkin_text))
    except ParserError:
        doc = None
    out: list[tuple[str, str]] = []
    if doc is not None:
        def walk(children: list[dict[str, Any]]) -> None:
            for child in children:
                if "rule" in child:
                    walk(child["rule"].get("children") or [])
                elif "scenario" in child:
                    node = child["scenario"]
                    parts = [str(node.get("name", ""))]
                    for step in node.get("steps") or []:
                        parts += _step_text(step)
                    for ex in node.get("examples") or []:
                        for row in [ex.get("tableHeader")] + list(ex.get("tableBody") or []):
                            if row:
                                parts += [str(c.get("value", "")) for c in row.get("cells", [])]
                    out.append((parts[0], "\n".join(parts)))

        walk(((doc.get("feature") or {}).get("children")) or [])
        return out
    current: list[str] | None = None
    for line in gherkin_text.splitlines():
        m = _SCENARIO_LINE.match(line)
        if m:
            current = [m.group(1).strip()]
            out.append((current[0], ""))
            continue
        s = _STEP_LINE.match(line)
        if s and current is not None:
            current.append(s.group(1))
            out[-1] = (current[0], "\n".join(current))
    return [(name, text or name) for name, text in out]


class AtomKeywords(BaseCheck):
    name = "atom_keywords"
    kind: CheckKind = "deterministic"

    def __init__(self, *, min_keyword_share: float, critical: bool) -> None:
        super().__init__(critical=critical)
        if isinstance(min_keyword_share, bool) or not isinstance(min_keyword_share, (int, float)) or not (
            0.0 < float(min_keyword_share) <= 1.0
        ):
            raise TypeError(f"atom_keywords: min_keyword_share must be in (0, 1], got {min_keyword_share!r}")
        self.min_keyword_share = float(min_keyword_share)

    def needed(self, n_keywords: int) -> int:
        # The small epsilon keeps float error (0.6 * 5 = 3.0000000000000004) from adding one.
        return max(1, math.ceil(self.min_keyword_share * n_keywords - 1e-9))

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        source = case.source or {}
        atoms = [a for a in source.get("atoms") or [] if isinstance(a, dict)]
        if not atoms or not all(a.get("keywords") for a in atoms):
            return self.fail("case has no source atoms with keywords; atom_keywords needs constructed cases")
        hinted_ids = set(source.get("hinted_atom_ids") or [])
        hinted_ids |= {str(a.get("id")) for a in atoms if a.get("mode") == "hinted"}

        ticket_text = "\n".join(
            v for v in (output.get(k) for k in ("title", "description", "acceptance_criteria")) if isinstance(v, str)
        ).lower()

        sub_scores: list[dict[str, Any]] = []
        failures: list[str] = []
        required = covered_required = 0
        all_keywords: list[str] = []
        for atom in atoms:
            atom_id = str(atom.get("id"))
            keywords = [str(k).lower() for k in atom["keywords"]]
            all_keywords += keywords
            matched = [k for k in keywords if k in ticket_text]
            need = self.needed(len(keywords))
            covered = len(matched) >= need
            hinted = atom_id in hinted_ids
            sub_scores.append({
                "atom": atom_id,
                "covered": covered,
                "hinted": hinted,
                "matched": matched,
                "missing": [k for k in keywords if k not in matched],
                "needed": need,
            })
            if not hinted:
                required += 1
                covered_required += covered
                if not covered:
                    failures.append(
                        f"atom {atom_id} uncovered: {len(matched)}/{len(keywords)} keywords "
                        f"(needs {need}; missing {', '.join(k for k in keywords if k not in matched)})"
                    )

        ac = output.get("acceptance_criteria")
        scenarios = scenario_texts(ac) if isinstance(ac, str) else []
        keyword_set = sorted(set(all_keywords))
        for name, text in scenarios:
            lowered = text.lower()
            if not any(k in lowered for k in keyword_set):
                sub_scores.append({"invented_scenario": name, "reason": "no atom keyword in name or steps"})
                failures.append(f"invented scenario {name!r}: no atom keyword in its name or steps")

        value = covered_required / required if required else 1.0
        if failures:
            return self.fail("; ".join(failures), value=value, sub_scores=sub_scores)
        n_hinted = sum(1 for s in sub_scores if s.get("hinted"))
        return Score(
            check=self.name,
            kind=self.kind,
            passed=True,
            value=value,
            reason=(
                f"all {required} required atom(s) covered"
                + (f" ({n_hinted} hinted atom(s) not required)" if n_hinted else "")
                + f"; {len(scenarios)} scenario(s), none invented"
            ),
            sub_scores=sub_scores,
        )


@register_check("atom_keywords")
def atom_keywords(*, min_keyword_share: float = 0.6, critical: bool = False) -> AtomKeywords:
    return AtomKeywords(min_keyword_share=min_keyword_share, critical=critical)
