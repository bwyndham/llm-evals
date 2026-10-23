"""`gherkin_parses`: the acceptance criteria are well-formed Gherkin.

Parsing uses gherkin-official (`gherkin.Parser().parse(text)`), which returns
the Cucumber AST as plain dicts and raises `gherkin.errors.ParserError`
(usually `CompositeParserException`, whose `.errors` carry line/column
locations) on syntax errors.

Pass when the text parses, holds exactly one Feature, has at least one
Scenario or Scenario Outline (including those inside a Rule), and every
scenario has at least one When (keywordType "Action") and one Then
(keywordType "Outcome") step. And/But ("Conjunction") and `*` ("Unknown")
steps take the type of the step before them. Background steps (feature or
rule level) run before each scenario, so they count toward its step types.

The parser does not reject a second `Feature:` line that comes before any
scenario; it folds it into the first feature's description. This module scans
every description for a line that starts with a Feature keyword of the
document's dialect and counts each one as an extra Feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from gherkin import Parser
from gherkin.dialect import Dialect
from gherkin.errors import CompositeParserException, ParserError, ParserException

from llm_evals.checks._base import BaseCheck
from llm_evals.registry import register_check
from llm_evals.types import Case, CheckContext, CheckKind, Score

_LOCATION_PREFIX = re.compile(r"^\(\d+:\d+\):\s*")
_CONTINUATION_TYPES = {"Conjunction", "Unknown"}


@dataclass
class ScenarioInfo:
    name: str
    keyword: str
    line: int
    steps: int  # the scenario's own steps, not counting Background
    has_when: bool
    has_then: bool


@dataclass
class GherkinReport:
    parse_error: str | None = None
    feature_count: int = 0
    scenarios: list[ScenarioInfo] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parse_error is None and not self.problems


def _format_parse_error(exc: ParserError) -> str:
    errors: list[ParserException]
    if isinstance(exc, CompositeParserException):
        errors = list(exc.errors)
    elif isinstance(exc, ParserException):
        errors = [exc]
    else:
        return f"Gherkin parse error: {exc}"
    parts = []
    for err in errors:
        loc = getattr(err, "location", None) or {}
        message = _LOCATION_PREFIX.sub("", str(err.args[0]) if err.args else str(err))
        parts.append(f"line {loc.get('line', '?')}, column {loc.get('column', '?')}: {message}")
    return "Gherkin parse error at " + "; ".join(parts)


def _step_types(steps: list[dict[str, Any]], previous: str | None) -> tuple[set[str], str | None]:
    """Resolve And/But/* to the preceding step's type; return the types seen and the last one."""
    seen: set[str] = set()
    for step in steps:
        kind = step.get("keywordType")
        if kind in _CONTINUATION_TYPES or kind is None:
            kind = previous
        if kind is not None:
            seen.add(kind)
        previous = kind
    return seen, previous


def _description_feature_lines(node: dict[str, Any], keywords: list[str]) -> list[str]:
    hits = []
    for line in (node.get("description") or "").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f"{kw}:") for kw in keywords):
            hits.append(stripped)
    return hits


def analyze_gherkin(text: Any) -> GherkinReport:
    """Parse `text` and report its structure and any rule violations."""
    report = GherkinReport()
    if not isinstance(text, str):
        report.parse_error = f"acceptance_criteria is {type(text).__name__}, not a string"
        return report
    if not text.strip():
        report.parse_error = "acceptance_criteria is empty"
        return report
    try:
        doc = cast(dict[str, Any], Parser().parse(text))
    except ParserError as exc:
        report.parse_error = _format_parse_error(exc)
        return report

    feature = doc.get("feature")
    if not feature:
        report.problems.append("no Feature found")
        return report

    dialect = Dialect.for_name(feature.get("language", "en")) or Dialect.for_name("en")
    feature_keywords = dialect.feature_keywords if dialect else ["Feature"]

    extra_features: list[str] = []
    extra_features += _description_feature_lines(feature, feature_keywords)

    def walk(children: list[dict[str, Any]], background_steps: list[dict[str, Any]]) -> None:
        bg = list(background_steps)
        for child in children:
            if "background" in child:
                node = child["background"]
                extra_features.extend(_description_feature_lines(node, feature_keywords))
                bg = bg + list(node.get("steps") or [])
            elif "rule" in child:
                node = child["rule"]
                extra_features.extend(_description_feature_lines(node, feature_keywords))
                walk(node.get("children") or [], bg)
            elif "scenario" in child:
                node = child["scenario"]
                extra_features.extend(_description_feature_lines(node, feature_keywords))
                for ex in node.get("examples") or []:
                    extra_features.extend(_description_feature_lines(ex, feature_keywords))
                own = list(node.get("steps") or [])
                bg_types, last = _step_types(bg, None)
                own_types, _ = _step_types(own, last)
                types = bg_types | own_types
                report.scenarios.append(
                    ScenarioInfo(
                        name=node.get("name", ""),
                        keyword=node.get("keyword", "Scenario"),
                        line=(node.get("location") or {}).get("line", 0),
                        steps=len(own),
                        has_when="Action" in types,
                        has_then="Outcome" in types,
                    )
                )

    walk(feature.get("children") or [], [])

    report.feature_count = 1 + len(extra_features)
    if extra_features:
        report.problems.append(
            f"expected exactly one Feature, found {report.feature_count} "
            f"(extra: {', '.join(repr(f) for f in extra_features)})"
        )
    if not report.scenarios:
        report.problems.append("no Scenario or Scenario Outline found")
    for sc in report.scenarios:
        missing = [label for label, ok in (("When", sc.has_when), ("Then", sc.has_then)) if not ok]
        if missing:
            label = sc.name or "<unnamed>"
            report.problems.append(f"{sc.keyword} {label!r} (line {sc.line}) has no {' or '.join(missing)} step")
    return report


def scenario_count(gherkin_text: Any) -> int | None:
    """Number of scenarios (including outlines and those in Rules), or None if the text
    doesn't parse. A document that parses but has no Feature counts 0."""
    report = analyze_gherkin(gherkin_text)
    if report.parse_error is not None:
        return None
    return len(report.scenarios)


class GherkinParses(BaseCheck):
    name = "gherkin_parses"
    kind: CheckKind = "deterministic"

    def _score(self, case: Case, output: dict[str, Any], ctx: CheckContext) -> Score:
        if "acceptance_criteria" not in output:
            return self.fail("acceptance_criteria is missing")
        report = analyze_gherkin(output["acceptance_criteria"])
        sub_scores = [
            {"scenario": sc.name, "steps": sc.steps, "has_when": sc.has_when, "has_then": sc.has_then}
            for sc in report.scenarios
        ]
        if report.parse_error is not None:
            return self.fail(report.parse_error)
        if report.problems:
            return self.fail("; ".join(report.problems), sub_scores=sub_scores)
        return Score(
            check=self.name,
            kind=self.kind,
            passed=True,
            reason=f"1 Feature, {len(report.scenarios)} scenario(s), each with When and Then",
            sub_scores=sub_scores,
        )


@register_check("gherkin_parses")
def gherkin_parses(*, critical: bool = True) -> GherkinParses:
    return GherkinParses(critical=critical)
