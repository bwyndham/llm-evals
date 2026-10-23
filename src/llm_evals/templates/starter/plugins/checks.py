"""Your own checks. Each one is a class with a `_score` method and a factory
registered by name; list this file under `plugins:` in suite.yaml and use the
name under `checks:`. See docs/adopting.md in llm-evals for more."""

from __future__ import annotations

import re

from llm_evals.api import BaseCheck, Score, register_check

BLAME = re.compile(r"your fault|make sure you|you should have|did you (?:try|read)|user error", re.IGNORECASE)


class NoBlame(BaseCheck):
    """The reply never suggests the problem is the customer's fault."""

    name = "no_blame"
    kind = "deterministic"

    def _score(self, case, output, ctx):
        match = BLAME.search(str(output.get("reply", "")))
        if match:
            return self.fail(f"the reply blames the customer: {match.group(0)!r}")
        return Score(check=self.name, kind=self.kind, passed=True, reason="no blame")


@register_check("no_blame")
def no_blame(*, critical: bool = False) -> NoBlame:
    return NoBlame(critical=critical)
