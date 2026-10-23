"""Everything a suite's own plug-ins need, in one import.

    from llm_evals.api import BaseCheck, Score, register_check

    @register_check("mentions_order_id")
    def mentions_order_id(*, critical: bool = False):
        return MentionsOrderId(critical=critical)

List the plug-in module under `plugins:` in suite.yaml, as a module name or a `.py`
file next to it, and refer to the check by name under `checks:`. See
docs/adopting.md for checks, LLM judges, runners and output schemas.
"""

from __future__ import annotations

from llm_evals.checks._base import BaseCheck
from llm_evals.judges.base import JudgeCheckMixin, load_rubric
from llm_evals.registry import register_check, register_pairwise, register_renderer, register_runner
from llm_evals.runners.schemas import register_schema
from llm_evals.types import Case, CheckContext, CheckKind, GatingUnit, Result, Score

__all__ = [
    "BaseCheck",
    "Case",
    "CheckContext",
    "CheckKind",
    "GatingUnit",
    "JudgeCheckMixin",
    "Result",
    "Score",
    "load_rubric",
    "register_check",
    "register_pairwise",
    "register_renderer",
    "register_runner",
    "register_schema",
]
