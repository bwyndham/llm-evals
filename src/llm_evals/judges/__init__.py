"""LLM judges: a cached JudgeClient and the plumbing the judge checks share.

The checks themselves live in `llm_evals.checks` (`faithfulness`, `checklist`)
and the pairwise comparison in `llm_evals.checks.pairwise`.
"""

from llm_evals.judges.cache import CachedResponse, JudgeCache
from llm_evals.judges.client import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PARAMS,
    JudgeClient,
    JudgeError,
    JudgeResult,
    build_user_prompt,
    payload_block,
    prompt_hash,
    unit_hash,
)

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_PARAMS",
    "CachedResponse",
    "JudgeCache",
    "JudgeClient",
    "JudgeError",
    "JudgeResult",
    "build_user_prompt",
    "payload_block",
    "prompt_hash",
    "unit_hash",
]
