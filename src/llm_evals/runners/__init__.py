"""Runners: version manifests, the prompt runner, providers, errors, concurrency.

Importing this package registers the built-in runners: "prompt" and "simulated" (QA-1).
"""

from llm_evals.runners import prompt  # noqa: F401  (registers "prompt")
from llm_evals.runners import simulated  # noqa: F401  (registers "simulated")
from llm_evals.runners.concurrency import RunOutcome, run_all
from llm_evals.runners.errors import (
    FatalRunError,
    InfraError,
    classify_exception,
    infra_share,
    is_incomplete,
)
from llm_evals.runners.manifest import (
    ManifestError,
    compute_manifest_id,
    load_manifest,
    manifest_with_overrides,
)
from llm_evals.runners.prompt import PromptRunner
from llm_evals.runners.schemas import get_schema
from llm_evals.runners import agent_replay  # noqa: F401

__all__ = [
    "FatalRunError",
    "InfraError",
    "ManifestError",
    "PromptRunner",
    "RunOutcome",
    "classify_exception",
    "compute_manifest_id",
    "get_schema",
    "infra_share",
    "is_incomplete",
    "load_manifest",
    "manifest_with_overrides",
    "run_all",
]
