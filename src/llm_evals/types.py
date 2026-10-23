"""Shared types that every module builds against.

Runners, checks, stats, storage and the CLI all import from here, so a change
to a field is a change to every module's contract. Add fields with defaults;
don't rename or remove them without updating every caller.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Provenance = Literal["real", "constructed", "discovered", "public"]
DataClass = Literal["public", "internal", "restricted"]
ReviewStatus = Literal["approved", "pending", "rejected"]
CheckKind = Literal[
    "deterministic",
    "reference",
    "metamorphic",
    "faithfulness",
    "judge",
    "consistency",
    "structural",
]


class Case(BaseModel):
    """One test input. Cases live as JSONL lines, one Case per line."""

    id: str
    input: dict[str, Any]
    expected: dict[str, Any] | None = None  # known answer, if labeled or constructed
    source: dict[str, Any] | None = None  # facts the output must stay faithful to
    strata: dict[str, str] = Field(default_factory=dict)  # for slicing and weighting
    fixtures: dict[str, Any] | None = None  # recorded tool responses for agent replay
    provenance: Provenance
    tags: list[str] = Field(default_factory=list)
    held_out: bool = False  # CASE-7: never shown in PR comments or published
    review_status: ReviewStatus = "approved"


class VersionManifest(BaseModel):
    """VER-1: everything that defines the version under test.

    `id` is filled by `llm_evals.runners.manifest.load_manifest`, which hashes
    the model, params, skills, code SHA and the *contents* of each prompt
    template. Two manifests with the same id must behave identically.
    """

    name: str
    model: str  # pinned model id, never an alias
    prompt_templates: dict[str, str]  # role -> path relative to the suite dir
    params: dict[str, Any] = Field(default_factory=dict)  # max_tokens, effort, ...
    skills: list[str] = Field(default_factory=list)
    code_sha: str | None = None
    id: str = ""


class ErrorKind(str, Enum):
    """RUN-11. Infra errors are retried then excluded; system errors are failures."""

    NONE = "none"
    INFRA = "infra"  # timeouts, provider 5xx, rate limits still failing after retries
    SYSTEM = "system"  # malformed or unparseable output, refusals, tool misuse


class Trace(BaseModel):
    model: str | None = None  # so measured usage stays tied to its model after a prompt edit
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)


class Result(BaseModel):
    """One sample of one case on one version."""

    run_id: str
    case_id: str
    version_id: str
    sample_idx: int
    output: dict[str, Any] | None = None  # parsed structured output; None on error
    raw_output: str | None = None
    trace: Trace = Field(default_factory=Trace)
    error_kind: ErrorKind = ErrorKind.NONE
    error: str | None = None
    diverged: bool = False  # RUN-9 multi-turn only


class Score(BaseModel):
    """One check applied to one Result."""

    check: str
    kind: CheckKind
    passed: bool
    value: float | None = None  # 0-1 where the check has a graded score
    reason: str = ""
    sub_scores: list[dict[str, Any]] = Field(default_factory=list)  # e.g. one per claim
    judge_model: str | None = None
    prompt_hash: str | None = None


class Preference(BaseModel):
    """A pairwise judgment between a baseline (a) and candidate (b) output."""

    winner: Literal["a", "b", "tie"]
    reason: str = ""


class GatingUnit(BaseModel):
    """One judged question that can earn the right to gate (CAL-2, CAL-4).

    A judge check exposes one unit (faithfulness) or several (one per checklist
    item). A unit gates only when an approved Calibration exists for exactly this
    key, judge model and prompt hash; changing the judge model or the prompt or
    item text changes the hash and resets the unit to report-only.
    """

    key: str  # e.g. "faithfulness" or "checklist:testable_ac"
    judge_model: str
    prompt_hash: str  # sha256(template bytes + item text)[:12]


class Decision(str, Enum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    ROLLBACK = "ROLLBACK"
    INCOMPLETE = "INCOMPLETE"

    @property
    def exit_code(self) -> int:
        return {"PROMOTE": 0, "HOLD": 1, "ROLLBACK": 2, "INCOMPLETE": 3}[self.value]


# --- Plug-in protocols -------------------------------------------------------


@runtime_checkable
class Runner(Protocol):
    """Calls the system under test once for one case and one sample."""

    def run(
        self, case: Case, version: VersionManifest, sample_idx: int, run_id: str
    ) -> Result: ...


@runtime_checkable
class CheckContext(Protocol):
    """What a check may use besides the case and result."""

    suite_dir: Path

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:
        """Call the same version again with a transformed input (metamorphic checks)."""
        ...


@runtime_checkable
class Check(Protocol):
    name: str
    kind: CheckKind
    critical: bool  # a new failure blocks the build on its own

    def score(self, case: Case, result: Result, ctx: CheckContext) -> Score: ...


@runtime_checkable
class PairwiseCheck(Protocol):
    name: str

    def compare(self, case: Case, a: Result, b: Result) -> Preference: ...


@runtime_checkable
class Renderer(Protocol):
    """Builds constructed cases from a known answer."""

    def render(self, features: dict[str, Any], seed: int) -> Case: ...
