"""QA-1 simulated runner: known failure rates, for validating the statistics.

Each case fails with a known probability: `case.expected["p_fail"]` if set,
else `default_p_fail`, plus the version's `params["shift"]` (clamped to
[0, 1]). A version with shift > 0 is a regression of that many points.
`params["infra_error_rate"]` makes that share of samples INFRA errors.

Draws are a pure function of (seed, run_id, case id, version, sample index):
the uniforms come straight from a SHA-256 digest of those values, so results
are stable across processes and Python versions and need no per-sample RNG
setup. The same run_id reproduces a run exactly; a new run_id draws fresh
samples, the way calling a real model again would (A/A tests depend on this).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from llm_evals.registry import register_runner
from llm_evals.stats.paired import Outcomes
from llm_evals.types import Case, ErrorKind, Result, VersionManifest

_TWO_64 = float(2**64)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def version_key(version: VersionManifest) -> str:
    return version.id or version.name


def true_p_fail(case: Case, version: VersionManifest, default_p_fail: float = 0.1) -> float:
    """The failure probability the simulated runner uses for this case and version."""
    base = float((case.expected or {}).get("p_fail", default_p_fail))
    return _clamp01(base + float(version.params.get("shift", 0.0)))


@register_runner("simulated")
class SimulatedRunner:
    def __init__(self, seed: int = 0, default_p_fail: float = 0.1) -> None:
        if not 0.0 <= default_p_fail <= 1.0:
            raise ValueError(f"default_p_fail must be in [0, 1], got {default_p_fail}")
        self.seed = seed
        self.default_p_fail = default_p_fail

    def _uniforms(
        self, run_id: str, case_id: str, vkey: str, sample_idx: int
    ) -> tuple[float, float]:
        payload = "\x1f".join((str(self.seed), run_id, case_id, vkey, str(sample_idx)))
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        u_infra = int.from_bytes(digest[0:8], "big") / _TWO_64
        u_pass = int.from_bytes(digest[8:16], "big") / _TWO_64
        return u_infra, u_pass

    def run(
        self, case: Case, version: VersionManifest, sample_idx: int, run_id: str
    ) -> Result:
        vkey = version_key(version)
        u_infra, u_pass = self._uniforms(run_id, case.id, vkey, sample_idx)
        infra_rate = _clamp01(float(version.params.get("infra_error_rate", 0.0)))
        if u_infra < infra_rate:
            return Result(
                run_id=run_id,
                case_id=case.id,
                version_id=vkey,
                sample_idx=sample_idx,
                error_kind=ErrorKind.INFRA,
                error="simulated infra error",
            )
        p_fail = true_p_fail(case, version, self.default_p_fail)
        return Result(
            run_id=run_id,
            case_id=case.id,
            version_id=vkey,
            sample_idx=sample_idx,
            output={"passed": u_pass >= p_fail},
        )


def simulate_outcomes(
    cases: Iterable[Case],
    version: VersionManifest,
    k: int,
    runner: SimulatedRunner,
    run_id: str = "simulated",
) -> Outcomes:
    """Run k samples per case and return Outcomes, dropping INFRA samples.

    A case whose samples were all INFRA errors maps to an empty list; `compare`
    skips it. A non-infra result without a truthy `output["passed"]` is a fail.
    """
    outcomes: Outcomes = {}
    for case in cases:
        samples: list[bool] = []
        for i in range(k):
            r = runner.run(case, version, i, run_id)
            if r.error_kind is ErrorKind.INFRA:
                continue
            samples.append(bool(r.output and r.output.get("passed")))
        outcomes[case.id] = samples
    return outcomes
