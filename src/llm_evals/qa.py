"""QA-2/QA-3/QA-4: dry-run estimates, A/A tests and seeded regressions."""

from __future__ import annotations

from dataclasses import dataclass

from collections import Counter
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel

from llm_evals.engine import ComparisonReport, run_comparison
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Check, Decision, Runner, VersionManifest


class Estimate(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    minutes: float
    assumptions: str


@dataclass(frozen=True)
class UsageHistory:
    """Measured usage per sample for one model: one case run once, every agent turn
    included. `cost_usd` is what those samples were billed at, per call, so it already
    reflects turns and long-prompt pricing."""

    input_tokens: float
    output_tokens: float
    cost_usd: float
    samples: int


def estimate(
    suite: SuiteConfig,
    baseline: VersionManifest,
    candidate: VersionManifest,
    cases: list[Case],
    checks: list[Check],
    *,
    samples: int | None = None,
    avg_output_tokens: int = 700,
    seconds_per_call: float = 6.0,
    max_concurrency: int = 8,
    pairwise: bool = False,
    judge_input_tokens: int = 2500,
    judge_output_tokens: int = 500,
    history: dict[str, UsageHistory] | None = None,
) -> Estimate:
    """Count the calls a comparison will make and price them. Makes no model calls.

    `history` maps a model to its measured usage per sample from past runs. A version
    whose model has history is priced from it; one without is a rough guess from the
    prompt's length (about 4 characters a token) and `avg_output_tokens`, priced per
    call. Run a few cases for real (`--limit 5`) and the next estimate is measured.
    """
    k = samples or suite.samples
    # Extra model calls some checks make per scored sample (invariance reruns a reworded input).
    reruns = sum(int(getattr(c, "reruns_per_sample", 0)) for c in checks)
    # An agent makes several model calls per sample (search, fetch, answer).
    turns = int(suite.runner.options.get("expected_turns", 3)) if suite.runner.type == "agent_replay" else 1
    env = Environment(undefined=StrictUndefined)
    calls = in_tok = out_tok = 0
    cost = 0.0
    measured: dict[str, UsageHistory] = {}
    guessed: set[str] = set()
    for version in (baseline, candidate):
        h = (history or {}).get(version.model)
        templates = {
            role: env.from_string(Path(path).read_text())
            for role, path in version.prompt_templates.items()
        }
        for case in cases:
            n = k * (1 + reruns)  # samples of this case on this version
            calls += n * turns
            if h is not None:
                measured[version.model] = h
                in_tok += round(h.input_tokens * n)
                out_tok += round(h.output_tokens * n)
                cost += h.cost_usd * n
                continue
            guessed.add(version.model)
            chars = sum(len(t.render(**case.input)) for t in templates.values())
            per_call_in = max(1, chars // 4)
            in_tok += per_call_in * n * turns
            out_tok += avg_output_tokens * n * turns
            # Priced per call: long-prompt pricing depends on one call's size, not a total.
            cost += cost_usd(version.model, per_call_in, avg_output_tokens) * n * turns
    # Judge calls: one per scored sample per judge check, plus two per case for pairwise.
    judge_model = suite.judge.get("model", "claude-sonnet-5-5")
    judge_checks = sum(1 for c in checks if getattr(c, "needs_judge", False))
    judge_calls = judge_checks * k * 2 * len(cases) + (2 * len(cases) if pairwise else 0)
    calls += judge_calls
    in_tok += judge_calls * judge_input_tokens
    out_tok += judge_calls * judge_output_tokens
    cost += cost_usd(judge_model, judge_input_tokens, judge_output_tokens) * judge_calls
    basis = [f"{m}: measured ${h.cost_usd:.5f} and {h.input_tokens:.0f} in / {h.output_tokens:.0f} out "
             f"tokens per sample from {h.samples} past samples" for m, h in sorted(measured.items())]
    basis += [f"{m}: no history, so a rough guess (~4 chars/token input, {avg_output_tokens} output "
              "tokens per call); run --limit 5 once to measure it" for m in sorted(guessed)]
    return Estimate(
        calls=calls,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=round(cost, 4),
        minutes=round(calls * seconds_per_call / max_concurrency / 60, 1),
        assumptions=(
            "; ".join(basis) + f"; {seconds_per_call:.0f}s/call, concurrency {max_concurrency}; "
            f"{judge_calls} judge calls on {judge_model} at ~{judge_input_tokens} in / "
            f"{judge_output_tokens} out tokens"
        ),
    )


class AARun(BaseModel):
    run_id: str
    decision: Decision
    delta: float
    ci_low: float
    ci_high: float
    mde: float | None
    check_deltas: dict[str, float]  # per-check pass-rate difference, candidate minus baseline
    cost_usd: float


class AAResult(BaseModel):
    runs: int
    decisions: dict[str, int]
    rollback_rate: float
    details: list[AARun] = []

    def noise(self) -> dict:
        """Summary of A/A noise: what a no-op change looks like."""
        import statistics as st

        if not self.details:
            return {}
        deltas = [d.delta for d in self.details]
        half = [(d.ci_high - d.ci_low) / 2 for d in self.details]
        low = sorted(-d.ci_low for d in self.details)
        mdes = [d.mde for d in self.details if d.mde is not None]
        return {
            "delta_mean": st.mean(deltas),
            "delta_sd": st.stdev(deltas) if len(deltas) > 1 else 0.0,
            "delta_range": [min(deltas), max(deltas)],
            "ci_half_width_median": st.median(half),
            "ci_low_p90": low[min(len(low) - 1, int(0.9 * len(low)))],
            "mde_median": st.median(mdes) if mdes else None,
            "cost_usd": sum(d.cost_usd for d in self.details),
        }


def aa_test(suite: SuiteConfig, version: VersionManifest, *, runs: int, store: Store,
            runner: Runner, **kwargs) -> AAResult:
    """QA-3: compare a version against itself repeatedly."""
    counts: Counter[str] = Counter()
    details: list[AARun] = []
    for i in range(runs):
        report = run_comparison(suite, version, version, store=store, runner=runner,
                                seed=i, kind="aa", **kwargs)
        counts[report.decision.value] += 1
        c = report.comparison
        details.append(AARun(
            run_id=report.run_id, decision=report.decision, delta=c.delta, ci_low=c.ci_low,
            ci_high=c.ci_high, mde=c.mde,
            check_deltas={k: report.candidate.check_pass_rates.get(k, 0.0) - v
                          for k, v in report.baseline.check_pass_rates.items()},
            cost_usd=report.baseline.cost_usd + report.candidate.cost_usd,
        ))
    return AAResult(runs=runs, decisions=dict(counts), details=details,
                    rollback_rate=counts[Decision.ROLLBACK.value] / runs if runs else 0.0)


class SeededResult(BaseModel):
    variant: str
    decision: Decision
    delta: float
    caught: bool


def seeded_test(suite: SuiteConfig, baseline: VersionManifest,
                variants: list[VersionManifest], *, store: Store, runner: Runner,
                **kwargs) -> list[tuple[SeededResult, ComparisonReport]]:
    """QA-4: each seeded (degraded) variant should be caught as ROLLBACK."""
    out = []
    for variant in variants:
        report = run_comparison(suite, baseline, variant, store=store, runner=runner,
                                kind="seeded", **kwargs)
        out.append((SeededResult(variant=variant.name, decision=report.decision,
                                 delta=report.comparison.delta,
                                 caught=report.decision == Decision.ROLLBACK), report))
    return out
