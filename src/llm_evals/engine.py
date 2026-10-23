"""Orchestration: run a baseline and a candidate on the same cases and gate the change.

The flow follows the PRD:
1. Build the case set (loaded and rendered cases; held-out cases only on release runs).
2. Run both versions fresh in the same job, interleaved case by case (STAT-6).
3. Score every non-infra sample with the suite's checks; a case-sample passes when
   every gating check passes.
4. Compare with the paired cluster bootstrap and decide (STAT-1, STAT-2).
"""

from __future__ import annotations

import inspect
import statistics
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_evals import registry
from llm_evals.cases.loader import case_set_hash, load_cases
from llm_evals.runners.concurrency import run_all
from llm_evals.runners.errors import infra_share
from llm_evals.runners.manifest import load_manifest
from llm_evals.runners.schemas import is_schema_file
from llm_evals.stats import PairedComparison, compare, decide
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import (
    Case,
    Check,
    Decision,
    ErrorKind,
    GatingUnit,
    PairwiseCheck,
    Preference,
    Result,
    Runner,
    Score,
    VersionManifest,
)

AA_SUFFIX = "~aa"


# --- Building blocks ---------------------------------------------------------


def build_cases(suite: SuiteConfig, *, include_held_out: bool = False) -> list[Case]:
    cases: list[Case] = []
    for source in suite.cases:
        source = dict(source)
        if "from" in source:
            cases.extend(load_cases(suite.path(source["from"])))
        elif "render" in source:
            name = source.pop("render")
            n = source.pop("n")
            seed = source.pop("seed", 0)
            kwargs = {
                key: suite.path(value)
                if isinstance(value, str) and (suite.dir / value).exists()
                else value
                for key, value in source.items()
            }
            renderer = registry.get("renderer", name)(**kwargs)
            cases.extend(renderer.render_suite(n=n, seed=seed))
        else:
            raise ValueError(f"case source needs 'from' or 'render': {source}")
    cases = [c for c in cases if c.review_status == "approved"]
    if not include_held_out:
        cases = [c for c in cases if not c.held_out]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id across sources: {case.id}")
        seen.add(case.id)
    return cases


def _construct(factory: Callable[..., Any], params: dict[str, Any], suite: SuiteConfig) -> Any:
    """Call a plug-in factory, passing suite_dir when it asks for one."""
    params = dict(params)
    try:
        accepts_suite_dir = "suite_dir" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        accepts_suite_dir = False
    if accepts_suite_dir and "suite_dir" not in params:
        params["suite_dir"] = str(suite.dir)
    return factory(**params)


def _bind(plugin: Any, judge: Any | None, label: str) -> Any:
    if getattr(plugin, "needs_judge", False):
        if judge is None:
            raise ValueError(f"{label} needs a judge; pass a JudgeClient")
        plugin.bind_judge(judge)
    return plugin


def build_checks(suite: SuiteConfig, judge: Any | None = None) -> list[Check]:
    """Build the suite's checks. `gate: false` in a check's YAML params keeps it
    scored and reported but out of the gate."""
    checks = []
    for c in suite.checks:
        params = dict(c.params)
        gate = params.pop("gate", True)
        check = _bind(_construct(registry.get("check", c.name), params, suite), judge, c.name)
        if gate is False:
            setattr(check, "report_only", True)
        checks.append(check)
    return checks


def build_pairwise(suite: SuiteConfig, judge: Any | None = None) -> PairwiseCheck | None:
    if suite.pairwise is None:
        return None
    params = dict(suite.pairwise)
    name = params.pop("type", "pairwise")
    return _bind(_construct(registry.get("pairwise", name), params, suite), judge, name)


def build_runner(suite: SuiteConfig, provider: Any | None = None,
                 tool_provider: Any | None = None) -> Runner:
    factory = registry.get("runner", suite.runner.type)
    options = dict(suite.runner.options)
    schema = options.get("output_schema")
    if isinstance(schema, str) and is_schema_file(schema) and not Path(schema).is_absolute():
        options["output_schema"] = str(suite.dir / schema)  # schema files sit in the suite folder
    if suite.runner.type == "prompt":
        if provider is None:
            raise ValueError("the prompt runner needs a provider")
        return factory(provider=provider, **options)
    if suite.runner.type == "agent_replay":
        if tool_provider is None:
            raise ValueError("the agent_replay runner needs a tool provider")
        return factory(tool_provider=tool_provider, **options)
    return factory(**options)


def approved_units(checks: list[Check], store: Store) -> dict[str, set[str] | None]:
    """check name -> approved gating unit keys, or None for checks that always gate.

    Judge checks (those exposing gating_units) gate only through units with an
    approved calibration for the exact (key, judge model, prompt hash) triple.
    """
    out: dict[str, set[str] | None] = {}
    approved_fn = getattr(store, "approved_calibration", None)
    for check in checks:
        if getattr(check, "report_only", False):
            out[check.name] = set()  # configured report-only: never gates
            continue
        units_fn = getattr(check, "gating_units", None)
        if units_fn is None:
            out[check.name] = None
            continue
        units: list[GatingUnit] = units_fn()
        out[check.name] = {
            u.key for u in units
            if approved_fn is not None and approved_fn(u.key, u.judge_model, u.prompt_hash)
        }
    return out


def _not_applicable(score: Score) -> bool:
    """A check that doesn't apply to this case (e.g. bundles_split on a single ask)."""
    return any(isinstance(s, dict) and s.get("applicable") is False for s in score.sub_scores)


def gating_pass(score: Score, approved: set[str] | None) -> bool | None:
    """Whether a score counts toward the case pass: its verdict, or None if report-only."""
    if approved is None:
        return score.passed
    if not approved:
        return None
    unit_verdicts = [s.get("passed") for s in score.sub_scores
                     if isinstance(s, dict) and s.get("unit") in approved]
    if not unit_verdicts:
        # A judge error leaves no unit entries: count it as a failure of an approved unit.
        return False if score.reason.startswith("judge error") else None
    return all(bool(v) for v in unit_verdicts)


def load_version(suite: SuiteConfig, manifest_path: str | Path) -> VersionManifest:
    return load_manifest(suite.path(str(manifest_path)), suite.dir)


class StoredRunner:
    """CHK-6: serve a stored run's outputs instead of calling the system again.

    Re-scoring a run with new or changed checks, or under new gate rules, makes no
    model calls. A sample the stored run never produced comes back as an INFRA
    error, so it is excluded rather than counted.
    """

    def __init__(self, store: Store, source_run_id: str) -> None:
        self.store = store
        self.source_run_id = source_run_id
        self.misses = 0

    def run(self, case: Case, version: VersionManifest, sample_idx: int, run_id: str) -> Result:
        stored = self.store.get_result(self.source_run_id, version.id, case.id, sample_idx)
        if stored is None:
            self.misses += 1
            return Result(run_id=run_id, case_id=case.id, version_id=version.id,
                          sample_idx=sample_idx, error_kind=ErrorKind.INFRA,
                          error=f"not in stored run {self.source_run_id}")
        return stored.model_copy(update={"run_id": run_id})


def find_manifest(suite: SuiteConfig, version_id: str) -> VersionManifest | None:
    """The suite's version manifest whose computed id matches, if any."""
    for path in sorted((suite.dir / "versions").glob("*.yaml")):
        manifest = load_manifest(path, suite.dir)
        if manifest.id == version_id.removesuffix(AA_SUFFIX):
            return manifest
    return None


@dataclass
class _Ctx:
    """CheckContext for metamorphic checks: reruns the same version on a new input."""

    suite_dir: Path
    runner: Runner
    version: VersionManifest
    run_id: str
    sample_idx: int
    store: Store
    reruns: list[Result] = field(default_factory=list)

    def rerun(self, case: Case, new_input: dict[str, Any]) -> Result:
        rerun_case = case.model_copy(update={"id": f"{case.id}~rerun", "input": new_input})
        cached = self.store.get_result(self.run_id, self.version.id, rerun_case.id, self.sample_idx)
        if cached is not None:
            return cached
        result = self.runner.run(rerun_case, self.version, self.sample_idx, self.run_id)
        self.store.save_result(result)
        self.reruns.append(result)
        return result


# --- Report models -----------------------------------------------------------


class ArmStats(BaseModel):
    version_id: str
    version_name: str
    samples: int
    infra_errors: int
    system_errors: int
    infra_share: float
    pass_rate: float  # mean of per-case rates
    check_pass_rates: dict[str, float]
    cost_usd: float
    p95_latency_ms: float


class FlipDetail(BaseModel):
    case_id: str
    direction: str
    baseline_rate: float
    candidate_rate: float
    strata: dict[str, str] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    baseline_output: Any = None
    candidate_output: Any = None
    held_out: bool = False


class CriticalConfirmation(BaseModel):
    """A unanimous critical flip and the reruns that confirmed or cleared it."""

    case_id: str
    check: str
    baseline_passes: int
    baseline_n: int
    candidate_passes: int
    candidate_n: int
    p_value: float
    confirmed: bool


class CheckRegression(BaseModel):
    """STAT-9: a gating check whose own paired Δ is clearly negative."""

    check: str
    delta: float
    ci_low: float
    ci_high: float
    alpha: float  # Bonferroni-adjusted


class PairwiseSummary(BaseModel):
    """CHK-7, report-only in v1: judged on each case's first sample, both orders."""

    cases: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    candidate_win_share: float


class ComparisonReport(BaseModel):
    suite: str
    run_id: str
    decision: Decision
    exit_code: int
    tolerance: float
    case_set_hash: str
    n_cases: int
    samples_per_case: int
    baseline: ArmStats
    candidate: ArmStats
    comparison: PairedComparison
    critical_new_failures: list[str]
    check_regressions: list[CheckRegression] = Field(default_factory=list)
    critical_confirmations: list[CriticalConfirmation] = Field(default_factory=list)
    flips: list[FlipDetail]
    warnings: list[str]
    stopped_by_cost_cap: bool
    gating: dict[str, list[str] | str] = Field(default_factory=dict)  # check -> "always" | approved units
    report_only: list[str] = Field(default_factory=list)  # checks or units scored but not gating
    pairwise: PairwiseSummary | None = None
    judge_cost_usd: float = 0.0


# --- The comparison ----------------------------------------------------------


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _majority(values: list[bool]) -> bool:
    return bool(values) and sum(values) / len(values) > 0.5


def run_comparison(
    suite: SuiteConfig,
    baseline: VersionManifest,
    candidate: VersionManifest,
    *,
    store: Store,
    runner: Runner,
    checks: list[Check] | None = None,
    cases: list[Case] | None = None,
    samples: int | None = None,
    max_concurrency: int = 8,
    cost_cap_usd: float | None = None,
    seed: int = 0,
    n_boot: int = 10_000,
    kind: str = "compare",
    on_progress: Callable[..., None] | None = None,
    judge: Any | None = None,
    pairwise: PairwiseCheck | None = None,
    run_id: str | None = None,
) -> ComparisonReport:
    checks = build_checks(suite, judge) if checks is None else checks
    judge_cost_start = float(getattr(judge, "total_cost_usd", 0.0) or 0.0)
    approved = approved_units(checks, store)
    cases = build_cases(suite) if cases is None else cases
    k = samples or suite.samples
    if baseline.id == candidate.id:  # A/A: keep the two arms' samples apart
        candidate = candidate.model_copy(update={"id": candidate.id + AA_SUFFIX})
    by_id = {c.id: c for c in cases}
    set_hash = case_set_hash(cases)

    run_id = store.create_run(
        suite=suite.suite,
        kind=kind,
        config_hash="",
        case_set_hash=set_hash,
        baseline_version_id=baseline.id,
        candidate_version_id=candidate.id,
        run_id=run_id,
    )
    if hasattr(store, "save_run_cases"):  # lets `eval calibration label` show inputs
        store.save_run_cases(run_id, cases)

    # STAT-6: both arms fresh, interleaved case by case.
    jobs: list[Callable[[], Result]] = []
    for case in cases:
        for s in range(k):
            for version in (baseline, candidate):
                jobs.append(_job(runner, store, case, version, s, run_id))
    total = len(jobs)
    done = 0

    def _progress(_: Result) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(done, total, "run")

    outcome = run_all(
        jobs, max_concurrency=max_concurrency, cost_cap_usd=cost_cap_usd, on_result=_progress
    )

    # Score. Samples are scored in parallel so metamorphic reruns and judge calls
    # share the concurrency pool instead of running one at a time.
    arm_results: dict[str, list[Result]] = {baseline.id: [], candidate.id: []}
    for result in outcome.results:
        arm_results[result.version_id].append(result)
    version_by_id = {baseline.id: baseline, candidate.id: candidate}

    to_score = [
        r for v in arm_results for r in sorted(arm_results[v], key=lambda r: (r.case_id, r.sample_idx))
        if r.error_kind != ErrorKind.INFRA
    ]
    scored_count = 0
    scored_lock = threading.Lock()

    def _score_one(result: Result) -> tuple[Result, list[Score], list[Result]]:
        nonlocal scored_count
        ctx = _Ctx(suite.dir, runner, version_by_id[result.version_id], run_id,
                   result.sample_idx, store)
        scores = [_score(check, by_id[result.case_id], result, ctx) for check in checks]
        with scored_lock:
            scored_count += 1
            n = scored_count
        if on_progress and (n % 20 == 0 or n == len(to_score)):
            on_progress(n, len(to_score), "scored")
        return result, scores, ctx.reruns

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        scored = list(pool.map(_score_one, to_score))

    # case -> check -> list[bool] per arm (reporting), case -> check -> gating verdicts,
    # and case -> list[bool] overall (the primary metric).
    per_check: dict[str, dict[str, dict[str, list[bool]]]] = {
        v: defaultdict(lambda: defaultdict(list)) for v in arm_results
    }
    gating_by_check: dict[str, dict[str, dict[str, list[bool]]]] = {
        v: defaultdict(lambda: defaultdict(list)) for v in arm_results
    }
    outcomes: dict[str, dict[str, list[bool]]] = {v: defaultdict(list) for v in arm_results}
    failed_checks: dict[str, dict[str, set[str]]] = {v: defaultdict(set) for v in arm_results}
    reruns: list[Result] = []

    for result, scores, sample_reruns in scored:
        reruns.extend(sample_reruns)
        version_id, case_id = result.version_id, result.case_id
        if any(s.reason.startswith("infra:") for s in scores):
            continue  # a rerun or judge call hit an infra error; exclude the sample
        verdicts: list[bool] = []
        for check, score in zip(checks, scores):
            store.save_score(run_id, version_id, case_id, result.sample_idx, score)
            if _not_applicable(score):
                continue  # measured only on the cases it applies to
            per_check[version_id][case_id][score.check].append(score.passed)
            if not score.passed:
                failed_checks[version_id][case_id].add(score.check)
            verdict = gating_pass(score, approved.get(check.name))
            if verdict is not None:
                verdicts.append(verdict)
                gating_by_check[version_id][case_id][check.name].append(verdict)
        outcomes[version_id][case_id].append(all(verdicts))

    cmp = compare(dict(outcomes[baseline.id]), dict(outcomes[candidate.id]), n_boot=n_boot, seed=seed)

    critical = {c.name: c for c in checks
                if getattr(c, "critical", False) and not getattr(c, "report_only", False)}
    flagged: list[tuple[str, str]] = []
    for case_id in sorted(set(per_check[baseline.id]) & set(per_check[candidate.id])):
        for name in critical:
            before = per_check[baseline.id][case_id].get(name, [])
            after = per_check[candidate.id][case_id].get(name, [])
            # A flip is flagged only when unanimous: every baseline sample passed and
            # every candidate sample failed.
            if before and after and all(before) and not any(after):
                flagged.append((case_id, name))
    # With k=3, a check that is noisy on some cases flips one of them unanimously
    # by chance (29.5% of A/A runs on 120 cases in simulation), so each flagged
    # flip is confirmed: rerun that case confirm_k more times per arm and fire only
    # if a one-sided Fisher exact test, Bonferroni-split over flagged flips, agrees.
    critical_new: list[str] = []
    confirmations: list[CriticalConfirmation] = []
    if flagged and suite.gate.critical_confirm_k > 0:
        confirmations = _confirm_critical(
            flagged, critical, by_id, runner, store, run_id, baseline, candidate, k,
            suite.gate.critical_confirm_k, suite.gate.critical_alpha, max_concurrency, suite.dir,
        )
        critical_new = [f"{c.case_id}:{c.check}" for c in confirmations if c.confirmed]
    elif flagged:
        critical_new = [f"{cid}:{name}" for cid, name in flagged]

    arm_stats = {
        v: _arm_stats(version_by_id[v], arm_results[v] + [r for r in reruns if r.version_id == v],
                      outcomes[v], per_check[v])
        for v in arm_results
    }
    incomplete = outcome.stopped_by_cost_cap or any(
        a.infra_share > suite.gate.max_infra_share for a in arm_stats.values()
    )
    decision = decide(cmp, tolerance=suite.gate.tolerance,
                      critical_new_failures=critical_new, incomplete=incomplete)

    # STAT-9: a regression in one gating check can be masked in the case pass rate
    # by gains in another, so each gating check also gets its own paired test.
    # The per-check tests share an error budget (gate.per_check_alpha, Bonferroni-split
    # across checks) so the A/A false-rollback rate stays near the PRD's 1-in-20 bar.
    gating_names = [c.name for c in checks
                    if any(gating_by_check[v][cid].get(c.name)
                           for v in gating_by_check for cid in gating_by_check[v])]
    check_regressions: list[CheckRegression] = []
    if gating_names:
        per_alpha = suite.gate.per_check_alpha / len(gating_names)
        for name in gating_names:
            base_o = {cid: checks_ for cid, d in gating_by_check[baseline.id].items()
                      if (checks_ := d.get(name))}
            cand_o = {cid: checks_ for cid, d in gating_by_check[candidate.id].items()
                      if (checks_ := d.get(name))}
            c_cmp = compare(base_o, cand_o, n_boot=n_boot, seed=seed, alpha=per_alpha)
            if c_cmp.n_cases and c_cmp.ci_high < 0:
                check_regressions.append(CheckRegression(
                    check=name, delta=c_cmp.delta, ci_low=c_cmp.ci_low, ci_high=c_cmp.ci_high,
                    alpha=per_alpha))
    if check_regressions and decision != Decision.INCOMPLETE:
        decision = Decision.ROLLBACK

    warnings: list[str] = []
    # Only a HOLD needs explaining: PROMOTE and ROLLBACK were decided despite the spread.
    # The figure is this run's own spread, which a large or uneven real change inflates,
    # so it isn't the suite's noise floor; an A/A run measures that.
    if decision == Decision.HOLD and cmp.mde is not None and cmp.mde > suite.gate.tolerance:
        warnings.append(
            f"This run couldn't tell: it could only reliably detect a drop of {cmp.mde:.1%}, more "
            f"than the {suite.gate.tolerance:.1%} tolerance. The figure comes from this run's own "
            "spread, which a large or uneven change inflates; `llm-evals aa` measures the noise "
            "alone. To resolve a change this size, add cases or raise samples."
        )
    for v, a in arm_stats.items():
        if a.infra_share > 0:
            warnings.append(f"{a.version_name}: {a.infra_share:.1%} of samples hit infra errors.")
    if outcome.stopped_by_cost_cap:
        warnings.append("The cost cap stopped the run before every sample finished.")

    outputs = {
        (r.version_id, r.case_id): r.output
        for r in outcome.results
        if r.sample_idx == 0
    }
    first = {
        (r.version_id, r.case_id): r
        for r in outcome.results
        if r.sample_idx == 0 and r.error_kind == ErrorKind.NONE and r.output is not None
    }
    pairwise_summary = None
    if pairwise is not None:
        prefs: list[Preference] = []
        for case in cases:
            a = first.get((baseline.id, case.id))
            b = first.get((candidate.id, case.id))
            if a is None or b is None:
                continue
            try:
                prefs.append(pairwise.compare(case, a, b))
            except Exception as exc:  # a judge failure must not sink the run
                warnings.append(f"pairwise judge failed on {case.id}: {type(exc).__name__}")
        wins = sum(p.winner == "b" for p in prefs)
        losses = sum(p.winner == "a" for p in prefs)
        pairwise_summary = PairwiseSummary(
            cases=len(prefs), candidate_wins=wins, baseline_wins=losses,
            ties=len(prefs) - wins - losses,
            candidate_win_share=wins / len(prefs) if prefs else 0.0,
        )

    gating_info: dict[str, list[str] | str] = {}
    report_only: list[str] = []
    for check in checks:
        units = approved.get(check.name)
        if units is None:
            gating_info[check.name] = "always"
            continue
        if getattr(check, "report_only", False):
            gating_info[check.name] = "report-only (configured)"
            report_only.append(check.name)
            continue
        gating_info[check.name] = sorted(units)
        all_units = [u.key for u in check.gating_units()]  # type: ignore[attr-defined]
        report_only.extend(u for u in all_units if u not in units)
    if report_only:
        warnings.append(
            "Reported but not gating (configured report-only, or not calibrated yet): "
            + ", ".join(sorted(report_only)) + "."
        )
    flips = [
        FlipDetail(
            case_id=f.case_id,
            direction=f.direction,
            baseline_rate=f.baseline_rate,
            candidate_rate=f.candidate_rate,
            strata=by_id[f.case_id].strata,
            failed_checks=sorted(
                failed_checks[candidate.id if f.direction == "broke" else baseline.id][f.case_id]
            ),
            baseline_output=outputs.get((baseline.id, f.case_id)),
            candidate_output=outputs.get((candidate.id, f.case_id)),
            held_out=by_id[f.case_id].held_out,
        )
        for f in cmp.flips
    ]

    report = ComparisonReport(
        suite=suite.suite,
        run_id=run_id,
        decision=decision,
        exit_code=decision.exit_code,
        tolerance=suite.gate.tolerance,
        case_set_hash=set_hash,
        n_cases=cmp.n_cases,
        samples_per_case=k,
        baseline=arm_stats[baseline.id],
        candidate=arm_stats[candidate.id],
        comparison=cmp,
        critical_new_failures=critical_new,
        check_regressions=check_regressions,
        critical_confirmations=confirmations,
        flips=flips,
        warnings=warnings,
        stopped_by_cost_cap=outcome.stopped_by_cost_cap,
        gating=gating_info,
        report_only=sorted(report_only),
        pairwise=pairwise_summary,
        judge_cost_usd=round(
            float(getattr(judge, "total_cost_usd", 0.0) or 0.0) - judge_cost_start, 6
        ),
    )
    store.update_run(run_id, status="incomplete" if incomplete else "complete",
                     decision=decision.value)
    return report


def _confirm_critical(flagged, critical, by_id, runner, store, run_id, baseline, candidate,
                      k, confirm_k, alpha, max_concurrency, suite_dir) -> list[CriticalConfirmation]:
    from scipy.stats import fisher_exact

    def one(item: tuple[str, str, VersionManifest, int]) -> tuple[str, str, str, bool]:
        case_id, name, version, idx = item
        case = by_id[case_id]
        result = store.get_result(run_id, version.id, case_id, idx)
        if result is None:
            result = runner.run(case, version, idx, run_id)
            store.save_result(result)
        if result.error_kind == ErrorKind.INFRA:
            return case_id, name, version.id, None  # excluded
        ctx = _Ctx(suite_dir, runner, version, run_id, idx, store)
        return case_id, name, version.id, _score(critical[name], case, result, ctx).passed

    jobs = [(cid, name, v, k + j) for cid, name in flagged
            for v in (baseline, candidate) for j in range(confirm_k)]
    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        verdicts = list(pool.map(one, jobs))
    tally: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for cid, name, vid, ok in verdicts:
        if ok is not None:
            tally[(cid, name, vid)].append(ok)
    out = []
    per_alpha = alpha / len(flagged)
    for cid, name in flagged:
        b, c = tally[(cid, name, baseline.id)], tally[(cid, name, candidate.id)]
        if not b or not c:
            out.append(CriticalConfirmation(case_id=cid, check=name, baseline_passes=sum(b),
                                            baseline_n=len(b), candidate_passes=sum(c),
                                            candidate_n=len(c), p_value=1.0, confirmed=False))
            continue
        table = [[sum(b), len(b) - sum(b)], [sum(c), len(c) - sum(c)]]
        _, p = fisher_exact(table, alternative="greater")  # baseline passes more often
        out.append(CriticalConfirmation(case_id=cid, check=name, baseline_passes=sum(b),
                                        baseline_n=len(b), candidate_passes=sum(c),
                                        candidate_n=len(c), p_value=float(p),
                                        confirmed=bool(p < per_alpha)))
    return out


def _job(runner: Runner, store: Store, case: Case, version: VersionManifest,
         sample_idx: int, run_id: str) -> Callable[[], Result]:
    def job() -> Result:
        cached = store.get_result(run_id, version.id, case.id, sample_idx)
        if cached is not None:
            return cached
        result = runner.run(case, version, sample_idx, run_id)
        store.save_result(result)
        return result

    return job


def _score(check: Check, case: Case, result: Result, ctx: _Ctx) -> Score:
    try:
        return check.score(case, result, ctx)
    except Exception as exc:  # a broken check must not crash the run
        return Score(check=check.name, kind=check.kind, passed=False,
                     reason=f"check raised {type(exc).__name__}: {exc}")


def _arm_stats(version: VersionManifest, results: list[Result],
               outcomes: dict[str, list[bool]],
               per_check: dict[str, dict[str, list[bool]]]) -> ArmStats:
    rates = [sum(v) / len(v) for v in outcomes.values() if v]
    check_rates: dict[str, list[float]] = defaultdict(list)
    for case_checks in per_check.values():
        for name, values in case_checks.items():
            check_rates[name].append(sum(values) / len(values))
    return ArmStats(
        version_id=version.id,
        version_name=version.name,
        samples=len(results),
        infra_errors=sum(r.error_kind == ErrorKind.INFRA for r in results),
        system_errors=sum(r.error_kind == ErrorKind.SYSTEM for r in results),
        infra_share=infra_share(results) if results else 0.0,
        pass_rate=sum(rates) / len(rates) if rates else 0.0,
        check_pass_rates={k: sum(v) / len(v) for k, v in sorted(check_rates.items())},
        cost_usd=round(sum(r.trace.cost_usd for r in results), 6),
        p95_latency_ms=_p95([r.trace.latency_ms for r in results if r.trace.latency_ms]),
    )
