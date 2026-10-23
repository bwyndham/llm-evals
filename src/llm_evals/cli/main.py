"""The `eval` command line (INT-1)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from llm_evals import engine, qa
from llm_evals.report.markdown import render_markdown, write_reports
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig, load_suite

def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from the repo's gitignored .env without overriding the shell."""
    import os

    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parents[3] / ".env")

app = typer.Typer(help="Paired regression evals with a PROMOTE / HOLD / ROLLBACK gate.",
                  no_args_is_help=True)


def _store(suite: SuiteConfig, db: Optional[Path]) -> Store:
    path = db or suite.dir / ".evals" / "evals.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return Store(path)


PROVIDER_HELP = "anthropic (real model calls) or mock (offline demo mode: rule-based, no API key)"


class _Env:
    """The runner, judge, checks and pairwise check for one suite and provider."""

    def __init__(self, suite: SuiteConfig, provider: str, skip_judges: bool = False) -> None:
        if provider not in ("anthropic", "mock"):
            raise typer.BadParameter(f"--provider must be anthropic or mock, not {provider!r}")
        self.offline = provider == "mock"
        if skip_judges:
            # Judge checks are report-only until calibrated; skipping them cuts most of the spend.
            suite = suite.model_copy(update={
                "checks": [c for c in suite.checks if c.name not in _JUDGE_CHECKS],
                "pairwise": None,
            })
        text_provider = tool_provider = None
        if self.offline:
            from llm_evals.runners.providers.mock import MockProvider
            from llm_evals.runners.providers.mock_research import MockResearchModel

            responder = None
            if suite.mock:
                from llm_evals.registry import load_callable

                responder = load_callable(suite.mock, suite.dir)
            text_provider = MockProvider(responder=responder)
            if suite.runner.type == "agent_replay":
                tool_provider = MockResearchModel(
                    memory_path=suite.dir / "data" / "world_facts.jsonl"
                )
        else:
            from llm_evals.runners.providers.anthropic import AnthropicProvider

            text_provider = AnthropicProvider()
            if suite.runner.type == "agent_replay":
                from llm_evals.runners.providers.anthropic_tools import AnthropicToolProvider

                tool_provider = AnthropicToolProvider()
        self.runner = engine.build_runner(suite, text_provider, tool_provider)
        self.judge = None
        if any(c.name in _JUDGE_CHECKS for c in suite.checks) or suite.pairwise is not None:
            from llm_evals.judges.client import JudgeClient

            cache = suite.dir / ".evals" / ("judge_cache_mock.sqlite" if self.offline
                                            else "judge_cache.sqlite")
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.judge = JudgeClient(text_provider, model=suite.judge.get("model", "claude-sonnet-5-5"),
                                     params=suite.judge.get("params"), cache_path=cache)
        self.checks = engine.build_checks(suite, self.judge)
        self.pairwise = engine.build_pairwise(suite, self.judge)

    def annotate(self, report) -> None:
        if self.offline:
            report.warnings.insert(0, "Offline demo mode: outputs come from a rule-based mock, so "
                                      "this shows the pipeline working, not model quality. Costs "
                                      "are notional.")

    def kwargs(self) -> dict:
        return {"runner": self.runner, "checks": self.checks, "judge": self.judge,
                "pairwise": self.pairwise}


# Checks that need a judge; the engine binds one when present.
_JUDGE_CHECKS = {"faithfulness", "checklist", "claim_support"}


def _baseline(suite: SuiteConfig, store: Store, baseline: Optional[str],
              baseline_dir: Optional[Path] = None):
    if baseline_dir and not baseline:
        # CI: the base branch's own suite.yaml says which manifest is its baseline.
        from llm_evals.suite import baseline_of

        baseline = baseline_of(baseline_dir)  # not load_suite: its plug-ins would clash with the PR's
    if baseline and baseline_dir:
        # CI: the baseline comes from the base branch's checkout of the same suite.
        from llm_evals.runners.manifest import load_manifest

        return load_manifest(baseline_dir.resolve() / baseline, baseline_dir.resolve())
    if baseline:
        return engine.load_version(suite, baseline)
    current = store.current_baseline(suite.suite)
    if current is not None:
        return engine.load_version(suite, current["manifest_path"])
    return engine.load_version(suite, suite.baseline)


def _progress(done: int, total: int, phase: str = "run") -> None:
    if done == total or done % 20 == 0:
        label = "samples run" if phase == "run" else "samples scored"
        typer.echo(f"  {done}/{total} {label}", err=True)


@app.command()
def compare(
    suite_path: Path = typer.Argument(..., help="Suite folder or suite.yaml"),
    candidate: Optional[str] = typer.Option(
        None, help="Candidate manifest, relative to the suite (default: the suite's baseline "
                   "manifest as this checkout has it, e.g. the PR's version)"),
    baseline: Optional[str] = typer.Option(None, help="Baseline manifest (default: promoted)"),
    baseline_dir: Optional[Path] = typer.Option(
        None, help="Suite folder the baseline manifest resolves against (e.g. a base-branch checkout)"
    ),
    samples: Optional[int] = typer.Option(None, help="Override samples per case"),
    limit: Optional[int] = typer.Option(None, help="Use only the first N cases"),
    concurrency: int = typer.Option(8),
    cost_cap: Optional[float] = typer.Option(None, help="Stop when spend passes this (USD)"),
    out: Optional[Path] = typer.Option(None, help="Write report.json/report.md/comment.md here"),
    db: Optional[Path] = typer.Option(None),
    seed: int = typer.Option(0),
    provider: str = typer.Option("anthropic", help=PROVIDER_HELP),
    skip_judges: bool = typer.Option(False, help="Leave out LLM judge checks and pairwise (cheaper)"),
    skip_unchanged: bool = typer.Option(
        False, help="Exit 0 without running when the candidate is the same version as the "
                    "baseline (e.g. a PR that doesn't touch the prompt)"),
) -> None:
    """Run baseline and candidate on the same cases and print the gate decision."""
    suite = load_suite(suite_path)
    store = _store(suite, db)
    base_version = _baseline(suite, store, baseline, baseline_dir)
    cand_version = engine.load_version(suite, candidate or suite.baseline)
    if skip_unchanged and cand_version.id == base_version.id:
        from llm_evals.report.markdown import render_unchanged

        text = render_unchanged(suite.suite, cand_version)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / "comment.md").write_text(text)
            (out / "report.md").write_text(text)
        typer.echo(text)
        raise typer.Exit(0)
    env = _Env(suite, provider, skip_judges)
    cases = engine.build_cases(suite)[:limit] if limit else engine.build_cases(suite)
    report = engine.run_comparison(
        suite, base_version, cand_version,
        store=store, cases=cases, samples=samples, max_concurrency=concurrency,
        cost_cap_usd=cost_cap, seed=seed, on_progress=_progress, **env.kwargs(),
    )
    env.annotate(report)
    if out:
        paths = write_reports(report, out)
        typer.echo(f"Wrote {', '.join(str(p) for p in paths.values())}", err=True)
    typer.echo(render_markdown(report))
    raise typer.Exit(report.exit_code)


app.command("gate", help="Alias of compare: exits 0/1/2/3 for PROMOTE/HOLD/ROLLBACK/INCOMPLETE.")(
    compare
)


def _usage_history(store: Store, suite: SuiteConfig) -> dict[str, qa.UsageHistory]:
    """Measured usage per sample, by model, from this suite's stored real runs.

    Results record their model; older ones are matched to a version file by id, or,
    when every version in the suite uses one model, to that model.
    """
    import sqlite3

    from llm_evals.types import Result

    try:
        rows = store._read("SELECT result_json FROM results")  # noqa: SLF001
    except (AttributeError, sqlite3.Error):
        return {}
    manifests = [engine.load_manifest(p, suite.dir) for p in sorted((suite.dir / "versions").glob("*.yaml"))]
    by_id = {m.id: m.model for m in manifests}
    only_model = manifests[0].model if manifests and len({m.model for m in manifests}) == 1 else None
    sums: dict[str, list[float]] = {}
    for (blob,) in rows:
        r = Result.model_validate_json(blob)
        t = r.trace
        if not t.output_tokens or (t.request_ids and str(t.request_ids[0]).startswith("mock-")):
            continue  # failed before the model answered, or the offline stand-in
        model = t.model or by_id.get(r.version_id.removesuffix(engine.AA_SUFFIX)) or only_model
        if model is None:
            continue
        acc = sums.setdefault(model, [0.0, 0.0, 0.0, 0])
        acc[0] += t.input_tokens
        acc[1] += t.output_tokens
        acc[2] += t.cost_usd
        acc[3] += 1
    return {m: qa.UsageHistory(i / n, o / n, c / n, int(n)) for m, (i, o, c, n) in sums.items() if n >= 20}


@app.command()
def estimate(
    suite_path: Path,
    candidate: Optional[str] = typer.Option(None),
    baseline: Optional[str] = typer.Option(None),
    samples: Optional[int] = typer.Option(None),
    limit: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
    db: Optional[Path] = typer.Option(None),
    skip_judges: bool = typer.Option(False, help="Leave out LLM judge checks and pairwise (cheaper)"),
) -> None:
    """Count calls and estimate cost and time. Makes no model calls."""
    suite = load_suite(suite_path)
    store = _store(suite, db)
    base = _baseline(suite, store, baseline)
    cand = engine.load_version(suite, candidate) if candidate else base
    cases = engine.build_cases(suite)
    cases = cases[:limit] if limit else cases
    env = _Env(suite, "mock", skip_judges)  # builds checks without touching the network
    history = _usage_history(store, suite)
    est = qa.estimate(suite, base, cand, cases, env.checks, samples=samples,
                      max_concurrency=concurrency, pairwise=env.pairwise is not None,
                      history=history)
    typer.echo(f"{len(cases)} cases → {est.calls} calls, ~{est.input_tokens:,} input / "
               f"~{est.output_tokens:,} output tokens, ~${est.cost_usd:.2f}, ~{est.minutes} min")
    typer.echo(f"Assumes {est.assumptions}.")


@app.command()
def aa(
    suite_path: Path,
    runs: int = typer.Option(20),
    baseline: Optional[str] = typer.Option(None),
    samples: Optional[int] = typer.Option(None),
    limit: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
    db: Optional[Path] = typer.Option(None),
    provider: str = typer.Option("anthropic", help=PROVIDER_HELP),
    skip_judges: bool = typer.Option(False, help="Leave out LLM judge checks and pairwise (cheaper)"),
) -> None:
    """Compare a version against itself and count each decision."""
    suite = load_suite(suite_path)
    store = _store(suite, db)
    env = _Env(suite, provider, skip_judges)
    cases = engine.build_cases(suite)
    cases = cases[:limit] if limit else cases
    result = qa.aa_test(suite, _baseline(suite, store, baseline), runs=runs, store=store,
                        cases=cases, samples=samples, max_concurrency=concurrency,
                        on_progress=_progress, **env.kwargs())
    typer.echo(result.model_dump_json(indent=2))
    typer.echo("noise: " + __import__("json").dumps(result.noise(), indent=2))


@app.command()
def seeded(
    suite_path: Path,
    baseline: Optional[str] = typer.Option(None),
    samples: Optional[int] = typer.Option(None),
    limit: Optional[int] = typer.Option(None),
    concurrency: int = typer.Option(8),
    db: Optional[Path] = typer.Option(None),
    provider: str = typer.Option("anthropic", help=PROVIDER_HELP),
    skip_judges: bool = typer.Option(False, help="Leave out LLM judge checks and pairwise (cheaper)"),
) -> None:
    """Run each seeded (degraded) variant against the baseline."""
    suite = load_suite(suite_path)
    store = _store(suite, db)
    env = _Env(suite, provider, skip_judges)
    cases = engine.build_cases(suite)
    cases = cases[:limit] if limit else cases
    variants = [engine.load_version(suite, v) for v in suite.seeded_variants]
    results = qa.seeded_test(suite, _baseline(suite, store, baseline), variants, store=store,
                             cases=cases, samples=samples, max_concurrency=concurrency,
                             on_progress=_progress, **env.kwargs())
    for res, _ in results:
        typer.echo(f"{res.variant}: {res.decision.value} (Δ {res.delta:+.1%})"
                   f"{'' if res.caught else '  ← not caught'}")
    caught = sum(r.caught for r, _ in results)
    typer.echo(f"Caught {caught} of {len(results)}.")


@app.command()
def rescore(
    suite_path: Path,
    run: str = typer.Option(..., help="Stored run id to replay"),
    out: Optional[Path] = typer.Option(None),
    db: Optional[Path] = typer.Option(None),
    seed: int = typer.Option(0),
    with_judges: Optional[str] = typer.Option(
        None, help="Comma-separated judge checks to run on the stored outputs (e.g. faithfulness). "
                   "Makes judge calls only; the system under test is never called."),
    concurrency: int = typer.Option(8),
) -> None:
    """Re-score a stored run with the current checks and gate rules.

    Without --with-judges this makes no model calls at all. With it, only the named
    judges are called, which is how you produce judged outputs to label (CAL-1).
    """
    suite = load_suite(suite_path)
    store = _store(suite, db)
    record = store.get_run(run)
    if record is None:
        raise typer.BadParameter(f"no run {run!r} in this store")
    baseline = engine.find_manifest(suite, record["baseline_version_id"])
    candidate = engine.find_manifest(suite, record["candidate_version_id"])
    if baseline is None or candidate is None:
        raise typer.BadParameter("the run's versions no longer match any manifest in versions/")
    stored_cases = store.run_cases(run) if hasattr(store, "run_cases") else {}
    case_ids = set(stored_cases)
    if not case_ids:  # runs from before cases were stored: use the ids the results carry
        case_ids = {r.case_id for r in store.results_for(run, baseline.id) if "~" not in r.case_id}
    cases = [c for c in engine.build_cases(suite) if c.id in case_ids]
    # Replay as many samples per case as the stored run has (it may have used
    # --samples); confirmation reruns sit above that range and are served on demand.
    stored_idx = [r.sample_idx for r in store.results_for(run, baseline.id)
                  if "~" not in r.case_id and r.sample_idx < suite.samples]
    samples = (max(stored_idx) + 1) if stored_idx else suite.samples
    replay = engine.StoredRunner(store, run)
    env = _Env(suite, "mock", skip_judges=True)  # checks only; the runner is the stored run
    checks, judge = list(env.checks), None
    if with_judges:
        wanted = {name.strip() for name in with_judges.split(",") if name.strip()}
        unknown = wanted - {c.name for c in suite.checks}
        if unknown:
            raise typer.BadParameter(f"not checks in this suite: {', '.join(sorted(unknown))}")
        from llm_evals.judges.client import JudgeClient
        from llm_evals.runners.providers.anthropic import AnthropicProvider

        cache = suite.dir / ".evals" / "judge_cache.sqlite"
        judge = JudgeClient(AnthropicProvider(), model=suite.judge.get("model", "claude-sonnet-5-5"),
                            params=suite.judge.get("params"), cache_path=cache)
        judged = suite.model_copy(update={
            "checks": [c for c in suite.checks if c.name in wanted], "pairwise": None})
        checks += engine.build_checks(judged, judge)
    report = engine.run_comparison(
        suite, baseline, candidate, store=store, runner=replay, checks=checks, judge=judge,
        cases=cases, samples=samples, seed=seed, kind="rescore", on_progress=_progress,
        max_concurrency=concurrency,
    )
    calls = "judge calls only" if with_judges else "no model calls"
    report.warnings.insert(0, f"Re-scored stored run {run} with current checks and gate rules; "
                              f"{calls}. {replay.misses} samples weren't in the stored run.")
    if out:
        write_reports(report, out)
    typer.echo(render_markdown(report))
    raise typer.Exit(report.exit_code)


@app.command()
def promote(
    suite_path: Path,
    manifest: str = typer.Argument(..., help="Manifest to make the suite baseline"),
    run_id: Optional[str] = typer.Option(None, help="Run that justified the promotion"),
    db: Optional[Path] = typer.Option(None),
) -> None:
    """Make a version the suite's baseline (history is kept)."""
    suite = load_suite(suite_path)
    store = _store(suite, db)
    version = engine.load_version(suite, manifest)
    store.promote(suite.suite, version, manifest, run_id)
    typer.echo(f"{suite.suite}: baseline is now {version.name} ({version.id}).")


@app.command()
def init(
    path: Path = typer.Argument(..., help="Folder to create the suite in"),
    ci: bool = typer.Option(False, "--ci", help="Also write .github/workflows/llm-evals.yml at the repo root"),
    force: bool = typer.Option(False, help="Write into a folder that isn't empty, replacing starter files"),
    action: str = typer.Option("bwyndham/llm-evals@v0.1.0",
                               help="The Action the --ci workflow uses: a GitHub reference, or a local "
                                    "copy such as ./.github/actions/llm-evals"),
) -> None:
    """Create a starter suite that runs offline straight away."""
    import re
    import subprocess
    from importlib.resources import files

    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise typer.BadParameter(f"{path} isn't empty; pick a new folder or pass --force")
    starter = files("llm_evals") / "templates" / "starter"
    written = []

    def copy(node, dest: Path) -> None:
        for child in node.iterdir():
            if child.name == "__pycache__":
                continue
            target = dest / child.name
            if child.is_dir():
                copy(child, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(child.read_bytes())
                written.append(target)

    copy(starter, path)
    slug = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-") or "my-suite"
    suite_yaml = path / "suite.yaml"
    suite_yaml.write_text(suite_yaml.read_text().replace("suite: support-triage", f"suite: {slug}", 1))
    typer.echo(f"Created {len(written)} files in {path}.")

    if ci:
        try:
            root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, check=True,
                                       capture_output=True, text=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            root = Path.cwd()
        workflow = root / ".github" / "workflows" / "llm-evals.yml"
        if workflow.exists() and not force:
            typer.echo(f"{workflow} already exists; left it alone (pass --force to replace it).")
        else:
            suite_rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            template = (files("llm_evals") / "templates" / "workflow.yml").read_text()
            workflow.parent.mkdir(parents=True, exist_ok=True)
            workflow.write_text(template.replace("__SUITE__", suite_rel).replace("__ACTION__", action))
            typer.echo(f"Wrote {workflow}. Add an ANTHROPIC_API_KEY repository secret to use it.")

    shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    typer.echo(f"Try it offline: llm-evals gate {shown} --candidate versions/v2.yaml --provider mock --skip-judges")


def _mount_calibration() -> None:
    try:
        from llm_evals.cli.calibrate import calibrate_app
    except ImportError:  # pragma: no cover - module arrives with Phase 2
        return
    app.add_typer(calibrate_app, name="calibration",
                  help="Label judged samples and compute judge agreement.")


_mount_calibration()


def main() -> None:  # pragma: no cover
    sys.exit(app())
