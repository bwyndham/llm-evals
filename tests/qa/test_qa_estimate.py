"""`estimate`: priced per call and per sample, so it lands near what runs really cost."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_evals import engine, qa
from llm_evals.cli.main import app
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.store.sqlite import Store
from llm_evals.suite import load_suite
from llm_evals.types import Result, Trace

runner = CliRunner()


def _starter(tmp_path: Path) -> Path:
    suite = tmp_path / "triage"
    assert runner.invoke(app, ["init", str(suite)]).exit_code == 0
    return suite


def _parts(suite_dir: Path):
    suite = load_suite(suite_dir)
    base = engine.load_version(suite, "versions/v1.yaml")
    cand = engine.load_version(suite, "versions/v2.yaml")
    return suite, base, cand, engine.build_cases(suite), engine.build_checks(suite)


def test_measured_history_prices_per_sample(tmp_path):
    suite, base, cand, cases, checks = _parts(_starter(tmp_path))
    h = qa.UsageHistory(input_tokens=500, output_tokens=200, cost_usd=0.0002, samples=100)
    est = qa.estimate(suite, base, cand, cases, checks, history={base.model: h})
    samples = 2 * len(cases) * suite.samples  # two versions, no reruns in the starter
    assert est.cost_usd == round(0.0002 * samples, 4)
    assert "measured" in est.assumptions and "rough guess" not in est.assumptions


def test_a_guess_is_priced_per_call_so_volume_never_triggers_long_prompt_pricing(tmp_path):
    suite, base, cand, cases, checks = _parts(_starter(tmp_path))
    est = qa.estimate(suite, base, cand, cases, checks, samples=50)  # lots of calls, short prompts
    per_call = cost_usd(base.model, est.input_tokens // est.calls, 700)
    assert abs(est.cost_usd - per_call * est.calls) / est.cost_usd < 0.05
    assert "no history" in est.assumptions and "--limit 5" in est.assumptions


def test_only_checks_that_rerun_add_calls(tmp_path):
    suite, base, cand, cases, checks = _parts(_starter(tmp_path))
    plain = qa.estimate(suite, base, cand, cases, checks)

    class Metamorphic:  # metamorphic but makes no extra calls, like research's follows_corpus
        kind = "metamorphic"

    class Rerunner:
        kind = "metamorphic"
        reruns_per_sample = 1

    assert qa.estimate(suite, base, cand, cases, [*checks, Metamorphic()]).calls == plain.calls
    assert qa.estimate(suite, base, cand, cases, [*checks, Rerunner()]).calls == 2 * plain.calls


def test_cli_learns_history_by_model_including_results_saved_before_traces_named_it(tmp_path):
    suite_dir = _starter(tmp_path)
    suite, base, *_ = _parts(suite_dir)
    store = Store(suite_dir / ".evals" / "evals.sqlite")
    store.create_run(suite=suite.suite, kind="compare", run_id="r")
    for i in range(30):  # old results: no model on the trace, matched through the version id
        store.save_result(Result(run_id="r", case_id=f"c{i}", version_id=base.id, sample_idx=0,
                                 output={}, trace=Trace(input_tokens=400, output_tokens=100,
                                                        cost_usd=0.0001, request_ids=["req"])))
    res = runner.invoke(app, ["estimate", str(suite_dir)])
    assert res.exit_code == 0, res.output
    assert "measured $0.00010" in res.output and "from 30 past samples" in res.output
