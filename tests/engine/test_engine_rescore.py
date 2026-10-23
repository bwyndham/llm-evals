"""CHK-6: re-scoring a stored run replays its outputs with no new calls."""

from pathlib import Path

from llm_evals.engine import StoredRunner, run_comparison
from llm_evals.runners.simulated import SimulatedRunner
from llm_evals.store.sqlite import Store
from llm_evals.suite import SuiteConfig
from llm_evals.types import Case, Score, VersionManifest


class PassedCheck:
    name = "simulated_pass"
    kind = "deterministic"
    critical = False

    def score(self, case, result, ctx):
        return Score(check=self.name, kind=self.kind, passed=bool(result.output and result.output.get("passed")))


class CountingRunner(SimulatedRunner):
    calls = 0

    def run(self, *a, **kw):
        CountingRunner.calls += 1
        return super().run(*a, **kw)


def test_rescore_reproduces_the_run_without_calls(tmp_path):
    suite = SuiteConfig(suite="s", runner={"type": "simulated"}, baseline="x", cases=[], checks=[],
                        samples=3, dir=tmp_path)
    cases = [Case(id=f"c{i}", input={}, expected={"p_fail": 0.2}, provenance="constructed") for i in range(40)]
    base = VersionManifest(name="b", model="sim", prompt_templates={}, id="b")
    cand = VersionManifest(name="c", model="sim", prompt_templates={}, id="c", params={"shift": 0.3})
    store = Store(Path(":memory:"))
    first = run_comparison(suite, base, cand, store=store, runner=CountingRunner(),
                           checks=[PassedCheck()], cases=cases, n_boot=500, run_id="orig")
    calls = CountingRunner.calls
    replay = StoredRunner(store, "orig")
    again = run_comparison(suite, base, cand, store=store, runner=replay, checks=[PassedCheck()],
                           cases=cases, n_boot=500, run_id="replay")
    assert CountingRunner.calls == calls and replay.misses == 0
    assert again.comparison.delta == first.comparison.delta
    assert again.decision == first.decision
