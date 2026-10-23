from llm_evals import registry
from llm_evals.runners.simulated import SimulatedRunner, simulate_outcomes, true_p_fail
from llm_evals.types import Case, ErrorKind, Runner, VersionManifest


def case(cid: str, p_fail: float | None = None) -> Case:
    expected = None if p_fail is None else {"p_fail": p_fail}
    return Case(id=cid, input={}, expected=expected, provenance="constructed")


def version(name: str = "v", **params) -> VersionManifest:
    return VersionManifest(name=name, model="m", prompt_templates={}, params=params)


def test_protocol_and_registration():
    assert isinstance(SimulatedRunner(), Runner)
    assert registry._REGISTRIES["runner"]["simulated"] is SimulatedRunner


def test_result_shape():
    v = version()
    r = SimulatedRunner().run(case("a"), v, 2, "run-1")
    assert (r.run_id, r.case_id, r.version_id, r.sample_idx) == ("run-1", "a", "v", 2)
    assert set(r.output) == {"passed"} and r.error_kind is ErrorKind.NONE
    v.id = "abc123"
    assert SimulatedRunner().run(case("a"), v, 0, "r").version_id == "abc123"


def _passes(runner, cases, v, run_id, k=5):
    return [runner.run(c, v, i, run_id).output["passed"] for c in cases for i in range(k)]


def test_deterministic_for_same_run_id():
    cases = [case(f"c{i}", 0.5) for i in range(100)]
    v = version()
    a = _passes(SimulatedRunner(seed=3), cases, v, "run-1")
    assert a == _passes(SimulatedRunner(seed=3), cases, v, "run-1")
    assert a != _passes(SimulatedRunner(seed=3), cases, v, "run-2")  # fresh draws
    assert a != _passes(SimulatedRunner(seed=4), cases, v, "run-1")
    assert a != _passes(SimulatedRunner(seed=3), cases, version("other"), "run-1")


def test_failure_probability_and_shift():
    runner = SimulatedRunner(default_p_fail=0.2)
    cases = [case(f"c{i}") for i in range(2000)]
    rate = sum(_passes(runner, cases, version(), "r", k=1)) / len(cases)
    assert abs(rate - 0.8) < 0.03
    rate = sum(_passes(runner, cases, version(shift=0.3), "r", k=1)) / len(cases)
    assert abs(rate - 0.5) < 0.04
    assert not any(_passes(runner, [case("x", 0.9)], version(shift=0.5), "r", k=50))
    assert all(_passes(runner, [case("x", 0.1)], version(shift=-0.2), "r", k=50))
    assert true_p_fail(case("x", 0.9), version(shift=0.5)) == 1.0
    assert true_p_fail(case("x"), version(), default_p_fail=0.3) == 0.3


def test_infra_error_rate():
    runner = SimulatedRunner(seed=1)
    v = version(infra_error_rate=0.2)
    results = [runner.run(case(f"c{i}"), v, j, "r") for i in range(1000) for j in range(5)]
    infra = [r for r in results if r.error_kind is ErrorKind.INFRA]
    assert abs(len(infra) / len(results) - 0.2) < 0.02
    assert all(r.output is None and r.error for r in infra)

    cases = [case(f"c{i}") for i in range(1000)]
    outcomes = simulate_outcomes(cases, v, 5, runner, run_id="r")
    assert set(outcomes) == {c.id for c in cases}
    assert sum(map(len, outcomes.values())) == len(results) - len(infra)
    clean = simulate_outcomes(cases, version(), 5, runner, run_id="r")
    assert all(len(s) == 5 for s in clean.values())
