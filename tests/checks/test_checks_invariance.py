from __future__ import annotations

import sys
import types

import pytest
from checks_helpers import MISSING_THEN, SYNTAX_ERROR, FakeContext, make_output, make_result

import llm_evals.checks.invariance as inv
from llm_evals.types import ErrorKind

ONE_SCENARIO = "Feature: F\n  Scenario: A\n    When a\n    Then b\n"
THREE_SCENARIOS = ONE_SCENARIO + "  Scenario: B\n    When c\n    Then d\n  Scenario: C\n    When e\n    Then f\n"


@pytest.fixture
def reword_calls(monkeypatch):
    calls = []

    def fake_reword_ask(case, seed):
        calls.append((case.id, seed))
        return {"ask": f"reworded[{seed}]: {case.input['ask']}"}

    monkeypatch.setattr(inv, "_load_reword_ask", lambda: fake_reword_ask)
    return calls


def run(output, rerun, case, **params):
    ctx = FakeContext(rerun)
    s = inv.invariance(**params).score(case, make_result(output), ctx)
    return s, ctx


def test_defaults_and_attributes():
    check = inv.invariance()
    assert (check.name, check.kind, check.critical) == ("invariance", "metamorphic", False)
    assert (check.relation, check.seed, check.max_scenario_delta) == ("reword_ask", 0, 1)


def test_bad_params():
    with pytest.raises(TypeError, match="unexpected keyword argument 'delta'"):
        inv.invariance(delta=2)
    with pytest.raises(ValueError, match="unknown relation 'shuffle'"):
        inv.invariance(relation="shuffle")
    with pytest.raises(TypeError, match="max_scenario_delta"):
        inv.invariance(max_scenario_delta=-1)


def test_passes_when_structure_holds(case, reword_calls):
    s, ctx = run(make_output(), make_result(make_output()), case, seed=7)
    assert s.passed, s.reason
    assert reword_calls == [("c1", 7)]
    assert ctx.calls[0][1] == {"ask": f"reworded[7]: {case.input['ask']}"}
    original, rerun = s.sub_scores
    assert original == {"role": "original", "scenarios": 2, "open_questions_empty": False, "gherkin_ok": True}
    assert rerun["role"] == "rerun" and rerun["scenarios"] == 2 and rerun["seed"] == 7
    assert "Export usage report" in rerun["output"]


def test_scenario_delta_within_limit_passes(case, reword_calls):
    s, _ = run(make_output(), make_result(make_output(acceptance_criteria=ONE_SCENARIO)), case)
    assert s.passed, s.reason


def test_scenario_delta_over_limit_fails(case, reword_calls):
    s, _ = run(
        make_output(acceptance_criteria=ONE_SCENARIO),
        make_result(make_output(acceptance_criteria=THREE_SCENARIOS)),
        case,
    )
    assert not s.passed
    assert "scenario count changed 1 -> 3 (max delta 1)" in s.reason
    s, _ = run(
        make_output(acceptance_criteria=ONE_SCENARIO),
        make_result(make_output(acceptance_criteria=THREE_SCENARIOS)),
        case,
        max_scenario_delta=2,
    )
    assert s.passed


def test_open_questions_emptiness_must_agree(case, reword_calls):
    s, _ = run(make_output(), make_result(make_output(open_questions=[])), case)
    assert not s.passed
    assert "open_questions emptiness changed (non-empty -> empty)" in s.reason


def test_invalid_gherkin_on_either_side_fails(case, reword_calls):
    s, _ = run(make_output(), make_result(make_output(acceptance_criteria=MISSING_THEN)), case)
    assert not s.passed
    assert "reworded Gherkin invalid" in s.reason
    s, _ = run(make_output(acceptance_criteria=SYNTAX_ERROR), make_result(make_output()), case)
    assert not s.passed
    assert "original Gherkin invalid (Gherkin parse error at line 5" in s.reason


def test_rerun_infra_error_is_marked_infra(case, reword_calls):
    rerun = make_result(None, error_kind=ErrorKind.INFRA, error="rate limited after 3 retries")
    s, _ = run(make_output(), rerun, case)
    assert not s.passed
    assert s.reason.startswith("infra:")
    assert "rate limited" in s.reason


def test_rerun_system_error_fails_without_infra_prefix(case, reword_calls):
    rerun = make_result(None, error_kind=ErrorKind.SYSTEM, error="invalid JSON", raw_output="not json")
    s, _ = run(make_output(), rerun, case)
    assert not s.passed
    assert s.reason == "rerun has no output (system error: invalid JSON)"
    assert s.sub_scores[0]["output"] == "not json"


def test_rerun_exception_fails_loudly(case, reword_calls):
    s, _ = run(make_output(), RuntimeError("boom"), case)
    assert not s.passed
    assert s.reason == "check error: rerun raised RuntimeError: boom"


def test_reworder_unavailable_fails_loudly(case, monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'llm_evals.cases.renderers.ask_renderer'")

    monkeypatch.setattr(inv, "_load_reword_ask", missing)
    s, ctx = run(make_output(), make_result(make_output()), case)
    assert not s.passed
    assert s.reason.startswith("check error: could not build the reword_ask input")
    assert ctx.calls == []


def test_long_rerun_output_is_truncated(case, reword_calls):
    s, _ = run(make_output(), make_result(make_output(description="x" * 5000)), case)
    assert s.passed
    assert "[truncated" in s.sub_scores[1]["output"]
    assert len(s.sub_scores[1]["output"]) < 2100


def test_original_error_result_skips_rerun(case, reword_calls, error_results):
    for r in error_results:
        ctx = FakeContext()
        s = inv.invariance().score(case, r, ctx)
        assert not s.passed and s.reason.startswith("no output (")
        assert ctx.calls == []


def test_lazy_import_uses_ask_renderer_module(case, monkeypatch):
    """The real loader imports reword_ask from the renderer module path."""
    fake = types.ModuleType("llm_evals.cases.renderers.ask_renderer")
    fake.reword_ask = lambda case, seed: {"ask": "fake"}  # type: ignore[attr-defined]
    import llm_evals.cases.renderers as renderers_pkg

    monkeypatch.setitem(sys.modules, "llm_evals.cases.renderers.ask_renderer", fake)
    monkeypatch.setattr(renderers_pkg, "ask_renderer", fake, raising=False)
    s, ctx = run(make_output(), make_result(make_output()), case)
    assert s.passed, s.reason
    assert ctx.calls[0][1] == {"ask": "fake"}


@pytest.mark.parametrize("returned, expected", [("plain ask", {"ask": "plain ask"}), ({"ask": "d"}, {"ask": "d"})])
def test_reworder_return_shapes(case, monkeypatch, returned, expected):
    monkeypatch.setattr(inv, "_load_reword_ask", lambda: lambda c, s: returned)
    _, ctx = run(make_output(), make_result(make_output()), case)
    assert ctx.calls[0][1] == expected
