from __future__ import annotations

import json

import pytest
from judges_helpers import SUITE_DIR, TICKET, make_case, make_result, payload_of

from llm_evals import registry
from llm_evals.checks.pairwise import Pairwise, pairwise, win_rate
from llm_evals.judges import JudgeClient
from llm_evals.runners.errors import InfraError
from llm_evals.runners.providers.base import FakeProvider
from llm_evals.types import ErrorKind, PairwiseCheck, Preference

OTHER = {**TICKET, "title": "Customer statements"}


def w(winner: str, reason: str = "") -> str:
    return json.dumps({"winner": winner, "reason": reason or f"picked {winner}"})


def bound(*items: object) -> tuple[Pairwise, FakeProvider]:
    provider = FakeProvider(script=list(items))
    check = pairwise(suite_dir=str(SUITE_DIR))
    check.bind_judge(JudgeClient(provider))
    return check, provider


def test_registered_and_protocol() -> None:
    check = registry.get("pairwise", "pairwise")(suite_dir=str(SUITE_DIR))
    assert isinstance(check, PairwiseCheck)
    assert check.needs_judge is True
    assert check.gating_units() == []


def test_both_orders_agree_gives_winner() -> None:
    # a-first: judge picks "b" (the candidate); b-first: picks "a" (still the candidate)
    check, provider = bound(w("b"), w("a"))
    pref = check.compare(make_case(), make_result(TICKET), make_result(OTHER))
    assert pref.winner == "b"
    assert "agree" in pref.reason
    first, second = (payload_of(c) for c in provider.calls)
    assert first["task"] == second["task"] == "pairwise"
    assert (first["a"], first["b"]) == (TICKET, OTHER)
    assert (second["a"], second["b"]) == (OTHER, TICKET)
    assert first["criteria"] and all(isinstance(c, str) for c in first["criteria"])
    assert first["ask"].startswith("can bookkeepers")


def test_baseline_wins_when_consistent() -> None:
    check, _ = bound(w("a"), w("b"))
    assert check.compare(make_case(), make_result(TICKET), make_result(OTHER)).winner == "a"


@pytest.mark.parametrize("answers", [("a", "a"), ("b", "b"), ("a", "tie"), ("tie", "b")])
def test_position_bias_or_partial_tie_gives_tie(answers: tuple[str, str]) -> None:
    check, _ = bound(*(w(x) for x in answers))
    pref = check.compare(make_case(), make_result(TICKET), make_result(OTHER))
    assert pref.winner == "tie"
    assert "disagree" in pref.reason


def test_both_tie_is_tie() -> None:
    check, _ = bound(w("tie"), w("tie"))
    assert check.compare(make_case(), make_result(TICKET), make_result(OTHER)).winner == "tie"


def test_judge_error_is_tie() -> None:
    check, _ = bound("a is better", w("a"))
    pref = check.compare(make_case(), make_result(TICKET), make_result(OTHER))
    assert pref.winner == "tie" and pref.reason.startswith("judge error:")


def test_infra_error_is_tie() -> None:
    check, _ = bound(w("a"), InfraError("timeout"))
    pref = check.compare(make_case(), make_result(TICKET), make_result(OTHER))
    assert pref.winner == "tie" and pref.reason.startswith("infra:")


def test_missing_outputs() -> None:
    check, provider = bound()
    bad = make_result(None, error_kind=ErrorKind.SYSTEM, error="bad json")
    assert check.compare(make_case(), bad, make_result(OTHER)).winner == "b"
    assert check.compare(make_case(), make_result(TICKET), bad).winner == "a"
    assert check.compare(make_case(), bad, bad).winner == "tie"
    assert provider.calls == []


def test_compare_before_bind_raises() -> None:
    check = pairwise(suite_dir=str(SUITE_DIR))
    with pytest.raises(RuntimeError, match="bind_judge"):
        check.compare(make_case(), make_result(TICKET), make_result(OTHER))


def test_compare_without_suite_dir_raises() -> None:
    check = pairwise()
    check.bind_judge(JudgeClient(FakeProvider()))
    with pytest.raises(RuntimeError, match="suite_dir"):
        check.compare(make_case(), make_result(TICKET), make_result(OTHER))


def test_win_rate() -> None:
    prefs = [Preference(winner=x) for x in ["b", "b", "a", "tie"]]
    r = win_rate(prefs)
    assert (r["n"], r["wins"], r["losses"], r["ties"]) == (4, 2, 1, 1)
    assert r["win_share"] == 0.5 and r["loss_share"] == 0.25 and r["tie_share"] == 0.25
    assert r["win_rate"] == pytest.approx(0.625)
    assert r["decisive_win_share"] == pytest.approx(2 / 3)


def test_win_rate_empty_and_all_ties() -> None:
    assert win_rate([])["win_share"] is None
    r = win_rate([Preference(winner="tie")] * 3)
    assert r["win_rate"] == 0.5 and r["decisive_win_share"] is None
