from __future__ import annotations

import logging

import pytest

from llm_evals.runners.errors import infra_share, is_incomplete
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.types import ErrorKind, Result


def results(n_ok: int, n_infra: int, n_system: int = 0) -> list[Result]:
    kinds = [ErrorKind.NONE] * n_ok + [ErrorKind.INFRA] * n_infra + [ErrorKind.SYSTEM] * n_system
    return [
        Result(run_id="r", case_id=f"c{i}", version_id="v", sample_idx=0, error_kind=k)
        for i, k in enumerate(kinds)
    ]


def test_infra_share() -> None:
    assert infra_share([]) == 0.0
    assert infra_share(results(90, 5, 5)) == pytest.approx(0.05)


def test_is_incomplete() -> None:
    assert not is_incomplete(results(95, 5))  # exactly 5% is fine
    assert is_incomplete(results(94, 6))
    assert is_incomplete(results(8, 2), max_infra_share=0.1) is True
    assert not is_incomplete(results(0, 0, 10))  # SYSTEM errors don't count
    assert is_incomplete([])


def test_prices() -> None:
    assert cost_usd("claude-haiku-5-5", 50_000, 1_000_000) == pytest.approx(0.505)
    # Above 100K prompt tokens Haiku 5.5 bills $0.50 / $2.50 per 1M.
    assert cost_usd("claude-haiku-5-5", 200_000, 1_000_000) == pytest.approx(2.6)
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd("claude-sonnet-5-5", 0, 1_000_000) == pytest.approx(10.0)
    assert cost_usd("claude-opus-5-5", 500_000, 100_000) == pytest.approx(4.0)


def test_unknown_model_costs_zero_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert cost_usd("mystery-model-xyz", 100, 100) == 0.0
    assert "mystery-model-xyz" in caplog.text
