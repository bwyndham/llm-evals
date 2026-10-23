from __future__ import annotations

import threading
import time

import pytest

from llm_evals.runners.concurrency import run_all
from llm_evals.runners.errors import FatalRunError
from llm_evals.types import Result, Trace


def job(i: int, cost: float = 0.0, delay: float = 0.0, started: list[int] | None = None):
    def run() -> Result:
        if started is not None:
            started.append(i)
        time.sleep(delay)
        return Result(
            run_id="r", case_id=f"c{i}", version_id="v", sample_idx=0, trace=Trace(cost_usd=cost)
        )

    return run


def test_runs_everything_and_calls_on_result() -> None:
    seen: list[str] = []
    out = run_all([job(i) for i in range(20)], max_concurrency=4, cost_cap_usd=None,
                  on_result=lambda r: seen.append(r.case_id))
    assert sorted(r.case_id for r in out.results) == sorted(f"c{i}" for i in range(20))
    assert sorted(seen) == sorted(r.case_id for r in out.results)
    assert not out.stopped_by_cost_cap and out.n_skipped == 0


def test_respects_max_concurrency() -> None:
    lock = threading.Lock()
    live = peak = 0

    def make():
        def run() -> Result:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.01)
            with lock:
                live -= 1
            return Result(run_id="r", case_id="c", version_id="v", sample_idx=0)
        return run

    run_all([make() for _ in range(12)], max_concurrency=3, cost_cap_usd=None)
    assert peak <= 3


def test_cost_cap_stops_submission() -> None:
    started: list[int] = []
    jobs = [job(i, cost=1.0, started=started) for i in range(50)]
    out = run_all(jobs, max_concurrency=1, cost_cap_usd=2.5)
    # Serial: after 3 jobs cost is 3.0 > 2.5, so nothing else starts.
    assert out.stopped_by_cost_cap
    assert len(out.results) == 3 and started == [0, 1, 2]
    assert out.cost_usd == pytest.approx(3.0)
    assert out.n_skipped == 47


def test_cost_cap_lets_in_flight_finish() -> None:
    jobs = [job(i, cost=1.0, delay=0.02) for i in range(40)]
    out = run_all(jobs, max_concurrency=4, cost_cap_usd=1.5)
    assert out.stopped_by_cost_cap
    assert 2 <= len(out.results) < 40
    assert len(out.results) == out.n_submitted  # every submitted job's result kept


def test_cap_not_reached() -> None:
    out = run_all([job(i, cost=0.1) for i in range(5)], max_concurrency=2, cost_cap_usd=10.0)
    assert not out.stopped_by_cost_cap and len(out.results) == 5


def test_fatal_cancels_rest_and_reraises() -> None:
    started: list[int] = []

    def fatal() -> Result:
        started.append(-1)
        raise FatalRunError("bad key")

    jobs = [fatal] + [job(i, delay=0.01, started=started) for i in range(100)]
    with pytest.raises(FatalRunError, match="bad key"):
        run_all(jobs, max_concurrency=2, cost_cap_usd=None)
    assert len(started) < 10  # nothing new was submitted after the failure
