"""RUN-4: run jobs on a thread pool under a concurrency limit and a cost cap."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from llm_evals.runners.errors import FatalRunError
from llm_evals.types import Result


@dataclass
class RunOutcome:
    results: list[Result] = field(default_factory=list)  # completion order
    stopped_by_cost_cap: bool = False
    cost_usd: float = 0.0
    n_submitted: int = 0
    n_jobs: int = 0

    @property
    def n_skipped(self) -> int:
        return self.n_jobs - self.n_submitted


def run_all(
    jobs: list[Callable[[], Result]],
    *,
    max_concurrency: int,
    cost_cap_usd: float | None,
    on_result: Callable[[Result], None] | None = None,
) -> RunOutcome:
    """Run `jobs` with at most `max_concurrency` in flight.

    Jobs are submitted lazily, so when cumulative `trace.cost_usd` exceeds
    `cost_cap_usd` no new job starts; in-flight jobs finish and are kept.
    `on_result` is called from the calling thread, one result at a time, so it
    can write to a store without extra locking.

    A FatalRunError from any job cancels the unsubmitted jobs, waits for the
    in-flight ones, and re-raises. Any other exception from a job propagates
    the same way (jobs are expected to return a Result for per-sample errors).
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    outcome = RunOutcome(n_jobs=len(jobs))
    pending_jobs = iter(jobs)
    in_flight: set[Future[Result]] = set()
    stop = threading.Event()

    def over_cap() -> bool:
        return cost_cap_usd is not None and outcome.cost_usd > cost_cap_usd

    def fill(executor: ThreadPoolExecutor) -> None:
        while len(in_flight) < max_concurrency and not stop.is_set() and not over_cap():
            job = next(pending_jobs, None)
            if job is None:
                return
            in_flight.add(executor.submit(job))
            outcome.n_submitted += 1

    executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="eval")
    error: BaseException | None = None
    try:
        fill(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.discard(future)
                try:
                    result = future.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    if error is None or (
                        isinstance(exc, FatalRunError) and not isinstance(error, FatalRunError)
                    ):
                        error = exc
                    stop.set()
                    continue
                outcome.results.append(result)
                outcome.cost_usd += result.trace.cost_usd
                if on_result is not None and error is None:
                    on_result(result)
            if error is None:
                fill(executor)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if error is not None:
        raise error
    outcome.stopped_by_cost_cap = over_cap() and outcome.n_submitted < outcome.n_jobs
    return outcome
