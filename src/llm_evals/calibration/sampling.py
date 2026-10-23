"""Choose which samples of a run a human should label (CAL-1).

A sample is one (run, version, case, sample_idx) scored by the check. Its judge
verdict is FAIL when any gating unit in the check's sub_scores failed. The
choice is stratified on that verdict so both classes show up: by default half
the picks are judge-fails when enough exist, and the other stratum fills any
shortfall. It is deterministic for a given seed, and the picks come back
shuffled so their order does not reveal the judge's verdict.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from llm_evals.store.sqlite import Store, unit_entries
from llm_evals.types import Case, ErrorKind


@dataclass(frozen=True)
class LabelSample:
    run_id: str
    version_id: str
    case_id: str
    sample_idx: int
    check: str
    units: tuple[str, ...]  # gating unit keys present in this sample's score
    judge_passed: bool  # every unit passed; never shown to the labeler

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.run_id, self.version_id, self.case_id, self.sample_idx)


def candidate_samples(store: Store, run_id: str, check: str) -> list[LabelSample]:
    """Every scored sample of the check in the run that has gating units and a
    usable output (no stored Result, or an errored one, is skipped), sorted by key."""
    out = []
    for row in store.score_rows(check=check, run_id=run_id):
        entries = unit_entries(row.score)
        if not entries:
            continue
        result = store.get_result(row.run_id, row.version_id, row.case_id, row.sample_idx)
        if result is None or result.error_kind != ErrorKind.NONE or result.output is None:
            continue
        out.append(LabelSample(
            run_id=row.run_id, version_id=row.version_id, case_id=row.case_id,
            sample_idx=row.sample_idx, check=check,
            units=tuple(sorted({key for key, *_ in entries})),
            judge_passed=all(passed for _, passed, *_ in entries),
        ))
    return sorted(out, key=lambda s: s.key)


def choose_samples(
    store: Store,
    run_id: str,
    check: str,
    *,
    labeler: str,
    n: int = 50,
    seed: int = 0,
    fail_share: float = 0.5,
    skip_held_out: bool = False,
    cases: dict[str, Case] | None = None,
) -> list[LabelSample]:
    """Up to `n` samples to label, stratified by judge verdict.

    - Samples where `labeler` already labeled every unit are skipped; partly
      labeled samples stay eligible (the CLI asks only for the missing units).
    - With `skip_held_out`, held-out cases are skipped. Held-out status comes from
      `cases`, else the run's stored cases; a case whose status is unknown is
      skipped too, since it can't be shown to be safe.
    """
    if n <= 0:
        return []
    if not 0.0 <= fail_share <= 1.0:
        raise ValueError("fail_share must be between 0 and 1")
    done: dict[tuple[str, str, str, int], set[str]] = {}
    for label in store.labels_for(run_id=run_id, labeler=labeler):
        done.setdefault((label.run_id, label.version_id, label.case_id, label.sample_idx),
                        set()).add(label.unit_key)
    known = cases if cases is not None else (store.run_cases(run_id) if skip_held_out else {})

    pool = []
    for sample in candidate_samples(store, run_id, check):
        if set(sample.units) <= done.get(sample.key, set()):
            continue
        if skip_held_out:
            case = known.get(sample.case_id)
            if case is None or case.held_out:
                continue
        pool.append(sample)

    fails = [s for s in pool if not s.judge_passed]
    passes = [s for s in pool if s.judge_passed]
    n_fail = min(round(n * fail_share), len(fails))
    n_pass = min(n - n_fail, len(passes))
    n_fail = min(len(fails), n - n_pass)  # fill a short pass stratum with fails

    rng = random.Random(seed)
    labeled_cases = {key[2] for key in done}
    # The same case can be in both strata (one output judged pass, another fail), so
    # the second stratum avoids the first stratum's cases as well as labeled ones.
    fail_picks = _spread(fails, n_fail, rng, labeled_cases)
    pass_picks = _spread(passes, n_pass, rng, labeled_cases | {s.case_id for s in fail_picks})
    chosen = fail_picks + pass_picks
    # Show every case once before any case comes back, so repeats (the same ask
    # with a different output) are rare and spread out rather than back to back.
    rng.shuffle(chosen)
    first: set[str] = set(labeled_cases)
    ordered, later = [], []
    for sample in chosen:
        (later if sample.case_id in first else ordered).append(sample)
        first.add(sample.case_id)
    return ordered + later


def _spread(samples: list[LabelSample], k: int, rng: random.Random,
            avoid: set[str]) -> list[LabelSample]:
    """Pick `k` samples preferring cases not yet picked (or already labeled).

    Labels on many outputs of the same few asks are correlated, so the sample
    should cover as many distinct cases as the run has before repeating any.
    """
    by_case: dict[str, list[LabelSample]] = {}
    for sample in samples:
        by_case.setdefault(sample.case_id, []).append(sample)
    for group in by_case.values():
        rng.shuffle(group)
    case_ids = list(by_case)
    rng.shuffle(case_ids)
    case_ids.sort(key=lambda c: c in avoid)  # unlabeled cases first; stable keeps the shuffle
    picked: list[LabelSample] = []
    round_ = 0
    while len(picked) < k:
        added = False
        for case_id in case_ids:
            group = by_case[case_id]
            if round_ < len(group):
                picked.append(group[round_])
                added = True
                if len(picked) == k:
                    break
        if not added:
            break
        round_ += 1
    return picked
