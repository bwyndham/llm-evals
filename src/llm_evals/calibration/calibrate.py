"""Compute and store a calibration for one gating unit (CAL-2, CAL-4).

Human labels are joined with the judge's stored verdicts for the same samples
across every run. Only verdicts produced by exactly this judge model and prompt
hash count, so a new prompt or model needs its verdicts on labeled samples
before it can be approved. Labels themselves are judge-independent and reused.
"""

from __future__ import annotations

from collections import defaultdict

from llm_evals.calibration.metrics import agreement_metrics
from llm_evals.store.sqlite import CalibrationRecord, Store, unit_entries
from llm_evals.types import GatingUnit

SampleKey = tuple[str, str, str, int]


def unit_triples(store: Store, check: str) -> list[GatingUnit]:
    """The distinct (unit, judge model, prompt hash) triples in a check's stored
    scores. A unit's model/hash come from its sub_score entry, falling back to
    Score.judge_model / Score.prompt_hash; units missing either are skipped."""
    seen: set[tuple[str, str, str]] = set()
    for row in store.score_rows(check=check):
        for key, _, model, phash in unit_entries(row.score):
            if model and phash:
                seen.add((key, model, phash))
    return [GatingUnit(key=k, judge_model=m, prompt_hash=h) for k, m, h in sorted(seen)]


def human_labels(store: Store, unit_key: str, *, labeler: str | None = None
                 ) -> tuple[dict[SampleKey, bool], int]:
    """One human verdict per sample (True = pass) and the number of samples dropped.

    With several labelers on a sample the majority wins; a tie is dropped.
    """
    votes: dict[SampleKey, list[bool]] = defaultdict(list)
    for label in store.labels_for(unit_key=unit_key, labeler=labeler):
        votes[(label.run_id, label.version_id, label.case_id, label.sample_idx)].append(
            label.label == "pass")
    out, ties = {}, 0
    for key, vs in votes.items():
        passes = sum(vs)
        if passes * 2 == len(vs):
            ties += 1
        else:
            out[key] = passes * 2 > len(vs)
    return out, ties


def judge_verdicts(store: Store, unit_key: str, judge_model: str, prompt_hash: str
                   ) -> dict[SampleKey, bool]:
    """The judge's verdict per sample for exactly this triple (first check by name
    if two checks carry the same unit on one sample)."""
    out: dict[SampleKey, bool] = {}
    for v in sorted(store.unit_verdicts(unit_key), key=lambda v: v.check):
        if v.judge_model == judge_model and v.prompt_hash == prompt_hash:
            out.setdefault((v.run_id, v.version_id, v.case_id, v.sample_idx), v.passed)
    return out


def calibrate(
    store: Store,
    unit_key: str,
    judge_model: str,
    prompt_hash: str,
    *,
    min_n: int = 50,
    min_kappa: float = 0.6,
    labeler: str | None = None,
) -> CalibrationRecord:
    """Compute agreement for the triple, set `approved` (n >= min_n and kappa defined
    and >= min_kappa), append the record to the store's history, and return it."""
    humans, ties = human_labels(store, unit_key, labeler=labeler)
    judged = judge_verdicts(store, unit_key, judge_model, prompt_hash)
    keys = sorted(set(humans) & set(judged))
    key_set = set(keys)
    m = agreement_metrics((humans[k], judged[k]) for k in keys)
    approved = m.n >= min_n and m.kappa is not None and m.kappa >= min_kappa
    record = CalibrationRecord(
        unit_key=unit_key, judge_model=judge_model, prompt_hash=prompt_hash,
        n=m.n, agreement=m.agreement, kappa=m.kappa, sensitivity=m.sensitivity,
        specificity=m.specificity, prevalence=m.prevalence, approved=approved,
        label_set={
            "run_ids": sorted({k[0] for k in keys}),
            "count": m.n,
            "labelers": sorted({
                lab.labeler for lab in store.labels_for(unit_key=unit_key, labeler=labeler)
                if (lab.run_id, lab.version_id, lab.case_id, lab.sample_idx) in key_set
            }),
            "ties_dropped": ties,
        },
        tp=m.tp, fp=m.fp, fn=m.fn, tn=m.tn, min_n=min_n, min_kappa=min_kappa,
    )
    return store.save_calibration(record)
