"""Fake runs for calibration tests: results plus judge Scores with gating units."""

from __future__ import annotations

from llm_evals.store.sqlite import Store
from llm_evals.types import Case, ErrorKind, Result, Score

JUDGE_MODEL = "claude-sonnet-5-5"
HASH_A = "aaaaaaaaaaaa"
HASH_B = "bbbbbbbbbbbb"
SECRET = "JUDGE-REASON-SHOULD-NOT-SHOW"


def make_case(case_id: str, *, held_out: bool = False) -> Case:
    return Case(
        id=case_id,
        input={"ask": f"ask text for {case_id}"},
        source={"atoms": [{"id": "a1", "text": f"atom one of {case_id}", "kind": "functional"},
                          {"id": "a2", "text": f"atom two of {case_id}", "kind": "constraint"}],
                "actor": "admin"},
        provenance="constructed",
        held_out=held_out,
    )


def unit_score(check: str, units: dict[str, bool], *, per_unit_hash: bool = True,
               prompt_hash: str = HASH_A, judge_model: str = JUDGE_MODEL) -> Score:
    subs = []
    for key, passed in units.items():
        sub = {"unit": key, "passed": passed, "reason": SECRET, "answer": "no-" + SECRET,
               "question": f"Is {key} satisfied?"}
        if per_unit_hash:
            sub["prompt_hash"] = prompt_hash
        subs.append(sub)
    return Score(check=check, kind="judge", passed=all(units.values()), reason=SECRET,
                 sub_scores=subs, judge_model=judge_model,
                 prompt_hash=None if per_unit_hash else prompt_hash)


def seed_run(
    store: Store,
    verdicts: dict[str, dict[str, bool]],
    *,
    check: str = "checklist",
    run_id: str | None = None,
    version_id: str = "v1",
    per_unit_hash: bool = True,
    prompt_hash: str = HASH_A,
    held_out: set[str] = frozenset(),
    save_cases: bool = True,
    errored: set[str] = frozenset(),
) -> str:
    """verdicts: case_id -> {unit_key: judge passed}; one sample (idx 0) per case."""
    if run_id is None or store.get_run(run_id) is None:
        run_id = store.create_run(suite="spec-writing", kind="compare", run_id=run_id)
    cases = [make_case(cid, held_out=cid in held_out) for cid in verdicts]
    if save_cases:
        store.save_run_cases(run_id, cases)
    for cid, units in verdicts.items():
        bad = cid in errored
        store.save_result(Result(
            run_id=run_id, case_id=cid, version_id=version_id, sample_idx=0,
            output=None if bad else {"title": f"Ticket for {cid}", "description": "d",
                                     "acceptance_criteria": "Feature: x", "assumptions": [],
                                     "open_questions": []},
            error_kind=ErrorKind.SYSTEM if bad else ErrorKind.NONE,
        ))
        store.save_score(run_id, version_id, cid, 0,
                         unit_score(check, units, per_unit_hash=per_unit_hash,
                                    prompt_hash=prompt_hash))
    return run_id
