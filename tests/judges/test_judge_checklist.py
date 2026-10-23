from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from judges_helpers import SUITE_DIR, TICKET, Ctx, make_case, make_result, payload_of

from llm_evals import registry
from llm_evals.checks.checklist import Checklist, checklist
from llm_evals.judges import JudgeClient, unit_hash
from llm_evals.runners.errors import FatalRunError
from llm_evals.runners.providers.base import FakeProvider
from llm_evals.types import Check

RUBRIC = yaml.safe_load((SUITE_DIR / "rubrics" / "spec_standard.yaml").read_text())
IDS = [item["id"] for item in RUBRIC]


def answer(no: set[str] = frozenset(), skip: set[str] = frozenset(), case: str = "lower") -> str:
    items = []
    for i in IDS:
        if i in skip:
            continue
        a = "no" if i in no else "yes"
        items.append({"id": i, "answer": a.upper() if case == "upper" else a, "reason": f"r {i}"})
    return json.dumps({"items": items})


def bound(*texts: str, suite_dir: Path = SUITE_DIR) -> tuple[Checklist, FakeProvider]:
    provider = FakeProvider.from_texts(texts)
    check = checklist(suite_dir=str(suite_dir))
    check.bind_judge(JudgeClient(provider))
    return check, provider


def test_rubric_has_the_spec_standard_items() -> None:
    assert len(IDS) == 8 and len(set(IDS)) == 8
    assert {"title_states_change", "testable_ac", "no_unstated_requirements", "description_concise"} <= set(IDS)


def test_registered_and_protocol() -> None:
    check = registry.get("check", "checklist")(suite_dir=str(SUITE_DIR))
    assert isinstance(check, Check)
    assert check.needs_judge is True and check.kind == "judge"


def test_all_yes_passes_with_one_call() -> None:
    check, provider = bound(answer(case="upper"))
    score = check.score(make_case(), make_result(), Ctx())
    assert score.passed, score.reason
    assert score.value == 1.0
    assert len(provider.calls) == 1
    assert [s["unit"] for s in score.sub_scores] == [f"checklist:{i}" for i in IDS]
    assert all(s["passed"] and s["answer"] == "yes" for s in score.sub_scores)
    payload = payload_of(provider.calls[0])
    assert payload["task"] == "checklist"
    assert payload["items"] == [{"id": r["id"], "question": r["question"].strip()} for r in RUBRIC]
    assert payload["ticket"] == TICKET and payload["ask"]


def test_one_no_fails_that_unit_only() -> None:
    check, _ = bound(answer(no={"testable_ac"}))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    assert score.value == pytest.approx(7 / 8)
    assert "testable_ac" in score.reason
    by_unit = {s["unit"]: s for s in score.sub_scores}
    assert by_unit["checklist:testable_ac"]["passed"] is False
    assert by_unit["checklist:testable_ac"]["reason"] == "r testable_ac"
    assert sum(s["passed"] for s in score.sub_scores) == 7


def test_skipped_item_fails() -> None:
    check, _ = bound(answer(skip={"description_concise"}))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    s = next(s for s in score.sub_scores if s["unit"] == "checklist:description_concise")
    assert s["answer"] is None and s["reason"] == "judge gave no answer"


@pytest.mark.parametrize("text", ["all good!", '{"items": [{"id": "testable_ac", "answer": "maybe"}]}'])
def test_malformed_answer_is_judge_error(text: str) -> None:
    check, _ = bound(text)
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed and score.reason.startswith("judge error:")
    units = [s for s in score.sub_scores if "unit" in s]
    assert len(units) == 8 and not any(u["passed"] for u in units)


def test_gating_units_one_per_item_with_item_hashes() -> None:
    check, _ = bound()
    units = check.gating_units()
    assert [u.key for u in units] == [f"checklist:{i}" for i in IDS]
    t = SUITE_DIR / "judges" / "checklist.jinja"
    s = SUITE_DIR / "judges" / "checklist_system.jinja"
    for u, item in zip(units, RUBRIC, strict=True):
        assert u.prompt_hash == unit_hash(t, item["question"].strip(), system_template=s)
        assert u.judge_model == "claude-sonnet-5-5"
    assert len({u.prompt_hash for u in units}) == 8


def test_sub_score_hashes_match_gating_units() -> None:
    check, _ = bound(answer())
    score = check.score(make_case(), make_result(), Ctx())
    assert [s["prompt_hash"] for s in score.sub_scores] == [u.prompt_hash for u in check.gating_units()]


def test_editing_one_question_changes_only_that_hash(tmp_suite: Path) -> None:
    before = {u.key: u.prompt_hash for u in bound(suite_dir=tmp_suite)[0].gating_units()}
    path = tmp_suite / "rubrics" / "spec_standard.yaml"
    items = yaml.safe_load(path.read_text())
    items[2]["question"] = "Can every scenario be checked as pass or fail?"
    path.write_text(yaml.safe_dump(items))
    after = {u.key: u.prompt_hash for u in bound(suite_dir=tmp_suite)[0].gating_units()}
    changed = [k for k in before if before[k] != after[k]]
    assert changed == [f"checklist:{items[2]['id']}"]


def test_unit_hashes_are_case_independent() -> None:
    check, _ = bound(answer(), answer())
    a = check.score(make_case(), make_result(), Ctx())
    other = make_case()
    other.input = {"ask": "something else entirely"}
    b = check.score(other, make_result(), Ctx())
    assert [s["prompt_hash"] for s in a.sub_scores] == [s["prompt_hash"] for s in b.sub_scores]
    assert a.sub_scores[0]["call_hash"] != b.sub_scores[0]["call_hash"]


def test_scoring_before_bind_raises() -> None:
    check = checklist(suite_dir=str(SUITE_DIR))
    with pytest.raises(RuntimeError, match="bind_judge"):
        check.score(make_case(), make_result(), Ctx())


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "- {id: a, question: 'Q?'}\n- {id: a, question: 'Q2?'}\n",
        "- {id: a}\n",
    ],
)
def test_bad_rubric_is_fatal(tmp_suite: Path, content: str) -> None:
    (tmp_suite / "rubrics" / "bad.yaml").write_text(content)
    with pytest.raises(FatalRunError):
        checklist(rubric="rubrics/bad.yaml", suite_dir=str(tmp_suite))


def test_rubric_may_be_a_mapping_with_items(tmp_suite: Path) -> None:
    (tmp_suite / "rubrics" / "m.yaml").write_text("items:\n  - {id: a, question: 'Q?'}\n")
    check = checklist(rubric="rubrics/m.yaml", suite_dir=str(tmp_suite))
    assert check.items == [{"id": "a", "question": "Q?"}]
