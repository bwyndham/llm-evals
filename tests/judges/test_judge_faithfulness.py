from __future__ import annotations

import json
from pathlib import Path

import pytest
from judges_helpers import SUITE_DIR, TICKET, Ctx, make_case, make_result, payload_of

from llm_evals import registry
from llm_evals.checks.faithfulness import Faithfulness, faithfulness
from llm_evals.judges import JudgeClient, unit_hash
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.base import FakeProvider
from llm_evals.types import Check, ErrorKind, GatingUnit


def answer(covered: dict[str, bool], invented: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "atoms": [{"id": k, "covered": v, "evidence": f"ev {k}"} for k, v in covered.items()],
            "invented": invented or [],
        }
    )


def bound(*texts: str, **kw: object) -> tuple[Faithfulness, FakeProvider]:
    provider = FakeProvider.from_texts(texts)
    check = faithfulness(suite_dir=str(SUITE_DIR), **kw)  # type: ignore[arg-type]
    check.bind_judge(JudgeClient(provider))
    return check, provider


def test_registered_and_protocol() -> None:
    check = registry.get("check", "faithfulness")(suite_dir=str(SUITE_DIR))
    assert isinstance(check, Check)
    assert check.needs_judge is True
    assert check.kind == "faithfulness"
    assert check.critical is False


def test_pass_when_all_covered_and_nothing_invented() -> None:
    check, provider = bound(answer({"set/a1": True, "set/a2": True}))
    score = check.score(make_case(), make_result(), Ctx())
    assert score.passed, score.reason
    assert score.value == 1.0
    assert score.judge_model == "claude-sonnet-5-5"
    unit = score.sub_scores[0]
    assert unit["unit"] == "faithfulness" and unit["passed"] is True
    assert unit["prompt_hash"] == score.prompt_hash == check.gating_units()[0].prompt_hash
    assert (unit["covered"], unit["total"], unit["invented"]) == (2, 2, 0)
    atoms = [s for s in score.sub_scores if "atom" in s]
    assert [a["atom"] for a in atoms] == ["set/a1", "set/a2"]
    assert atoms[0]["evidence"] == "ev set/a1"
    assert score.sub_scores[-1] == {"invented": []}
    assert len(provider.calls) == 1


def test_payload_has_task_non_hinted_atoms_and_ticket() -> None:
    check, provider = bound(answer({"set/a1": True, "set/a2": True}))
    check.score(make_case(hinted_atom_ids=["set/a2"]), make_result(), Ctx())
    payload = payload_of(provider.calls[0])
    assert payload["task"] == "faithfulness"
    assert payload["atoms"] == [{"id": "set/a1", "text": "Bookkeepers can generate a statement of open invoices"}]
    assert payload["ticket"] == TICKET
    assert payload["ask"].startswith("can bookkeepers")


def test_fail_when_atom_uncovered() -> None:
    check, _ = bound(answer({"set/a1": True, "set/a2": False}))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    assert score.value == 0.5
    assert "set/a2" in score.reason and "not covered" in score.reason
    assert score.sub_scores[0]["passed"] is False


def test_fail_when_requirement_invented() -> None:
    inv = [{"text": "Statements are exported as PDF", "reason": "the ask never mentions PDF"}]
    check, _ = bound(answer({"set/a1": True, "set/a2": True}, inv))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    assert score.value == 1.0
    assert "invented" in score.reason and "PDF" in score.reason
    assert score.sub_scores[0]["invented"] == 1
    assert score.sub_scores[-1] == {"invented": inv}


def test_atom_missing_from_answer_counts_as_uncovered() -> None:
    check, _ = bound(answer({"set/a1": True}))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    missing = next(s for s in score.sub_scores if s.get("atom") == "set/a2")
    assert missing["covered"] is False and missing["evidence"] == "judge gave no verdict"


def test_malformed_json_is_judge_error() -> None:
    check, _ = bound("the ticket looks faithful to me")
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed
    assert score.reason.startswith("judge error:")
    assert score.sub_scores[0] == {"unit": "faithfulness", "passed": False, "error": "judge"}
    assert score.sub_scores[-1]["raw_judge_output"] == "the ticket looks faithful to me"


def test_infra_error_reason() -> None:
    provider = FakeProvider(script=[InfraError("503 after retries")])
    check = faithfulness(suite_dir=str(SUITE_DIR))
    check.bind_judge(JudgeClient(provider))
    score = check.score(make_case(), make_result(), Ctx())
    assert not score.passed and score.reason.startswith("infra:")


def test_no_output_fails_without_calling_judge() -> None:
    check, provider = bound()
    score = check.score(make_case(), make_result(None, error_kind=ErrorKind.SYSTEM, error="bad json"), Ctx())
    assert not score.passed and score.reason.startswith("no output")
    assert score.sub_scores == [{"unit": "faithfulness", "passed": False, "error": "no output"}]
    assert provider.calls == []


def test_case_without_atoms_is_check_error() -> None:
    check, _ = bound()
    case = make_case()
    case.source = None
    score = check.score(case, make_result(), Ctx())
    assert not score.passed and score.reason.startswith("check error:")


def test_scoring_before_bind_raises() -> None:
    check = faithfulness(suite_dir=str(SUITE_DIR))
    with pytest.raises(RuntimeError, match="bind_judge"):
        check.score(make_case(), make_result(), Ctx())
    with pytest.raises(RuntimeError, match="bind_judge"):
        check.gating_units()


def test_gating_unit() -> None:
    check, _ = bound()
    units = check.gating_units()
    t = SUITE_DIR / "judges" / "faithfulness.jinja"
    s = SUITE_DIR / "judges" / "faithfulness_system.jinja"
    assert units == [GatingUnit(key="faithfulness", judge_model="claude-sonnet-5-5", prompt_hash=unit_hash(t, system_template=s))]


def test_suite_dir_from_context_when_not_passed() -> None:
    check = faithfulness()
    check.bind_judge(JudgeClient(FakeProvider.from_texts([answer({"set/a1": True, "set/a2": True})])))
    with pytest.raises(RuntimeError, match="suite_dir"):
        check.gating_units()
    assert check.score(make_case(), make_result(), Ctx(SUITE_DIR)).passed
    assert check.gating_units()[0].key == "faithfulness"


def test_unit_hash_changes_with_template(tmp_suite: Path) -> None:
    check = faithfulness(suite_dir=str(tmp_suite))
    check.bind_judge(JudgeClient(FakeProvider()))
    before = check.gating_units()[0].prompt_hash
    (tmp_suite / "judges" / "faithfulness.jinja").write_text("A different prompt.\n")
    assert check.gating_units()[0].prompt_hash != before


def test_missing_template_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(FatalRunError, match="not found"):
        faithfulness(suite_dir=str(tmp_path))


def test_elaborations_are_recorded_but_never_fail() -> None:
    text = json.dumps({"atoms": [{"id": "set/a1", "covered": True}, {"id": "set/a2", "covered": True}],
                       "elaborations": [{"text": "I am told the file is too large", "reason": "error message"}],
                       "invented": []})
    check, _ = bound(text)
    score = check.score(make_case(), make_result(), Ctx())
    assert score.passed, score.reason
    assert {"elaborations": [{"text": "I am told the file is too large", "reason": "error message"}]} in score.sub_scores
