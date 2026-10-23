from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from research_checks_helpers import FakeContext, FakeJudge, JudgeError, make_case, make_output, make_result

from llm_evals.checks.research import ClaimSupportResponse, claim_support
from llm_evals.runners.errors import FatalRunError
from llm_evals.types import ErrorKind, GatingUnit

SUITE_DIR = Path(__file__).resolve().parents[2] / "suites" / "research-agent"


@pytest.fixture
def suite(tmp_path):
    judges = tmp_path / "judges"
    judges.mkdir()
    (judges / "claim_support.jinja").write_text("Judge {{ n_claims }} claim(s) for: {{ question }}\n")
    (judges / "claim_support_system.jinja").write_text("You are a fact-checker.\n")
    return tmp_path


def two_claims():
    return make_output(claims=[
        {"text": "Harbor Town has 12,500 people.", "citations": ["d1"]},
        {"text": "Harbor Town was founded in 1852.", "citations": ["d2", "d1"]},
    ])


def bound(suite_dir, judge, **params):
    check = claim_support(suite_dir=str(suite_dir), **params)
    check.bind_judge(judge)
    return check


def expected_hash(suite_dir: Path) -> str:
    """User template bytes then system template bytes (judges.client.unit_hash with system_template)."""
    h = hashlib.sha256((suite_dir / "judges" / "claim_support.jinja").read_bytes())
    h.update((suite_dir / "judges" / "claim_support_system.jinja").read_bytes())
    return h.hexdigest()[:12]


def test_defaults():
    check = claim_support()
    assert (check.name, check.kind, check.critical, check.needs_judge) == ("claim_support", "faithfulness", False, True)
    assert check.template == "judges/claim_support.jinja"
    assert check.system_template == "judges/claim_support_system.jinja"
    assert check.suite_dir is None


def test_shipped_templates_resolve_by_default():
    check = claim_support()
    assert check.template_path() == SUITE_DIR / "judges" / "claim_support.jinja"
    assert check.template_path().is_file() and check.system_template_path().is_file()
    text = check.template_path().read_text()
    assert "```json" not in text  # the JudgeClient appends the payload block
    assert "directly entails" in text


def test_gating_units(suite):
    check = bound(suite, FakeJudge(model="judge-m"))
    assert check.gating_units() == [
        GatingUnit(key="claim_support", judge_model="judge-m",
                   prompt_hash=expected_hash(suite))
    ]


def test_gating_units_need_a_judge():
    with pytest.raises(RuntimeError, match="bind_judge"):
        claim_support().gating_units()


def test_all_supported_passes_and_sends_contract_payload(suite):
    judge = FakeJudge()
    check = bound(suite, judge)
    score = check.score(make_case(), make_result(two_claims()), FakeContext())
    assert score.passed, score.reason
    assert score.value == 1.0
    assert score.judge_model == "claude-judge-test"
    assert score.prompt_hash == check.gating_units()[0].prompt_hash
    assert score.sub_scores[0] == {"unit": "claim_support", "passed": True, "cost_usd": 0.002}
    assert [(s["id"], s["supported"], s["judged"], s["cited"]) for s in score.sub_scores[1:]] == [
        ("c1", True, True, ["d1"]), ("c2", True, True, ["d2", "d1"]),
    ]
    assert all("unit" not in s for s in score.sub_scores[1:])

    (call,) = judge.calls
    assert call["user_template"] == suite / "judges" / "claim_support.jinja"
    assert call["system_template"] == suite / "judges" / "claim_support_system.jinja"
    assert call["response_model"] is ClaimSupportResponse
    assert call["variables"]["question"] == "How many people live in Harbor Town?"
    assert call["payload"] == {
        "task": "claim_support",
        "claims": [
            {"id": "c1", "text": "Harbor Town has 12,500 people.",
             "cited_docs": [{"id": "d1", "text": "The 2020 census counted 12,500 people in Harbor Town."}]},
            {"id": "c2", "text": "Harbor Town was founded in 1852.",
             "cited_docs": [{"id": "d2", "text": "Harbor Town was founded in 1852 by fishermen."},
                            {"id": "d1", "text": "The 2020 census counted 12,500 people in Harbor Town."}]},
        ],
    }


def test_unsupported_claim_fails(suite):
    score = bound(suite, FakeJudge({"c2": False})).score(make_case(), make_result(two_claims()), FakeContext())
    assert not score.passed
    assert score.value == 0.5
    assert score.sub_scores[0]["unit"] == "claim_support" and score.sub_scores[0]["passed"] is False
    assert score.sub_scores[2]["supported"] is False
    assert "1 of 2 claim(s) unsupported" in score.reason and "c2: verdict for c2" in score.reason


def test_uncited_and_non_corpus_claims_unsupported_without_judge(suite):
    judge = FakeJudge()
    output = make_output(claims=[
        {"text": "Supported.", "citations": ["d1"]},
        {"text": "No source.", "citations": []},
        {"text": "Ghost source.", "citations": ["d99"]},
    ])
    score = bound(suite, judge).score(make_case(), make_result(output), FakeContext())
    assert not score.passed
    assert [c["id"] for c in judge.calls[0]["payload"]["claims"]] == ["c1"]
    assert [(s["id"], s["supported"], s["judged"], s["reason"]) for s in score.sub_scores[1:]] == [
        ("c1", True, True, "verdict for c1"),
        ("c2", False, False, "no citations"),
        ("c3", False, False, "cited docs are not in the corpus"),
    ]


def test_judge_not_called_when_nothing_is_judgeable(suite):
    judge = FakeJudge()
    output = make_output(claims=[{"text": "No source.", "citations": []}])
    score = bound(suite, judge).score(make_case(), make_result(output), FakeContext())
    assert not score.passed and judge.calls == []


def test_no_claims_passes_vacuously(suite):
    judge = FakeJudge()
    output = make_output(status="not_found", answer="", claims=[])
    score = bound(suite, judge).score(make_case(), make_result(output), FakeContext())
    assert score.passed and judge.calls == []
    assert score.sub_scores == [{"unit": "claim_support", "passed": True, "cost_usd": 0.0}]


def test_judge_error_fails_with_a_failed_unit(suite):
    error = JudgeError("judge returned invalid JSON")
    error.raw_text = "not json"
    check = bound(suite, FakeJudge(error=error))
    score = check.score(make_case(), make_result(two_claims()), FakeContext())
    assert not score.passed
    assert score.reason == "judge error: judge returned invalid JSON"
    assert score.sub_scores == [
        {"unit": "claim_support", "passed": False, "error": "judge"},
        {"raw_judge_output": "not json"},
    ]
    assert score.judge_model == "claude-judge-test" and score.prompt_hash == expected_hash(suite)


def test_real_judge_error_class(suite):
    client = pytest.importorskip("llm_evals.judges.client")
    check = bound(suite, FakeJudge(error=client.JudgeError("bad json", raw_text="{")))
    score = check.score(make_case(), make_result(two_claims()), FakeContext())
    assert score.reason == "judge error: bad json"
    assert score.sub_scores[0] == {"unit": "claim_support", "passed": False, "error": "judge"}


def test_provider_failure_is_infra(suite):
    check = bound(suite, FakeJudge(error=TimeoutError("read timed out")))
    score = check.score(make_case(), make_result(two_claims()), FakeContext())
    assert not score.passed
    assert score.reason == "infra: judge call failed (TimeoutError: read timed out)"
    assert score.sub_scores == [{"unit": "claim_support", "passed": False, "error": "infra"}]


def test_config_error_raises(suite):
    check = bound(suite, FakeJudge(error=FatalRunError("template needs a variable")))
    with pytest.raises(FatalRunError):
        check.score(make_case(), make_result(two_claims()), FakeContext())


def test_no_output_fails_with_a_failed_unit(suite):
    result = make_result(None, error_kind=ErrorKind.SYSTEM, error="bad json")
    score = bound(suite, FakeJudge()).score(make_case(), result, FakeContext())
    assert not score.passed and score.reason == "no output (system error: bad json)"
    assert score.sub_scores == [{"unit": "claim_support", "passed": False, "error": "no output"}]
    assert score.judge_model == "claude-judge-test"


def test_missing_verdict_is_judge_error(suite):
    partial = ClaimSupportResponse.model_validate({"claims": [{"id": "c1", "supported": True}]})
    score = bound(suite, FakeJudge(parsed=partial)).score(make_case(), make_result(two_claims()), FakeContext())
    assert not score.passed and score.reason == "judge error: no verdict for claim(s) c2"
    assert score.sub_scores == [{"unit": "claim_support", "passed": False, "error": "judge"}]


def test_wrong_shape_response_is_judge_error(suite):
    check = bound(suite, FakeJudge(parsed={"claims": "yes"}))
    score = check.score(make_case(), make_result(two_claims()), FakeContext())
    assert not score.passed and score.reason.startswith("judge error: unusable response")


def test_dict_response_is_accepted(suite):
    parsed = {"claims": [{"id": "c1", "supported": True, "reason": ""}, {"id": "c2", "supported": True}]}
    score = bound(suite, FakeJudge(parsed=parsed)).score(make_case(), make_result(two_claims()), FakeContext())
    assert score.passed


def test_unbound_judge_raises():
    with pytest.raises(RuntimeError, match="bind_judge"):
        claim_support().score(make_case(), make_result(two_claims()), FakeContext())


def test_templates_resolve_against_context_when_no_suite_dir(suite):
    judge = FakeJudge()
    check = claim_support()
    check.bind_judge(judge)
    assert check.score(make_case(), make_result(two_claims()), FakeContext(suite)).passed
    assert judge.calls[0]["user_template"] == suite / "judges" / "claim_support.jinja"


@pytest.mark.parametrize("name", ["claim_support.jinja", "claim_support_system.jinja"])
def test_template_change_changes_prompt_hash(suite, name):
    check = bound(suite, FakeJudge())
    before = check.gating_units()[0].prompt_hash
    (suite / "judges" / name).write_text("A different prompt.\n")
    assert check.gating_units()[0].prompt_hash != before


def test_bad_template_param_rejected():
    with pytest.raises(TypeError, match="template"):
        claim_support(template="")
