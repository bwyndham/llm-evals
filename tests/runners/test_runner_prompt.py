from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from llm_evals import registry
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.manifest import manifest_with_overrides
from llm_evals.runners.prompt import PromptRunner, parse_output, strip_json_fence
from llm_evals.runners.providers.base import FakeProvider, ProviderResponse
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.types import Case, ErrorKind

from runners_helpers import GOOD_TICKET


def make_runner(*items) -> tuple[PromptRunner, FakeProvider]:
    fake = FakeProvider(script=list(items))
    return PromptRunner(fake, output_schema="spec_ticket"), fake


def resp(text: str, **kw) -> ProviderResponse:
    kw.setdefault("input_tokens", 1000)
    kw.setdefault("output_tokens", 500)
    return ProviderResponse(text=text, **kw)


def test_registered() -> None:
    assert registry.get("runner", "prompt") is PromptRunner


def test_good_json(case, version) -> None:
    runner, fake = make_runner(resp(GOOD_TICKET, request_id="req_1", stop_reason="end_turn"))
    r = runner.run(case, version, sample_idx=2, run_id="run1")
    assert r.error_kind is ErrorKind.NONE and r.error is None
    assert r.output["title"] == "Export usage as CSV"
    assert r.raw_output == GOOD_TICKET
    assert (r.run_id, r.case_id, r.version_id, r.sample_idx) == ("run1", "c1", version.id, 2)
    call = fake.calls[0]
    assert call["system"] == "You write specs.\n"
    assert call["user"] == "Ask: let admins export usage\n"
    assert call["model"] == "claude-haiku-5-5"
    assert call["params"] == {"max_tokens": 4000, "effort": "low"}


def test_trace_recorded(case, version) -> None:
    runner, _ = make_runner(resp(GOOD_TICKET, request_id="req_1"))
    t = runner.run(case, version, 0, "run1").trace
    assert (t.input_tokens, t.output_tokens) == (1000, 500)
    assert t.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", 1000, 500))
    assert t.cost_usd == pytest.approx((1000 * 0.10 + 500 * 0.50) / 1e6)
    assert t.request_ids == ["req_1"]
    assert t.latency_ms >= 0


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", "  ```JSON\n{}```  "])
def test_fenced_json(case, version, fence: str) -> None:
    runner, _ = make_runner(resp(fence.replace("{}", GOOD_TICKET)))
    r = runner.run(case, version, 0, "run1")
    assert r.error_kind is ErrorKind.NONE
    assert r.output["assumptions"] == ["90 days"]


def test_strip_fence_leaves_plain_text() -> None:
    assert strip_json_fence('  {"a": 1}\n') == '{"a": 1}'


def test_invalid_json_is_system(case, version) -> None:
    runner, _ = make_runner(resp("Here is your spec: {"))
    r = runner.run(case, version, 0, "run1")
    assert r.error_kind is ErrorKind.SYSTEM
    assert "not valid JSON" in r.error
    assert r.output is None and r.raw_output == "Here is your spec: {"


@pytest.mark.parametrize(
    "text",
    [
        '{"title": "t", "description": "d"}',  # missing fields
        GOOD_TICKET.replace('["90 days"]', '"90 days"'),  # wrong type
        GOOD_TICKET[:-1] + ', "extra": 1}',  # extra key
        "[1, 2]",  # not an object
    ],
)
def test_schema_mismatch_is_system(case, version, text: str) -> None:
    runner, _ = make_runner(resp(text))
    r = runner.run(case, version, 0, "run1")
    assert r.error_kind is ErrorKind.SYSTEM
    assert r.output is None and r.raw_output == text


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_bad_stop_reason_is_system(case, version, stop: str) -> None:
    runner, _ = make_runner(resp(GOOD_TICKET if stop == "max_tokens" else "", stop_reason=stop))
    r = runner.run(case, version, 0, "run1")
    assert r.error_kind is ErrorKind.SYSTEM
    assert stop in r.error
    assert r.trace.output_tokens == 500  # cost still recorded


def test_no_schema_returns_any_object(case, version) -> None:
    r = PromptRunner(FakeProvider(script=[resp('{"x": 1}')])).run(case, version, 0, "r")
    assert r.output == {"x": 1}


def test_infra_exception_is_infra(case, version) -> None:
    runner, _ = make_runner(InfraError("overloaded", request_id="req_9"))
    r = runner.run(case, version, 0, "run1")
    assert r.error_kind is ErrorKind.INFRA
    assert "overloaded" in r.error
    assert r.trace.request_ids == ["req_9"]


def test_raw_sdk_infra_exception_is_infra(case, version) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    runner, _ = make_runner(anthropic.APITimeoutError(request=request))
    assert runner.run(case, version, 0, "run1").error_kind is ErrorKind.INFRA


def test_fatal_propagates(case, version) -> None:
    runner, _ = make_runner(FatalRunError("bad key"))
    with pytest.raises(FatalRunError, match="bad key"):
        runner.run(case, version, 0, "run1")


def test_missing_variable_is_fatal(version) -> None:
    runner, fake = make_runner(resp(GOOD_TICKET))
    case = Case(id="c2", input={"not_ask": "x"}, provenance="constructed")
    with pytest.raises(FatalRunError, match="c2"):
        runner.run(case, version, 0, "run1")
    assert fake.calls == []


def test_system_role_optional(case, version) -> None:
    v = manifest_with_overrides(
        version, prompt_templates={"user": version.prompt_templates["user"]}
    )
    runner, fake = make_runner(resp(GOOD_TICKET))
    assert runner.run(case, v, 0, "run1").error_kind is ErrorKind.NONE
    assert fake.calls[0]["system"] is None


def test_user_role_required(case, version) -> None:
    v = manifest_with_overrides(
        version, prompt_templates={"system": version.prompt_templates["system"]}
    )
    runner, _ = make_runner(resp(GOOD_TICKET))
    with pytest.raises(FatalRunError, match="user"):
        runner.run(case, v, 0, "run1")


def test_unknown_schema_fails_fast() -> None:
    with pytest.raises(FatalRunError, match="nope"):
        PromptRunner(FakeProvider(), output_schema="nope")


def test_json_schema_file_validates_output(tmp_path) -> None:
    schema = tmp_path / "triage.schema.json"
    schema.write_text(json.dumps({
        "type": "object", "required": ["category"], "additionalProperties": False,
        "properties": {"category": {"enum": ["billing", "bug", "other"]}, "reply": {"type": "string"}}}))
    PromptRunner(FakeProvider(), output_schema=str(schema))  # a valid schema loads
    assert parse_output('{"category": "bug"}', str(schema)) == {"category": "bug"}
    with pytest.raises(ValueError, match="category: 'refund' is not one of"):
        parse_output('{"category": "refund"}', str(schema))
    with pytest.raises(ValueError, match="Additional properties"):
        parse_output('{"category": "bug", "mood": "sad"}', str(schema))


def test_yaml_schema_file_and_broken_or_missing_files(tmp_path) -> None:
    good = tmp_path / "out.schema.yaml"
    good.write_text("type: object\nrequired: [label]\nproperties:\n  label: {type: string}\n")
    assert parse_output('{"label": "x"}', str(good)) == {"label": "x"}
    bad = tmp_path / "bad.schema.json"
    bad.write_text(json.dumps({"type": "objekt"}))
    with pytest.raises(FatalRunError, match="not a valid JSON Schema"):
        PromptRunner(FakeProvider(), output_schema=str(bad))
    with pytest.raises(FatalRunError, match="not found"):
        PromptRunner(FakeProvider(), output_schema=str(tmp_path / "missing.json"))
