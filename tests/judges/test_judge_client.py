from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from judges_helpers import SUITE_DIR, payload_of
from pydantic import BaseModel

from llm_evals.judges import (
    DEFAULT_JUDGE_PARAMS,
    JudgeClient,
    JudgeError,
    build_user_prompt,
    prompt_hash,
    unit_hash,
)
from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.base import FakeProvider, ProviderResponse

TEMPLATE = SUITE_DIR / "judges" / "pairwise.jinja"
SYSTEM = SUITE_DIR / "judges" / "pairwise_system.jinja"


class Verdict(BaseModel):
    winner: str
    reason: str = ""


GOOD = '{"winner": "a", "reason": "a covers more"}'


def resp(text: str, **kw: object) -> ProviderResponse:
    return ProviderResponse(text=text, input_tokens=1000, output_tokens=100, **kw)  # type: ignore[arg-type]


def call(client: JudgeClient, payload: dict | None = None):
    return client.judge(SYSTEM, TEMPLATE, {}, payload or {"task": "pairwise", "a": {}, "b": {}}, Verdict)


# --- prompt shape ------------------------------------------------------------


def test_payload_block_is_present_and_last() -> None:
    provider = FakeProvider(script=[resp(GOOD)])
    client = JudgeClient(provider)
    payload = {"task": "pairwise", "b": {"x": 1}, "a": {"y": [1, 2]}, "ask": "héllo"}
    call(client, payload)
    sent = provider.calls[0]
    user = sent["user"]
    assert user.endswith("```\n")
    blocks = re.findall(r"```json\n(.*?)\n```", user, re.DOTALL)
    assert len(blocks) == 1
    assert blocks[0] == json.dumps(payload, sort_keys=True)  # the exact convention
    assert user.rstrip().endswith(blocks[0] + "\n```")
    assert payload_of(sent) == payload
    assert user.startswith("Compare tickets")  # rendered template first
    assert sent["system"] and sent["system"].startswith("You are a careful reviewer")


def test_backticks_in_payload_cannot_close_the_fence() -> None:
    payload = {"task": "x", "ticket": {"description": "code:\n```\nrm -rf\n```"}}
    user = build_user_prompt("Intro", payload)
    assert user.count("```") == 2
    assert payload_of({"user": user}) == payload


def test_default_model_and_params_reach_provider() -> None:
    provider = FakeProvider(script=[resp(GOOD)])
    client = JudgeClient(provider)
    call(client)
    assert provider.calls[0]["model"] == "claude-sonnet-5-5"
    assert provider.calls[0]["params"] == {"max_tokens": 4000, "effort": "medium"}
    assert DEFAULT_JUDGE_PARAMS == {"max_tokens": 4000, "effort": "medium"}
    assert "temperature" not in provider.calls[0]["params"]


def test_result_fields_and_cost() -> None:
    client = JudgeClient(FakeProvider(script=[resp(GOOD)]))
    res = call(client)
    assert res.parsed == Verdict(winner="a", reason="a covers more")
    assert res.raw_text == GOOD
    assert re.fullmatch(r"[0-9a-f]{16}", res.prompt_hash)
    assert res.cached is False
    assert (res.input_tokens, res.output_tokens) == (1000, 100)
    assert res.cost_usd == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6)
    assert client.total_cost_usd == pytest.approx(res.cost_usd)


def test_prompt_hash_is_model_system_user() -> None:
    provider = FakeProvider(script=[resp(GOOD)])
    client = JudgeClient(provider)
    res = call(client)
    c = provider.calls[0]
    assert res.prompt_hash == prompt_hash(c["model"], c["system"], c["user"])
    assert prompt_hash("m", None, "u") == prompt_hash("m", "", "u")
    assert prompt_hash("m", "s", "u") != prompt_hash("m2", "s", "u")
    assert prompt_hash("ms", "", "u") != prompt_hash("m", "s", "u")


@pytest.mark.parametrize(
    "text",
    [
        "```json\n" + GOOD + "\n```",
        "Here is my verdict:\n```json\n" + GOOD + "\n```\nThanks.",
        "Sure. " + GOOD,
    ],
)
def test_lenient_json_extraction(text: str) -> None:
    res = call(JudgeClient(FakeProvider(script=[resp(text)])))
    assert res.parsed.winner == "a"


# --- caching -----------------------------------------------------------------


def test_second_identical_call_hits_cache() -> None:
    provider = FakeProvider(script=[resp(GOOD)])
    client = JudgeClient(provider)
    first = call(client)
    second = call(client)
    assert len(provider.calls) == 1
    assert second.cached and not first.cached
    assert second.parsed == first.parsed and second.prompt_hash == first.prompt_hash
    assert second.cost_usd == 0.0
    assert client.total_cost_usd == pytest.approx(first.cost_usd)
    assert (client.n_calls, client.n_cached) == (1, 1)


def test_different_payload_misses_cache() -> None:
    provider = FakeProvider(default=resp(GOOD))
    client = JudgeClient(provider)
    a = call(client, {"task": "pairwise", "a": {"v": 1}})
    b = call(client, {"task": "pairwise", "a": {"v": 2}})
    assert a.prompt_hash != b.prompt_hash
    assert len(provider.calls) == 2


def test_model_change_misses_cache(tmp_path: Path) -> None:
    db = tmp_path / "judge.sqlite"
    provider = FakeProvider(default=resp(GOOD))
    call(JudgeClient(provider, cache_path=db))
    call(JudgeClient(provider, model="claude-opus-5-5", cache_path=db))
    assert len(provider.calls) == 2


def test_file_cache_persists_across_clients(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "judge.sqlite"
    provider = FakeProvider(script=[resp(GOOD)])
    call(JudgeClient(provider, cache_path=db))
    again = JudgeClient(provider, cache_path=db)
    res = call(again)
    assert res.cached and len(provider.calls) == 1
    assert again.total_cost_usd == 0.0


def test_concurrent_identical_calls_make_one_provider_call() -> None:
    def slow(_: dict) -> ProviderResponse:
        time.sleep(0.05)
        return resp(GOOD)

    provider = FakeProvider(default=slow)
    client = JudgeClient(provider)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: call(client), range(8)))
    assert len(provider.calls) == 1
    assert sum(not r.cached for r in results) == 1


def test_cost_total_is_thread_safe() -> None:
    provider = FakeProvider(default=resp(GOOD))
    client = JudgeClient(provider)
    barrier = threading.Barrier(8)

    def work(i: int) -> None:
        barrier.wait()
        for j in range(10):
            call(client, {"task": "pairwise", "i": i, "j": j})

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, range(8)))
    assert client.n_calls == 80
    assert client.total_cost_usd == pytest.approx(80 * 0.003)


# --- failures ----------------------------------------------------------------


def test_malformed_json_raises_judge_error_and_is_not_retried() -> None:
    provider = FakeProvider(script=[resp("I think a is better")], default=resp(GOOD))
    client = JudgeClient(provider)
    with pytest.raises(JudgeError) as info:
        call(client)
    assert info.value.raw_text == "I think a is better"
    assert re.fullmatch(r"[0-9a-f]{16}", info.value.prompt_hash)
    with pytest.raises(JudgeError):  # same prompt: cached answer, still an error
        call(client)
    assert len(provider.calls) == 1
    assert client.total_cost_usd > 0  # a failed answer still cost money


def test_wrong_shape_raises_judge_error() -> None:
    client = JudgeClient(FakeProvider(script=[resp('{"verdict": "a"}')]))
    with pytest.raises(JudgeError, match="wrong shape"):
        call(client)


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_bad_stop_reason_raises_judge_error(stop: str) -> None:
    client = JudgeClient(FakeProvider(script=[resp(GOOD, stop_reason=stop)]))
    with pytest.raises(JudgeError, match=stop):
        call(client)


def test_provider_exception_propagates_and_is_not_cached() -> None:
    provider = FakeProvider(script=[InfraError("timeout"), resp(GOOD)])
    client = JudgeClient(provider)
    with pytest.raises(InfraError):
        call(client)
    assert call(client).cached is False
    assert len(provider.calls) == 2


def test_missing_template_is_fatal(tmp_path: Path) -> None:
    client = JudgeClient(FakeProvider(default=resp(GOOD)))
    with pytest.raises(FatalRunError):
        client.judge(None, tmp_path / "nope.jinja", {}, {}, Verdict)


def test_undefined_template_variable_is_fatal(tmp_path: Path) -> None:
    t = tmp_path / "t.jinja"
    t.write_text("Hello {{ name }}")
    client = JudgeClient(FakeProvider(default=resp(GOOD)))
    with pytest.raises(FatalRunError):
        client.judge(None, t, {}, {}, Verdict)
    res = client.judge(None, t, {"name": "x"}, {"task": "t"}, Verdict)
    assert res.parsed.winner == "a"


# --- unit hashes -------------------------------------------------------------


def test_unit_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    t = tmp_path / "t.jinja"
    t.write_text("Judge this.\n")
    h = unit_hash(t, "Is the title specific?")
    assert re.fullmatch(r"[0-9a-f]{12}", h)
    assert unit_hash(t, "Is the title specific?") == h
    assert unit_hash(t, "Is the title short?") != h
    assert unit_hash(t) != h
    t.write_text("Judge this carefully.\n")
    assert unit_hash(t, "Is the title specific?") != h


def test_unit_hash_matches_contract_formula(tmp_path: Path) -> None:
    import hashlib

    t = tmp_path / "t.jinja"
    t.write_bytes(b"template bytes")
    assert unit_hash(t, "item") == hashlib.sha256(b"template bytesitem").hexdigest()[:12]


def test_unit_hash_includes_system_template(tmp_path: Path) -> None:
    t, s = tmp_path / "t.jinja", tmp_path / "s.jinja"
    t.write_text("user")
    s.write_text("system v1")
    h1 = unit_hash(t, "q", system_template=s)
    assert h1 != unit_hash(t, "q")
    s.write_text("system v2")
    assert unit_hash(t, "q", system_template=s) != h1


def test_extract_json_takes_the_judges_corrected_answer():
    from llm_evals.judges.client import _extract_json

    text = ('{"atoms": [{"id": "a1", "covered": true}], "invented": [{"text": "x"}]}\n\n'
            'Correction: that is an elaboration, so nothing is invented.\n\n'
            '{"atoms": [{"id": "a1", "covered": true}], "invented": []}')
    assert _extract_json(text) == {"atoms": [{"id": "a1", "covered": True}], "invented": []}
