"""claim_support through the real JudgeClient (scripted provider, no API calls)."""

from __future__ import annotations

import json
import re

import pytest
from research_checks_helpers import FakeContext, make_case, make_output, make_result

from llm_evals.checks.research import claim_support

client_module = pytest.importorskip("llm_evals.judges.client")
providers = pytest.importorskip("llm_evals.runners.providers.base")


def answer_from_payload(call):
    blocks = re.findall(r"```json\n(.*?)\n```", call["user"], re.DOTALL)
    payload = json.loads(blocks[-1])
    assert payload["task"] == "claim_support"
    verdicts = [{"id": c["id"], "supported": "1852" not in c["text"], "reason": "checked"} for c in payload["claims"]]
    return json.dumps({"claims": verdicts})


def test_shipped_templates_render_and_verdicts_flow_through(tmp_path):
    provider = providers.FakeProvider(default=answer_from_payload)
    judge = client_module.JudgeClient(provider, model="claude-judge-test", cache_path=tmp_path / "judge.sqlite")
    check = claim_support()  # default suite dir: suites/research-agent
    check.bind_judge(judge)
    output = make_output(claims=[
        {"text": "Harbor Town has 12,500 people.", "citations": ["d1"]},
        {"text": "Harbor Town was founded in 1852.", "citations": ["d1"]},
    ])
    score = check.score(make_case(), make_result(output), FakeContext(check.template_path().parents[1]))
    assert not score.passed
    assert [s.get("supported") for s in score.sub_scores[1:]] == [True, False]
    assert score.prompt_hash == check.gating_units()[0].prompt_hash

    (call,) = provider.calls
    assert "How many people live in Harbor Town?" in call["user"]
    assert call["user"].rstrip().endswith("```")
    assert call["user"].count("```json") == 1  # only the payload block the client appends
    assert "fact-checker" in call["system"]
