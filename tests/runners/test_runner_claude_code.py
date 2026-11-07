"""`ClaudeCodeProvider`: model calls through the claude CLI. No real CLI is run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.claude_code import ClaudeCodeProvider


def ok(text: str = "hello") -> SimpleNamespace:
    body = {"type": "result", "subtype": "success", "is_error": False, "result": text, "stop_reason": "end_turn",
            "session_id": "s1", "usage": {"input_tokens": 3, "cache_creation_input_tokens": 600,
                                          "cache_read_input_tokens": 100, "output_tokens": 9}}
    return SimpleNamespace(stdout=json.dumps(body), stderr="", returncode=0)


def err(result: str, status: int | None = None) -> SimpleNamespace:
    body = {"type": "result", "subtype": "success", "is_error": True, "result": result, "api_error_status": status}
    return SimpleNamespace(stdout=json.dumps(body), stderr="", returncode=1)


def provider(*outcomes, calls: list | None = None) -> ClaudeCodeProvider:
    seq = list(outcomes)

    def run(cmd, **kw):
        if calls is not None:
            calls.append((cmd, kw))
        item = seq.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return ClaudeCodeProvider(binary="/bin/claude", runner=run, sleep=lambda s: None)


def test_command_environment_and_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-reach-the-cli")
    calls: list = []
    r = provider(ok("the answer"), calls=calls).complete(
        model="claude-sonnet-5-5", system="SYS", user="USER", params={"max_tokens": 4000, "effort": "low"})
    assert (r.text, r.input_tokens, r.output_tokens, r.stop_reason, r.request_id) == (
        "the answer", 703, 9, "end_turn", "s1")
    cmd, kw = calls[0]
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5-5" and cmd[cmd.index("--effort") + 1] == "low"
    assert kw["input"] == "USER"
    assert "ANTHROPIC_API_KEY" not in kw["env"]  # must bill the subscription, not the key
    assert kw["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4000"
    assert Path(kw["cwd"]).is_dir() and not any(Path(kw["cwd"]).iterdir())


def test_transient_errors_are_retried():
    p = provider(err("API Error: Overloaded", 529), subprocess.TimeoutExpired("claude", 1),
                 SimpleNamespace(stdout="not json", stderr="", returncode=1), ok("fine"))
    assert p.complete(model="m", system=None, user="u", params={}).text == "fine"


def test_persistent_transient_errors_become_infra():
    with pytest.raises(InfraError, match="after 4 attempts"):
        provider(*[err("Overloaded", 529)] * 4).complete(model="m", system=None, user="u", params={})


@pytest.mark.parametrize("outcome, match", [
    (err("Claude AI usage limit reached|1762560000"), "usage limit"),
    (err("Invalid API key · Please run /login", 401), "signed in"),
    (err("model: claude-nope not found", 404), "rejected"),
])
def test_errors_that_retrying_wont_fix_stop_the_run(outcome, match):
    with pytest.raises(FatalRunError, match=match):
        provider(outcome).complete(model="m", system=None, user="u", params={})


def test_missing_cli_is_a_config_error(monkeypatch):
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FatalRunError, match="claude` CLI"):
        ClaudeCodeProvider()


def test_cli_offers_the_provider_and_rejects_it_for_agent_suites(monkeypatch, tmp_path):
    from llm_evals.cli.main import app

    monkeypatch.setenv("CLAUDE_BIN", "/bin/claude-fake")
    root = Path(__file__).resolve().parents[2]
    res = CliRunner().invoke(app, ["gate", str(root / "suites" / "research-agent"), "--candidate",
                                   "versions/v2.yaml", "--provider", "claude-code", "--db", str(tmp_path / "s.sqlite")])
    assert res.exit_code != 0 and "agent_replay" in res.output
    res = CliRunner().invoke(app, ["gate", str(root / "suites" / "spec-writing"), "--candidate",
                                   "versions/v2.yaml", "--provider", "bogus", "--db", str(tmp_path / "s.sqlite")])
    assert res.exit_code != 0 and "claude-code" in res.output
