"""`ClaudeCodeProvider`: model calls through the `claude` CLI, on a Claude subscription.

Selected with `--provider claude-code`. Instead of the Messages API and an API key,
each call runs one headless Claude Code request, signed in with the user's
claude.ai account (Pro or Max), so runs count against the plan's usage limits
rather than API billing:

    claude -p --system-prompt <system> --tools "" --output-format json --model <model> ...

What the model sees matches the API path as closely as the CLI allows:
- `--system-prompt` replaces Claude Code's own system prompt; tools are off;
  settings, MCP servers, slash commands and skills are skipped; the session
  isn't saved; and each call runs in an empty temporary folder.
- ANTHROPIC_API_KEY (and ANTHROPIC_AUTH_TOKEN) are removed from the child's
  environment. With a key present the CLI bills the key instead of the plan.
- `params["max_tokens"]` is passed as CLAUDE_CODE_MAX_OUTPUT_TOKENS and
  `params["effort"]` as `--effort`. Other params are ignored.

Errors map onto the run's error rules (RUN-11):
- overload, rate limits, timeouts and unparseable CLI output are retried with
  backoff, then raised as `InfraError` (the sample is excluded, not failed);
- reaching the plan's usage limit, not being signed in, or a rejected model
  raise `FatalRunError`: every remaining call would fail the same way, so the
  run stops with a clear message instead of recording failures.

Token counts come from the CLI's usage report (cache reads and writes count as
input), so `Trace.cost_usd` is a list-price equivalent, not a charge.

Text calls only: there's no tool-use support, so the `agent_replay` runner
can't use this provider.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

from llm_evals.runners.errors import FatalRunError, InfraError
from llm_evals.runners.providers.base import ProviderResponse

_LIMIT_WORDS = ("usage limit", "limit reached", "out of extra usage", "weekly limit", "session limit")
_REMOVED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
DEFAULT_SYSTEM = "You are a helpful assistant."


class ClaudeCodeProvider:
    def __init__(self, max_retries: int = 3, timeout: float = 300.0, *, binary: str | None = None,
                 runner: Any = subprocess.run, sleep: Any = time.sleep) -> None:
        self.binary = binary or os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        if not self.binary:
            raise FatalRunError("--provider claude-code needs the `claude` CLI on PATH: install Claude Code "
                                "and sign in with `claude auth login`")
        self.max_retries = max_retries
        self.timeout = timeout
        self._run = runner
        self._sleep = sleep
        self._cwd = tempfile.mkdtemp(prefix="llm-evals-claude-")

    def command(self, *, model: str, system: str | None, params: dict[str, Any]) -> list[str]:
        cmd = [self.binary, "-p", "--output-format", "json", "--model", model, "--tools", "",
               "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config",
               "--disable-slash-commands", "--system-prompt", system or DEFAULT_SYSTEM]
        if params.get("effort"):
            cmd += ["--effort", str(params["effort"])]
        return cmd

    def env(self, params: dict[str, Any]) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _REMOVED_ENV}
        if params.get("max_tokens"):
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(params["max_tokens"])
        return env

    def complete(self, *, model: str, system: str | None, user: str, params: dict[str, Any]) -> ProviderResponse:
        cmd, env = self.command(model=model, system=system, params=params), self.env(params)
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                proc = self._run(cmd, input=user, capture_output=True, text=True, env=env,
                                 cwd=self._cwd, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                last = f"claude CLI timed out after {self.timeout:.0f}s"
            else:
                outcome = self._parse(proc)
                if isinstance(outcome, ProviderResponse):
                    return outcome
                last = outcome
            if attempt < self.max_retries:
                self._sleep(min(60.0, 5.0 * 2 ** attempt))
        raise InfraError(f"claude CLI failed after {self.max_retries + 1} attempts: {last}")

    def _parse(self, proc: Any) -> ProviderResponse | str:
        """A response, or a retryable error message. Raises FatalRunError for errors retrying won't fix."""
        try:
            data = json.loads(proc.stdout or "")
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            msg = (proc.stderr or proc.stdout or "").strip()[:500] or f"exit code {proc.returncode}"
            _raise_if_fatal(msg)
            return msg
        if data.get("is_error") or data.get("subtype") not in (None, "success"):
            msg = str(data.get("result") or data.get("subtype") or "error")[:500]
            status = data.get("api_error_status")
            _raise_if_fatal(msg, status)
            return f"{status or ''} {msg}".strip()
        usage = data.get("usage") or {}
        input_tokens = sum(int(usage.get(k) or 0) for k in
                           ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return ProviderResponse(text=data.get("result") or "", input_tokens=input_tokens,
                                output_tokens=int(usage.get("output_tokens") or 0),
                                request_id=data.get("session_id"), stop_reason=data.get("stop_reason"))


def _raise_if_fatal(msg: str, status: Any = None) -> None:
    low = msg.lower()
    if any(w in low for w in _LIMIT_WORDS):
        raise FatalRunError(f"Claude subscription usage limit reached; rerun after it resets ({msg})")
    if status in (401, 403) or "not logged in" in low or "please run /login" in low or "invalid api key" in low:
        raise FatalRunError(f"the claude CLI isn't signed in: run `claude auth login` ({msg})")
    if status in (400, 404) or ("model" in low and ("not found" in low or "invalid" in low)):
        raise FatalRunError(f"the claude CLI rejected the request: {msg}")


__all__ = ["ClaudeCodeProvider"]
