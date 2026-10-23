"""JudgeClient: render a judge prompt, call the text Provider, parse, cache.

The user prompt is the rendered user template followed by ONE fenced json
block holding the payload (`json.dumps(payload, sort_keys=True)`), always
last, so the offline mock judge can read the task from it:

    <rendered user template>

    ```json
    {"task": "...", ...}
    ```

Backticks inside payload strings are written as the JSON escape \\u0060, so a
ticket that itself contains a code fence can't end the block early; the
payload still decodes to exactly the same object.

Caching: the key is `prompt_hash` = sha256(model, system, user)[:16]. Every
response the provider returns is cached, parseable or not, and a parse
failure is never retried: it raises `JudgeError` (now and on every rerun of
the same prompt). Provider exceptions propagate unchanged and are not cached;
the calling check classifies them (infra vs config).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import jinja2
from pydantic import BaseModel, ValidationError

from llm_evals.judges.cache import JudgeCache
from llm_evals.runners.errors import FatalRunError
from llm_evals.runners.prompt import render_template, strip_json_fence
from llm_evals.runners.providers.base import Provider
from llm_evals.runners.providers.pricing import cost_usd

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
DEFAULT_JUDGE_PARAMS: dict[str, Any] = {"max_tokens": 4000, "effort": "medium"}

# Stop reasons that mean the judge's answer can't be trusted.
_BAD_STOP_REASONS = {"refusal", "max_tokens"}

_LAST_JSON_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\n(.*?)\n?```", re.DOTALL)

M = TypeVar("M", bound=BaseModel)


class JudgeError(RuntimeError):
    """The judge answered, but the answer is unusable (bad JSON, wrong shape, refusal)."""

    def __init__(self, message: str, *, raw_text: str = "", prompt_hash: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.prompt_hash = prompt_hash


@dataclass(frozen=True)
class JudgeResult(Generic[M]):
    parsed: M
    raw_text: str
    prompt_hash: str  # sha256(model + system + user)[:16]: the cache key for this call
    model: str
    cost_usd: float  # spend incurred by THIS call: 0.0 on a cache hit
    input_tokens: int  # tokens of the original provider call (also on a cache hit)
    output_tokens: int
    cached: bool


def prompt_hash(model: str, system: str | None, user: str) -> str:
    """sha256 over model, system and user prompt (NUL-separated), first 16 hex chars."""
    h = hashlib.sha256()
    for part in (model, system or "", user):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def unit_hash(template_path: Path | str, item_text: str = "", *, system_template: Path | str | None = None) -> str:
    """GatingUnit.prompt_hash: sha256(template bytes + item text)[:12].

    With `system_template`, its bytes are hashed too (template bytes, then
    system bytes, then item text), so editing the judge's system prompt also
    resets calibration. Without it this is exactly the contract formula.
    """
    h = hashlib.sha256(Path(template_path).read_bytes())
    if system_template is not None:
        h.update(Path(system_template).read_bytes())
    h.update(item_text.encode("utf-8"))
    return h.hexdigest()[:12]


def payload_block(payload: dict[str, Any]) -> str:
    """The fenced json block appended to every judge user prompt."""
    body = json.dumps(payload, sort_keys=True, default=str).replace("`", "\\u0060")
    return f"```json\n{body}\n```\n"


def build_user_prompt(rendered: str, payload: dict[str, Any]) -> str:
    rendered = rendered.rstrip()
    return (f"{rendered}\n\n" if rendered else "") + payload_block(payload)


def _extract_json(text: str) -> Any:
    """Parse the judge's JSON: the whole text (fence stripped), else the last
    fenced block, else the outermost {...} span."""
    candidates = [strip_json_fence(text)]
    fences = _LAST_JSON_FENCE.findall(text)
    if fences:
        candidates.append(fences[-1])
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    last_exc: Exception | None = None
    for body in candidates:
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            last_exc = exc
    # A judge sometimes answers, writes "Correction: ..." and answers again. The last
    # complete object is its final answer.
    objects = _top_level_objects(strip_json_fence(text))
    if objects:
        return objects[-1]
    raise ValueError(f"judge output is not valid JSON: {last_exc}")


def _top_level_objects(text: str) -> list[Any]:
    """Every complete JSON object in `text`, outermost first, skipping prose between them."""
    decoder, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, i = decoder.raw_decode(text, i)
            out.append(obj)
        except json.JSONDecodeError:
            i += 1
    return out


class JudgeClient:
    """Calls a judge model through a text `Provider`, with a SQLite cache.

    `total_cost_usd`, `n_calls` (provider calls made) and `n_cached` (cache
    hits) are cumulative and thread-safe, so the engine can report judge spend.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        params: dict[str, Any] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.params = dict(DEFAULT_JUDGE_PARAMS if params is None else params)
        self.cache = JudgeCache(cache_path)
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._total_cost_usd = 0.0
        self._n_calls = 0
        self._n_cached = 0

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    @property
    def n_calls(self) -> int:
        with self._lock:
            return self._n_calls

    @property
    def n_cached(self) -> int:
        with self._lock:
            return self._n_cached

    def render(
        self,
        system_template: Path | None,
        user_template: Path,
        variables: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[str | None, str]:
        """The (system, user) prompts a `judge` call would send."""
        try:
            system = render_template(str(system_template), variables) if system_template is not None else None
            user = render_template(str(user_template), variables)
        except jinja2.UndefinedError as exc:
            raise FatalRunError(f"judge template {user_template} needs a variable that wasn't given: {exc}") from exc
        return system, build_user_prompt(user, payload)

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def judge(
        self,
        system_template: Path | None,
        user_template: Path,
        variables: dict[str, Any],
        payload: dict[str, Any],
        response_model: type[M],
    ) -> JudgeResult[M]:
        """Render, call (or hit the cache), parse into `response_model`.

        Raises `JudgeError` when the answer can't be used, `FatalRunError` for
        template problems, and whatever the provider raises for call failures.
        """
        system, user = self.render(system_template, user_template, variables, payload)
        key = prompt_hash(self.model, system, user)

        # One in-flight call per key: concurrent identical judgments share one call.
        with self._key_lock(key):
            hit = self.cache.get(key)
            if hit is not None:
                with self._lock:
                    self._n_cached += 1
                raw, stop, in_tok, out_tok, cost, cached = (
                    hit.raw_text, hit.stop_reason, hit.input_tokens, hit.output_tokens, 0.0, True,
                )
            else:
                response = self.provider.complete(model=self.model, system=system, user=user, params=dict(self.params))
                raw, stop = response.text, response.stop_reason
                in_tok, out_tok = response.input_tokens, response.output_tokens
                cost, cached = cost_usd(self.model, in_tok, out_tok), False
                self.cache.put(
                    key, model=self.model, raw_text=raw, input_tokens=in_tok, output_tokens=out_tok,
                    stop_reason=stop, cost_usd=cost,
                )
                with self._lock:
                    self._n_calls += 1
                    self._total_cost_usd += cost

        if stop in _BAD_STOP_REASONS:
            raise JudgeError(f"judge stopped with stop_reason={stop}", raw_text=raw, prompt_hash=key)
        try:
            data = _extract_json(raw)
            parsed = response_model.model_validate(data)
        except ValueError as exc:  # JSONDecodeError and ValidationError are ValueErrors
            detail = str(exc) if not isinstance(exc, ValidationError) else f"wrong shape: {exc}"
            raise JudgeError(detail, raw_text=raw, prompt_hash=key) from exc
        return JudgeResult(
            parsed=parsed, raw_text=raw, prompt_hash=key, model=self.model, cost_usd=cost,
            input_tokens=in_tok, output_tokens=out_tok, cached=cached,
        )


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_PARAMS",
    "JudgeClient",
    "JudgeError",
    "JudgeResult",
    "build_user_prompt",
    "payload_block",
    "prompt_hash",
    "unit_hash",
]
