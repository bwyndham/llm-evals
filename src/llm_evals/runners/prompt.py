"""The prompt runner: render templates, call a provider, parse JSON output."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2

from llm_evals.registry import register_runner
from llm_evals.runners.errors import FatalRunError, classify_exception
from llm_evals.runners.providers.base import Provider, ProviderResponse
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.runners.schemas import validator_for
from llm_evals.types import Case, ErrorKind, Result, Trace, VersionManifest

# Stop reasons that mean the output can't be trusted.
SYSTEM_STOP_REASONS = {"refusal", "max_tokens"}

_FENCE = re.compile(r"^\s*```(?:json|JSON)?[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)

_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


@lru_cache(maxsize=256)
def _load_template(path: str, mtime_ns: int) -> jinja2.Template:  # mtime busts the cache
    return _ENV.from_string(Path(path).read_text(encoding="utf-8"))


def render_template(path: str, variables: dict[str, Any]) -> str:
    p = Path(path)
    try:
        template = _load_template(str(p), p.stat().st_mtime_ns)
    except FileNotFoundError:
        raise FatalRunError(f"prompt template not found: {p}") from None
    except jinja2.TemplateSyntaxError as exc:
        raise FatalRunError(f"prompt template {p} has a syntax error: {exc}") from exc
    return template.render(**variables)


def strip_json_fence(text: str) -> str:
    """Remove one surrounding ```json ... ``` (or bare ```) fence, if present."""
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def parse_output(text: str, schema: str | None) -> dict[str, Any]:
    """Parse JSON output and validate it. Raises ValueError with a clear message."""
    body = strip_json_fence(text)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"output JSON must be an object, got {type(data).__name__}")
    if schema is None:
        return data
    return validator_for(schema)(data)


@register_runner("prompt")
class PromptRunner:
    """Single-turn runner. Roles: `user` (required) and `system` (optional).

    `provider` defaults to `AnthropicProvider()`. Template variables are `case.input`. Missing variables (StrictUndefined)
    and unreadable templates raise FatalRunError: they're config problems.
    """

    def __init__(self, provider: Provider | None = None, output_schema: str | None = None) -> None:
        if provider is None:  # default to the real API
            from llm_evals.runners.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider()
        self.provider = provider
        self.output_schema = output_schema
        if output_schema is not None:
            try:
                validator_for(output_schema)  # fail fast on a typo or a broken schema file
            except (KeyError, FileNotFoundError, ValueError) as exc:
                raise FatalRunError(str(exc)) from exc

    def render(self, case: Case, version: VersionManifest) -> tuple[str | None, str]:
        templates = version.prompt_templates
        if "user" not in templates:
            raise FatalRunError(f"version {version.name!r} has no 'user' prompt template")
        unknown = set(templates) - {"system", "user"}
        if unknown:
            raise FatalRunError(
                f"version {version.name!r}: prompt runner supports roles system and user, "
                f"not {sorted(unknown)}"
            )
        try:
            system = (
                render_template(templates["system"], case.input) if "system" in templates else None
            )
            user = render_template(templates["user"], case.input)
        except jinja2.UndefinedError as exc:
            raise FatalRunError(
                f"case {case.id!r}: template for version {version.name!r} needs a variable "
                f"the case input lacks: {exc}"
            ) from exc
        return system, user

    def run(self, case: Case, version: VersionManifest, sample_idx: int, run_id: str) -> Result:
        system, user = self.render(case, version)
        result = Result(
            run_id=run_id, case_id=case.id, version_id=version.id, sample_idx=sample_idx
        )

        start = time.perf_counter()
        try:
            response: ProviderResponse = self.provider.complete(
                model=version.model, system=system, user=user, params=dict(version.params)
            )
        except FatalRunError:
            raise
        except Exception as exc:
            result.trace.latency_ms = (time.perf_counter() - start) * 1000
            kind = classify_exception(exc)  # raises FatalRunError for config problems
            request_id = getattr(exc, "request_id", None)
            if request_id:
                result.trace.request_ids.append(request_id)
            result.error_kind = kind
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        latency_ms = (time.perf_counter() - start) * 1000

        result.raw_output = response.text
        result.trace = Trace(
            model=version.model,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost_usd(version.model, response.input_tokens, response.output_tokens),
            request_ids=[response.request_id] if response.request_id else [],
        )

        if response.stop_reason in SYSTEM_STOP_REASONS:
            result.error_kind = ErrorKind.SYSTEM
            result.error = f"stop_reason={response.stop_reason}"
            return result
        try:
            result.output = parse_output(response.text, self.output_schema)
        except ValueError as exc:
            result.error_kind = ErrorKind.SYSTEM
            result.error = str(exc)
        return result


__all__ = ["PromptRunner", "parse_output", "render_template", "strip_json_fence"]
