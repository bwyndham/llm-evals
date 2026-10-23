"""Anthropic Messages API adapter.

Requests send only `model`, `max_tokens`, `system` (when given), one user
message and, when `params["effort"]` is set, `output_config={"effort": ...}`.
They never send `temperature`/`top_p`/`top_k`, `thinking`, or an assistant
prefill: the 5.5 models reject those.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from llm_evals.runners.errors import FatalRunError, InfraError, classify_exception
from llm_evals.runners.providers.base import ProviderResponse
from llm_evals.types import ErrorKind

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000

# Params we send. Others are ignored with a warning; the rejected ones abort.
_SUPPORTED_PARAMS = {"max_tokens", "effort"}
_REJECTED_PARAMS = {"temperature", "top_p", "top_k", "thinking"}



def client_kwargs() -> dict:
    """Extra client settings from the environment.

    A key that isn't scoped to a workspace must send `anthropic-workspace-id` on
    every request; set ANTHROPIC_WORKSPACE_ID (e.g. in the repo's .env) to add it.
    """
    import os

    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    return {"default_headers": {"anthropic-workspace-id": workspace}} if workspace else {}

class AnthropicProvider:
    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 120.0,
        *,
        client: Any | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client

    @property
    def client(self) -> Any:
        # Built lazily so constructing the provider never needs credentials.
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(
                    max_retries=self.max_retries, timeout=self.timeout, **client_kwargs()
                )
            except anthropic.AnthropicError as exc:
                raise FatalRunError(f"could not create Anthropic client: {exc}") from exc
        return self._client

    @staticmethod
    def build_request(
        *, model: str, system: str | None, user: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        rejected = _REJECTED_PARAMS & params.keys()
        if rejected:
            raise FatalRunError(
                f"params {sorted(rejected)} are not supported on {model}; remove them "
                "from the manifest"
            )
        unknown = params.keys() - _SUPPORTED_PARAMS
        if unknown:
            log.warning(
                "params %s are not sent to the API (supported: %s)",
                sorted(unknown),
                sorted(_SUPPORTED_PARAMS),
            )
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": params.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": [{"role": "user", "content": user}],
        }
        if system is not None:
            request["system"] = system
        if "effort" in params:
            request["output_config"] = {"effort": params["effort"]}
        return request

    def complete(
        self, *, model: str, system: str | None, user: str, params: dict[str, Any]
    ) -> ProviderResponse:
        request = self.build_request(model=model, system=system, user=user, params=params)
        try:
            response = self.client.messages.create(**request)
        except FatalRunError:
            raise
        except anthropic.AnthropicError as exc:
            # The SDK has already retried per max_retries. classify_exception
            # raises FatalRunError for config problems.
            kind = classify_exception(exc)
            request_id = getattr(exc, "request_id", None)
            if kind is ErrorKind.INFRA:
                raise InfraError(f"{type(exc).__name__}: {exc}", request_id=request_id) from exc
            raise  # SYSTEM (e.g. 413): the runner records it as such
        text = "".join(b.text for b in response.content if b.type == "text")
        return ProviderResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=getattr(response, "_request_id", None),
            stop_reason=response.stop_reason,
        )
