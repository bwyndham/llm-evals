"""RUN-11: error classification and the run-completeness rule.

Providers translate their SDK exceptions into two of our own:

- `InfraError`: the provider failed after its own retries (timeouts, 5xx,
  rate limits). The sample is recorded with `ErrorKind.INFRA` and excluded
  from scoring.
- `FatalRunError`: a configuration problem (bad key, unknown model, a request
  the API rejects as malformed, a template that needs a variable the case
  lacks). Every sample would fail the same way, so the run aborts instead of
  recording per-sample failures.

SYSTEM errors (unparseable output, schema mismatch, refusals, max_tokens) are
decided by the runner from the response, not from exceptions.
"""

from __future__ import annotations

import logging

from llm_evals.types import ErrorKind, Result

log = logging.getLogger(__name__)

DEFAULT_MAX_INFRA_SHARE = 0.05


class FatalRunError(RuntimeError):
    """A config problem: abort the run rather than record per-sample failures."""


class InfraError(RuntimeError):
    """The provider still failed after retries. The sample is excluded, not failed."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


def _classify_anthropic(exc: BaseException) -> ErrorKind | None:
    """Map raw Anthropic SDK exceptions, most specific first. None if not one of them."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a hard dependency
        return None

    # Config problems: the same request will fail for every sample.
    if isinstance(
        exc,
        (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ),
    ):
        raise FatalRunError(f"{type(exc).__name__}: {exc}") from exc
    # APITimeoutError subclasses APIConnectionError; both are INFRA.
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return ErrorKind.INFRA
    if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError)):
        return ErrorKind.INFRA
    if isinstance(exc, anthropic.ConflictError):  # 409: transient
        return ErrorKind.INFRA
    if isinstance(exc, anthropic.RequestTooLargeError):
        # 413 depends on this case's input size: the system can't handle the input.
        return ErrorKind.SYSTEM
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:  # 503, 504, 529 and any other 5xx
            return ErrorKind.INFRA
        # Other 4xx (e.g. 422): the request itself is wrong, so it's config.
        raise FatalRunError(f"{type(exc).__name__} ({exc.status_code}): {exc}") from exc
    if isinstance(exc, anthropic.CredentialsError):
        raise FatalRunError(f"Anthropic credentials could not be loaded: {exc}") from exc
    if isinstance(exc, anthropic.APIResponseValidationError):
        return ErrorKind.INFRA
    return None


def classify_exception(exc: BaseException) -> ErrorKind:
    """Return the ErrorKind for an exception raised while calling the system.

    Raises `FatalRunError` (chained to `exc`) for config problems. Unknown
    exception types are treated as INFRA so they're excluded and, if common,
    push the run to INCOMPLETE rather than being counted against the version.
    """
    if isinstance(exc, FatalRunError):
        raise exc
    if isinstance(exc, InfraError):
        return ErrorKind.INFRA
    kind = _classify_anthropic(exc)
    if kind is not None:
        return kind
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ErrorKind.INFRA
    log.warning("unclassified exception treated as INFRA: %r", exc)
    return ErrorKind.INFRA


def infra_share(results: list[Result]) -> float:
    """Share of samples that ended in an INFRA error (0.0 for no results)."""
    if not results:
        return 0.0
    return sum(r.error_kind is ErrorKind.INFRA for r in results) / len(results)


def is_incomplete(
    results: list[Result], max_infra_share: float = DEFAULT_MAX_INFRA_SHARE
) -> bool:
    """True when too many samples were lost to infra errors to trust a decision.

    An empty result list is incomplete.
    """
    if not results:
        return True
    return infra_share(results) > max_infra_share
