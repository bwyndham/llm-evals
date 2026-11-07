"""Model providers the prompt runner can call."""

from llm_evals.runners.providers.base import FakeProvider, Provider, ProviderResponse
from llm_evals.runners.providers.pricing import cost_usd
from llm_evals.runners.providers.mock import MockProvider  # noqa: F401

__all__ = ["FakeProvider", "Provider", "ProviderResponse", "cost_usd", "AnthropicProvider", "ClaudeCodeProvider"]


def __getattr__(name: str):  # lazy: importing the SDK isn't needed for fakes
    if name == "AnthropicProvider":
        from llm_evals.runners.providers.anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "ClaudeCodeProvider":
        from llm_evals.runners.providers.claude_code import ClaudeCodeProvider

        return ClaudeCodeProvider
    raise AttributeError(name)
