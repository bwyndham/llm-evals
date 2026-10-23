"""Per-1M-token list prices, used to fill `Trace.cost_usd`."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# model -> (input $/1M tokens, output $/1M tokens)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Haiku 5.5's price above holds for prompts up to this many input tokens.
HAIKU_PRICE_INPUT_LIMIT = 100_000
# ...and this price applies above it.
HAIKU_LONG_PROMPT_PRICES = (0.50, 2.50)

_warned: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call. Unknown models cost 0.0 and log a warning once."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("no price for model %r; cost recorded as 0.0", model)
        return 0.0
    in_price, out_price = prices
    if model == "claude-haiku-5-5" and input_tokens > HAIKU_PRICE_INPUT_LIMIT:
        in_price, out_price = HAIKU_LONG_PROMPT_PRICES
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
