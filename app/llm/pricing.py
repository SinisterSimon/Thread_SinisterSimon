"""List prices per million tokens (USD, input / output), checked November 2026.

Used to put a cost on each agent, so a pricier model has to earn its keep in
true lift. Prices change: update this table, not the code that uses it. Costs
are kept in integer micro-dollars (1/1,000,000 of a dollar), the same rule as
money in cents elsewhere: no floating-point drift in sums.
"""

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100k tokens; Thread's are ~1k
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
}


def cost_micros(model: str, input_tokens: int, output_tokens: int) -> int:
    """What a call costs at list price, in micro-dollars. Unknown models cost 0 (and say so in the UI)."""
    price_in, price_out = PRICES_PER_MTOK.get(model.removeprefix("offline:"), (0.0, 0.0))
    # tokens x ($ per million tokens) = micro-dollars
    return round(input_tokens * price_in + output_tokens * price_out)


def has_price(model: str) -> bool:
    return model.removeprefix("offline:") in PRICES_PER_MTOK
