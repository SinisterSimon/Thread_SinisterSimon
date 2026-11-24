"""Did the shopper buy? A deliberately simple, documented model.

Real shoppers are messy; this model only needs to be plausible and, above all,
known. Because we decide the truth, later phases can check whether the
measurement method (a holdout group) recovers it.

Rules:
1. Shoppers the simulator marked as returning anyway always buy. The agent
   gets no real credit for them, but naive attribution would claim them.
2. "Just browsing" shoppers never buy.
3. Everyone else buys with probability
       persuadability x (0.5 if the agent resolved their objection else 0.05)
   plus a 0.15 bump for price-sensitive shoppers who were offered a discount.
   Objections that need a person only count as resolved on a handoff.
"""
import random
from dataclasses import dataclass

from app.shoppers.personas import Persona


@dataclass(frozen=True)
class Outcome:
    purchased: bool
    would_buy_anyway: bool
    resolved: bool


def decide(
    *,
    persona: Persona,
    agent_text: str,
    handed_off: bool,
    discount_pct: int,
    persuadability: int,
    returns_naturally: bool,
    rng: random.Random,
) -> Outcome:
    text = agent_text.lower()
    resolved = any(word in text for word in persona.resolved_by)
    if persona.needs_human:
        resolved = handed_off

    if returns_naturally:
        return Outcome(purchased=True, would_buy_anyway=True, resolved=resolved)
    if persona.objection == "none":
        return Outcome(purchased=False, would_buy_anyway=False, resolved=resolved)

    p = persuadability / 100 * (0.5 if resolved else 0.05)
    if persona.objection == "price" and discount_pct > 0:
        p += 0.15
    return Outcome(purchased=rng.random() < min(p, 1.0), would_buy_anyway=False, resolved=resolved)
