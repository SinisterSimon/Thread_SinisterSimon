"""Shopper personalities.

Each abandoned checkout is played by one persona. A persona has a reason for
leaving (its objection) and words that show the agent actually dealt with it.
The outcome model uses those to decide whether a good conversation tipped the
shopper into buying.
"""
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    description: str
    objection: str
    # Lines the shopper says, in order, when replying offline.
    lines: tuple[str, ...]
    # Words that show the agent addressed the objection (case-insensitive).
    resolved_by: tuple[str, ...]
    # Should a good agent hand this shopper to a human?
    needs_human: bool = False
    weight: int = 1


PERSONAS: tuple[Persona, ...] = (
    Persona(
        key="price",
        name="Price-sensitive",
        description="Liked the product but balked at the total.",
        objection="price",
        lines=("Honestly it's a bit pricey for me.", "Is there any deal right now?", "Hmm, ok."),
        resolved_by=("discount", "% off", "code", "free shipping", "bundle"),
        weight=3,
    ),
    Persona(
        key="sizing",
        name="Unsure about fit",
        description="Not sure which size or variant is right.",
        objection="fit",
        lines=("Not sure about the size, does it run small?", "I'm usually between sizes.", "Ok thanks."),
        resolved_by=("size", "fit", "exchange", "true to size", "measure"),
        weight=2,
    ),
    Persona(
        key="shipping",
        name="Worried about delivery",
        description="Needs it by a date and doesn't know if it'll arrive.",
        objection="delivery",
        lines=("When would it actually arrive?", "I need it before Friday.", "Alright."),
        resolved_by=("deliver", "arrive", "ship", "days", "tracking"),
        weight=2,
    ),
    Persona(
        key="browsing",
        name="Just browsing",
        description="Had no real intent to buy.",
        objection="none",
        lines=("Oh I was just looking around.", "Maybe later.", "Bye."),
        resolved_by=(),
        weight=3,
    ),
    Persona(
        key="angry",
        name="Frustrated",
        description="Had a bad experience and wants a person.",
        objection="complaint",
        lines=("Your checkout charged me twice last time.", "I want to talk to a real person.", "..."),
        resolved_by=("team", "person", "human", "colleague"),
        needs_human=True,
        weight=1,
    ),
)


def persona_for(checkout_id: int, seed: int = 0) -> Persona:
    """Stable persona per checkout: the same checkout always gets the same one."""
    rng = random.Random(f"{seed}:{checkout_id}")
    return rng.choices(PERSONAS, weights=[p.weight for p in PERSONAS])[0]
