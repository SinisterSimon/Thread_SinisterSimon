"""The contract every sales agent follows.

An agent gets the conversation so far plus facts about the store and cart,
and returns its next message. Anything that implements `respond` can be
plugged into an experiment, which is what makes agents comparable.
"""
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.llm import Message

HANDOFF_TAG = "[HANDOFF]"
END_TAG = "[END]"
DISCOUNT_RE = re.compile(r"(\d{1,2})\s*%\s*off", re.IGNORECASE)


@dataclass(frozen=True)
class AgentContext:
    store_name: str
    first_name: str
    cart_lines: list[str]
    cart_total: str
    store_facts: list[str] = field(default_factory=list)
    # Thread appends the checkout link after the agent's first text; the agent never writes URLs.
    checkout_url: str | None = None


@dataclass(frozen=True)
class AgentTurn:
    text: str
    handoff: bool
    end: bool
    discount_pct: int
    input_tokens: int
    output_tokens: int


class Agent(Protocol):
    key: str
    label: str
    model: str

    def respond(self, history: list[Message], ctx: AgentContext) -> AgentTurn: ...


def parse_turn(raw: str, input_tokens: int, output_tokens: int) -> AgentTurn:
    """Pull structured signals out of the agent's free text.

    Handoffs and conversation endings are signalled with tags the agent is told to use; the tag is
    stripped before the shopper sees the message. Discounts are read from
    phrases like "10% off".
    """
    handoff = HANDOFF_TAG in raw
    end = END_TAG in raw
    text = raw.replace(HANDOFF_TAG, "").replace(END_TAG, "").strip()
    found = [int(n) for n in DISCOUNT_RE.findall(text)]
    return AgentTurn(
        text=text,
        handoff=handoff,
        end=end,
        discount_pct=max(found, default=0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
