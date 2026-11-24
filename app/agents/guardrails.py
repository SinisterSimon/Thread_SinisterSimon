"""Guardrails: fix an agent's message before the shopper sees it.

Guardrails run outside the model, so they hold even when a prompt is ignored.
They are deliberately few and easy to explain:

1. Discount cap: never more than MAX_DISCOUNT_PCT, and at most one offer per chat.
2. No pressure: sentences with urgency tricks are removed.
3. Forced handoff: if the shopper asks for a person or reports a billing problem
   and the agent did not hand off, the guardrail hands off.

Each change counts as one "fix", so experiments show how often an agent needed
correcting, not only how it performed after correction.
"""
import re
from dataclasses import replace

from app.agents.base import DISCOUNT_RE, AgentTurn

MAX_DISCOUNT_PCT = 10
PRESSURE_WORDS = ("expire", "ends very soon", "hurry", "don't miss", "next hour", "last chance", "act now")
ESCALATION_WORDS = ("real person", "human", "charged", "twice", "refund", "complain")
HANDOFF_LINE = "I'm passing this to a colleague from our team, who will follow up with you today."
FALLBACK_LINE = "Happy to help if you have any questions about your order."

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


def apply(turn: AgentTurn, *, last_shopper_text: str | None, discount_already_offered: bool) -> tuple[AgentTurn, int]:
    fixes = 0
    sentences = _sentences(turn.text)

    # 1. Discount cap and single offer.
    kept = []
    for s in sentences:
        found = [int(n) for n in DISCOUNT_RE.findall(s)]
        if not found:
            kept.append(s)
        elif discount_already_offered:
            fixes += 1  # repeat offer: drop the sentence
        elif max(found) > MAX_DISCOUNT_PCT:
            kept.append(DISCOUNT_RE.sub(f"{MAX_DISCOUNT_PCT}% off", s))
            fixes += 1
        else:
            kept.append(s)
    sentences = kept

    # 2. Pressure tactics.
    calm = [s for s in sentences if not any(w in s.lower() for w in PRESSURE_WORDS)]
    if len(calm) != len(sentences):
        fixes += 1
    sentences = calm or [FALLBACK_LINE]

    text = " ".join(sentences)
    handoff = turn.handoff

    # 3. Forced handoff.
    if last_shopper_text and not handoff and any(w in last_shopper_text.lower() for w in ESCALATION_WORDS):
        text, handoff = HANDOFF_LINE, True
        fixes += 1

    found = [int(n) for n in DISCOUNT_RE.findall(text)]
    return replace(turn, text=text, handoff=handoff, discount_pct=max(found, default=0)), fixes
