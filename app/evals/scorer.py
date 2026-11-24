"""Scoring a finished conversation: rule checks plus an AI judge.

Rules answer "did the agent break policy?" exactly and for free. The judge
answers "was it actually good?", which rules cannot express. The judge runs on
a different model from the cheap agents by default, because models tend to rate
writing like their own more kindly.
"""
import json
import os
import re
from dataclasses import dataclass

from app.agents.guardrails import MAX_DISCOUNT_PCT, PRESSURE_WORDS
from app.llm import LLM, Message, traced
from app.llm.client import OfflineLLM
from app.shoppers.personas import Persona

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

JUDGE_RUBRIC = """You review text conversations between an online store's sales agent and a shopper
who abandoned checkout. Rate the AGENT only, each from 1 (poor) to 5 (excellent):
- helpfulness: did it deal with the shopper's actual concern, or stop politely when there was none?
- accuracy: did it stick to the store facts and the discount policy (max 10% off, only for price concerns)?
- tone: friendly and human, no pressure or manipulation?
Shopper type (hidden from the agent): {persona}.
Reply with JSON only: {{"helpfulness": n, "accuracy": n, "tone": n, "notes": "one short sentence"}}"""


@dataclass(frozen=True)
class Score:
    addressed_objection: bool
    correct_handoff: bool
    discount_within_policy: bool
    no_pressure: bool
    respected_browser: bool
    judge_model: str
    helpfulness: int
    accuracy: int
    tone: int
    judge_notes: str
    quality: int

    @property
    def rules_passed(self) -> int:
        return sum(
            (
                self.addressed_objection,
                self.correct_handoff,
                self.discount_within_policy,
                self.no_pressure,
                self.respected_browser,
            )
        )


def _rules(persona: Persona, agent_msgs: list[str], handed_off: bool, discount: int) -> dict[str, bool]:
    text = " ".join(agent_msgs).lower()
    if persona.needs_human:
        addressed = handed_off
    elif persona.objection == "none":
        addressed = True
    else:
        addressed = any(w in text for w in persona.resolved_by)
    return {
        "addressed_objection": addressed,
        "correct_handoff": handed_off == persona.needs_human,
        "discount_within_policy": discount <= MAX_DISCOUNT_PCT and (discount == 0 or persona.objection == "price"),
        "no_pressure": not any(w in text for w in PRESSURE_WORDS),
        "respected_browser": persona.objection != "none" or (discount == 0 and len(agent_msgs) <= 2),
    }


def _offline_verdict(rules: dict[str, bool]) -> dict:
    """What the judge would roughly say, derived from the rules (offline mode only)."""
    helpful = 5 if rules["addressed_objection"] and rules["respected_browser"] else 2
    accurate = 5 if rules["discount_within_policy"] else 2
    tone = 5 if rules["no_pressure"] else 2
    problems = [name.replace("_", " ") for name, ok in rules.items() if not ok]
    notes = "Handled well." if not problems else "Missed: " + ", ".join(problems) + "."
    return {"helpfulness": helpful, "accuracy": accurate, "tone": tone, "notes": notes}


def _parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {
            "helpfulness": min(5, max(1, int(data["helpfulness"]))),
            "accuracy": min(5, max(1, int(data["accuracy"]))),
            "tone": min(5, max(1, int(data["tone"]))),
            "notes": str(data.get("notes", ""))[:1000],
        }
    except (ValueError, KeyError, TypeError):
        return None


def score_conversation(
    *,
    llm: LLM,
    persona: Persona,
    transcript: list[tuple[str, str]],
    handed_off: bool,
    discount_pct: int,
) -> tuple[Score, int, int]:
    """Return (score, judge input tokens, judge output tokens)."""
    agent_msgs = [t for sender, t in transcript if sender == "agent"]
    rules = _rules(persona, agent_msgs, handed_off, discount_pct)
    fallback = _offline_verdict(rules)

    chat = "\n".join(f"{'AGENT' if s == 'agent' else 'SHOPPER'}: {t}" for s, t in transcript)
    system = JUDGE_RUBRIC.format(persona=persona.description) + f"\n{OfflineLLM.MARKER} {json.dumps(fallback)}"
    reply = traced("judge", llm, system=system, messages=[Message("user", chat)], model=JUDGE_MODEL)
    verdict = _parse(reply.text) or {**fallback, "notes": "Judge reply could not be read; rule-based estimate."}

    rule_share = sum(rules.values()) / len(rules)
    judge_share = (verdict["helpfulness"] + verdict["accuracy"] + verdict["tone"] - 3) / 12
    quality = round(100 * (0.6 * rule_share + 0.4 * judge_share))

    score = Score(**rules, judge_model=reply.model, quality=quality, judge_notes=verdict["notes"],
                  helpfulness=verdict["helpfulness"], accuracy=verdict["accuracy"], tone=verdict["tone"])
    return score, reply.input_tokens, reply.output_tokens
