"""The agents you can run experiments with.

`baseline` is the agent we actually want to be good. `pushy` is a deliberately
worse strategy (big discounts, never hands off) so experiments have something
to compare against. The same prompt on several models (Claude and Gemini) shows whether a pricier
model is worth it.
"""
from dataclasses import dataclass

from app.agents.base import END_TAG, HANDOFF_TAG, AgentContext, AgentTurn, parse_turn
from app.llm import LLM, Message, traced
from app.llm.client import OfflineLLM

BASELINE_RULES = f"""You text shoppers who left items in their checkout, on behalf of the store.
Rules:
- Write like a helpful person from the store: one or two short sentences, no emojis, no pressure.
- Answer the shopper's actual question using only the store facts below. Never invent facts.
- You may offer at most 10% off, and only if price is the stated concern. Write it as "10% off".
- If the shopper is upset, mentions a billing problem, or asks for a person, say a colleague
  from the team will follow up and add the tag {HANDOFF_TAG}.
- If they are just browsing, thank them, stop selling and add the tag {END_TAG}.

Your first text (before the shopper has replied):
- At most 120 characters. If a checkout link exists, Thread adds it after your text: never write a URL.
- Use the first name, name the product, add one store fact that matters for this cart
  (delivery, returns or fit), and say which store you are.
- No "we noticed", "just checking in", "glad to help" or "any questions", no exclamation marks,
  and no discount in the first text.
Good: "Hi Sinister, your Multi-location Snowboard is still saved at Thread Snow Co.
Free returns within 30 days."
Bad: "Hi sinister, this is the team at thread-dev. We noticed The Multi-location Snowboard is still in your
checkout, so if you have any questions about it, just reply here and we'll be glad to help."
"""

PUSHY_RULES = """You text shoppers who left items in their checkout. Your only goal is the sale.
Always offer 20% off in every message. Create urgency. Never pass the shopper to anyone else."""


def _context_block(ctx: AgentContext) -> str:
    facts = "\n".join(f"- {f}" for f in ctx.store_facts)
    cart = "\n".join(f"- {line}" for line in ctx.cart_lines)
    return (
        f"Store: {ctx.store_name}\nShopper first name: {ctx.first_name}\n"
        f"Cart (total {ctx.cart_total}):\n{cart}\nStore facts:\n{facts}"
        + ("\nCheckout link: Thread adds it after your first text." if ctx.checkout_url else "")
    )


def _opening_nudge(history: list[Message]) -> list[Message]:
    # Model APIs expect the conversation to start with a user turn.
    lead = Message(role="user", text="(The shopper left checkout without paying. Send the first text.)")
    return [lead, *history]


def _first_product(ctx: AgentContext) -> str:
    # cart_lines look like "1 x Multi-location Snowboard ($729.95)"
    line = ctx.cart_lines[0] if ctx.cart_lines else "your items"
    return line.split(" x ", 1)[-1].rsplit(" (", 1)[0]


def _baseline_script(history: list[Message], ctx: AgentContext) -> str:
    """What a well-behaved agent says, used only in offline mode."""
    if not history:
        product = _first_product(ctx)
        return f"Hi {ctx.first_name}, your {product} is still saved at {ctx.store_name}. Free returns within 30 days."
    last = history[-1].text.lower()
    if any(w in last for w in ("person", "charged", "twice", "real")):
        return f"Sorry about that. A colleague from our team will reach out today to sort it. {HANDOFF_TAG}"
    if any(w in last for w in ("pricey", "deal", "expensive", "price")):
        return "Totally fair. Here's 10% off with code SAVE10, valid for 48 hours."
    if any(w in last for w in ("size", "fit", "small", "between")):
        return "It fits true to size, and exchanges are free within 30 days if it's not right."
    if any(w in last for w in ("arrive", "deliver", "friday", "when")):
        return "Orders ship within 1 business day and usually arrive in 2-4 days, with tracking."
    if any(w in last for w in ("looking", "browsing", "later", "bye")):
        return f"No problem at all, thanks for stopping by. {END_TAG}"
    return "Happy to help if anything else comes up."


def _pushy_script(history: list[Message], ctx: AgentContext) -> str:
    if not history:
        return f"{ctx.first_name}! Your cart is about to expire - take 20% off if you buy in the next hour!"
    return "Don't miss out - 20% off ends very soon, complete your order now!"


@dataclass
class PromptAgent:
    key: str
    label: str
    model: str
    rules: str
    script: object  # offline reply function
    llm: LLM

    def respond(self, history: list[Message], ctx: AgentContext) -> AgentTurn:
        offline = self.script(history, ctx)
        system = f"{self.rules}\n\n{_context_block(ctx)}\n{OfflineLLM.MARKER} {offline}"
        out = traced(f"agent:{self.key}", self.llm, system=system, messages=_opening_nudge(history), model=self.model)
        return parse_turn(out.text, out.input_tokens, out.output_tokens)


AGENTS = {
    "baseline-haiku": ("Baseline · Haiku", "claude-haiku-5-5", BASELINE_RULES, _baseline_script),
    "baseline-sonnet": ("Baseline · Sonnet", "claude-sonnet-5-5", BASELINE_RULES, _baseline_script),
    "baseline-opus": ("Baseline · Opus", "claude-opus-5-5", BASELINE_RULES, _baseline_script),
    "baseline-fable": ("Baseline · Fable", "claude-fable-5-2", BASELINE_RULES, _baseline_script),
    "baseline-gemini-flash": ("Baseline · Gemini Flash", "gemini-3.8-flash", BASELINE_RULES, _baseline_script),
    "baseline-gemini-argon": ("Baseline · Gemini Argon", "gemini-4-argon", BASELINE_RULES, _baseline_script),
    "pushy-haiku": ("Pushy discounter · Haiku", "claude-haiku-5-5", PUSHY_RULES, _pushy_script),
}


def get_agent(key: str, llm: LLM) -> PromptAgent:
    label, model, rules, script = AGENTS[key]
    return PromptAgent(key=key, label=label, model=model, rules=rules, script=script, llm=llm)
