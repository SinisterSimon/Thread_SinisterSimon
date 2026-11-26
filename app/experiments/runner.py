"""Running an agent against a sample of abandoned checkouts.

For each checkout: the agent opens, the simulated shopper replies, and they
go back and forth for up to MAX_ROUNDS (or until the agent hands off). The
outcome model then decides whether the shopper bought.
"""
import os
import random
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import tracing
from app.agents import AgentContext, get_agent
from app.agents import guardrails as guard
from app.agents.base import AgentTurn
from app.evals import score_conversation
from app.experiments.stats import LiftEstimate, estimate_lift
from app.llm import LLM, Message
from app.llm.client import LLMError
from app.llm.pricing import cost_micros
from app.models import (
    Checkout,
    CheckoutItem,
    CheckoutStatus,
    Conversation,
    ConversationMessage,
    ConversationScore,
    Customer,
    Experiment,
    ExperimentStatus,
    Store,
)
from app.shoppers import ShopperBot, persona_for
from app.shoppers.outcome import decide

MAX_ROUNDS = 3
STORE_FACTS = [
    "Orders ship within 1 business day; delivery usually takes 2-4 business days, with tracking.",
    "Free exchanges and returns within 30 days.",
    "Apparel fits true to size.",
    "Agents may offer up to 10% off once, only for price concerns.",
]


SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


def _money(cents: int, currency: str = "USD") -> str:
    symbol = SYMBOLS.get(currency)
    amount = f"{cents / 100:,.2f}"
    return f"{symbol}{amount}" if symbol else f"{amount} {currency}"


def create_experiment(
    session: Session,
    store_id: int,
    agent_key: str,
    sample_size: int,
    guardrails: bool = False,
    holdout_pct: int = 0,
    shadow: bool = False,
) -> Experiment:
    exp = Experiment(
        store_id=store_id,
        agent_key=agent_key,
        sample_size=sample_size,
        guardrails=guardrails,
        holdout_pct=0 if shadow else max(0, min(holdout_pct, 90)),
        shadow=shadow,
        status=ExperimentStatus.queued,
        created_at=datetime.now(),
    )
    session.add(exp)
    session.commit()
    return exp


def _sample(session: Session, exp: Experiment) -> list[Checkout]:
    """Abandoned checkouts we may text (phone + consent), in a fixed order.

    Every experiment on the same store gets the same checkouts, so agents are
    compared on identical shoppers.
    """
    stmt = (
        select(Checkout)
        .join(Customer, Customer.id == Checkout.customer_id)
        .where(
            Checkout.store_id == exp.store_id,
            Checkout.status == CheckoutStatus.abandoned,
            Customer.sms_consent,
        )
        .options(
            selectinload(Checkout.customer),
            selectinload(Checkout.items).selectinload(CheckoutItem.product),
        )
        .order_by(Checkout.id)
        .limit(exp.sample_size)
    )
    return list(session.scalars(stmt))


def _shadow_sample(session: Session, exp: Experiment) -> list[Checkout]:
    """Every real abandoned checkout we could contact by text or email."""
    stmt = (
        select(Checkout)
        .join(Customer, Customer.id == Checkout.customer_id)
        .where(
            Checkout.store_id == exp.store_id,
            Checkout.status == CheckoutStatus.abandoned,
            (Customer.phone.is_not(None)) | (Customer.email.is_not(None)),
        )
        .options(
            selectinload(Checkout.customer),
            selectinload(Checkout.items).selectinload(CheckoutItem.product),
        )
        .order_by(Checkout.started_at.desc())
        .limit(exp.sample_size)
    )
    return list(session.scalars(stmt))


def clean_name(name: str) -> str:
    """"sinister" or "SINISTER" -> "Sinister". Mixed case ("McKay") is kept as typed."""
    name = (name or "").strip()
    if name and name != "there" and (name.islower() or name.isupper()):
        return name.title()
    return name or "there"


def clean_title(title: str) -> str:
    """Product titles read mid-sentence: "The Multi-location Snowboard" -> "Multi-location Snowboard"."""
    title = " ".join((title or "").split())
    return title[4:] if title.lower().startswith("the ") and len(title) > 4 else title


def store_facts(store: Store) -> list[str]:
    """Facts the agent may use. A real store can set its own in SHOPIFY_STORE_FACTS ("fact | fact")."""
    custom = os.getenv("SHOPIFY_STORE_FACTS", "") if store.source == "shopify" else ""
    facts = [f.strip() for f in custom.split("|") if f.strip()]
    return facts or STORE_FACTS


def _context(store: Store, checkout: Checkout) -> AgentContext:
    return AgentContext(
        store_name=store.name,
        first_name=clean_name(checkout.customer.first_name),
        cart_lines=[
            f"{i.quantity} x {clean_title(i.product.title)} ({_money(i.unit_price_cents, store.currency)})"
            for i in checkout.items
        ],
        cart_total=_money(checkout.total_cents, store.currency),
        store_facts=store_facts(store),
        checkout_url=checkout.recovery_url,
    )


def _draft(session: Session, exp: Experiment, checkout: Checkout, llm: LLM, store: Store) -> None:
    """Shadow mode: write the opening message for a real shopper, and keep it. Send nothing.

    There is no reply and no outcome, so no lift: shadow mode answers "what would
    this agent say to our actual customers?" before anyone trusts it with a send button.
    """
    meta = {"experiment_id": exp.id, "agent": exp.agent_key, "checkout_id": checkout.id, "shadow": True}
    with tracing.conversation_trace(f"draft:{exp.agent_key}", meta):
        agent = get_agent(exp.agent_key, llm)
        agent_model = agent.model
        try:
            turn = agent.respond([], _context(store, checkout))
        except LLMError as err:
            # One bad reply shouldn't sink the other drafts; show why instead.
            turn = AgentTurn(text=f"(No draft: {err}.)", handoff=False, end=False, discount_pct=0,
                             input_tokens=0, output_tokens=0)
        fixes = 0
        if exp.guardrails and not turn.text.startswith("(No draft"):
            turn, fixes = guard.apply(turn, last_shopper_text=None, discount_already_offered=False)
    text = turn.text
    if checkout.recovery_url and not text.startswith("(No draft"):
        # The link is added by Thread, never written by the model, so it can't be invented or mistyped.
        text = f"{text} {checkout.recovery_url}"
    session.add(
        Conversation(
            experiment_id=exp.id,
            checkout_id=checkout.id,
            persona_key="real",
            arm="shadow",
            handed_off=turn.handoff,
            discount_pct=turn.discount_pct,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            guardrail_fixes=fixes,
            agent_cost_micros=cost_micros(agent_model, turn.input_tokens, turn.output_tokens),
            messages=[ConversationMessage(seq=0, sender="agent", text=text)],
        )
    )


def in_holdout(checkout_id: int, holdout_pct: int) -> bool:
    """Random but stable: the same checkout always lands in the same arm for a
    given holdout share, so different agents are compared on the same split."""
    return random.Random(f"arm:{checkout_id}").random() * 100 < holdout_pct


def _holdout(session: Session, exp: Experiment, checkout: Checkout) -> None:
    """No message is sent. The shopper buys only if they were coming back anyway, at full price."""
    anyway = checkout.sim_returns_naturally
    session.add(
        Conversation(
            experiment_id=exp.id,
            checkout_id=checkout.id,
            persona_key=persona_for(checkout.id).key,
            arm="holdout",
            purchased=anyway,
            would_buy_anyway=anyway,
            revenue_cents=checkout.total_cents if anyway else 0,
            baseline_revenue_cents=checkout.total_cents if anyway else 0,
        )
    )


def _converse(session: Session, exp: Experiment, checkout: Checkout, llm: LLM, store: Store) -> None:
    if in_holdout(checkout.id, exp.holdout_pct):
        _holdout(session, exp, checkout)
        return
    meta = {"experiment_id": exp.id, "agent": exp.agent_key, "checkout_id": checkout.id, "guardrails": exp.guardrails}
    with tracing.conversation_trace(f"conversation:{exp.agent_key}", meta):
        _converse_inner(session, exp, checkout, llm, store)


def _converse_inner(session: Session, exp: Experiment, checkout: Checkout, llm: LLM, store: Store) -> None:
    agent = get_agent(exp.agent_key, llm)
    persona = persona_for(checkout.id)
    ctx = _context(store, checkout)
    shopper = ShopperBot(persona, llm, ctx.first_name, ", ".join(ctx.cart_lines))

    history: list[Message] = []
    transcript: list[tuple[str, str]] = []
    agent_text, handed_off, discount = [], False, 0
    tokens_in = tokens_out = fixes = agent_cost = 0

    for _ in range(MAX_ROUNDS):
        turn = agent.respond(history, ctx)
        agent_cost += cost_micros(agent.model, turn.input_tokens, turn.output_tokens)
        if exp.guardrails:
            last_shopper = history[-1].text if history and history[-1].role == "user" else None
            turn, n = guard.apply(turn, last_shopper_text=last_shopper, discount_already_offered=discount > 0)
            fixes += n
        history.append(Message(role="assistant", text=turn.text))
        transcript.append(("agent", turn.text))
        agent_text.append(turn.text)
        discount = max(discount, turn.discount_pct)
        tokens_in += turn.input_tokens
        tokens_out += turn.output_tokens
        if turn.handoff:
            handed_off = True
            break
        if turn.end:
            break
        reply, t_in, t_out = shopper.reply(history)
        history.append(Message(role="user", text=reply))
        transcript.append(("shopper", reply))
        tokens_in += t_in
        tokens_out += t_out

    outcome = decide(
        persona=persona,
        agent_text=" ".join(agent_text),
        handed_off=handed_off,
        discount_pct=discount,
        persuadability=checkout.sim_persuadability,
        returns_naturally=checkout.sim_returns_naturally,
        # Same seed per checkout for every agent ("common random numbers"): two agents
        # that behave identically get identical outcomes, so differences are real.
        rng=random.Random(f"outcome:{checkout.id}"),
    )
    revenue = round(checkout.total_cents * (100 - discount) / 100) if outcome.purchased else 0
    score, j_in, j_out = score_conversation(
        llm=llm, persona=persona, transcript=transcript, handed_off=handed_off, discount_pct=discount
    )
    tracing.score_trace("quality", score.quality, score.judge_notes)
    tracing.score_trace("caused_sale", float(outcome.purchased and not outcome.would_buy_anyway))
    session.add(
        Conversation(
            experiment_id=exp.id,
            checkout_id=checkout.id,
            persona_key=persona.key,
            arm="treatment",
            purchased=outcome.purchased,
            would_buy_anyway=outcome.would_buy_anyway,
            handed_off=handed_off,
            discount_pct=discount,
            revenue_cents=revenue,
            baseline_revenue_cents=checkout.total_cents if outcome.would_buy_anyway else 0,
            input_tokens=tokens_in + j_in,
            output_tokens=tokens_out + j_out,
            guardrail_fixes=fixes,
            agent_cost_micros=agent_cost,
            score=ConversationScore(
                addressed_objection=score.addressed_objection,
                correct_handoff=score.correct_handoff,
                discount_within_policy=score.discount_within_policy,
                no_pressure=score.no_pressure,
                respected_browser=score.respected_browser,
                judge_model=score.judge_model,
                helpfulness=score.helpfulness,
                accuracy=score.accuracy,
                tone=score.tone,
                judge_notes=score.judge_notes,
                quality=score.quality,
            ),
            messages=[
                ConversationMessage(seq=i, sender=sender, text=text)
                for i, (sender, text) in enumerate(transcript)
            ],
        )
    )


# A run survives a few provider failures, but not many: dropped conversations are not
# random (long or tricky ones fail more), so too many would bias the lift estimate.
MAX_SKIP_SHARE = 0.05


def _too_many_skipped(skipped: int, attempted: int) -> bool:
    return skipped > max(1, int(attempted * MAX_SKIP_SHARE))


def run_experiment(session: Session, exp: Experiment, llm: LLM) -> Experiment:
    exp.status = ExperimentStatus.running
    exp.started_at = datetime.now()
    session.commit()
    try:
        store = session.get(Store, exp.store_id)
        if exp.shadow:
            checkouts, step = _shadow_sample(session, exp), _draft
        else:
            checkouts, step = _sample(session, exp), _converse
        skipped, attempted, last_error = 0, 0, ""
        for checkout in checkouts:
            messaged = exp.shadow or not in_holdout(checkout.id, exp.holdout_pct)
            attempted += messaged
            try:
                step(session, exp, checkout, llm, store)
            except LLMError as err:
                # The provider already retried. Drop this one conversation, keep the run.
                # Nothing was added to the session yet: rows are only added at the end of a step.
                skipped += 1
                last_error = str(err)
                if _too_many_skipped(skipped, attempted):
                    raise LLMError(f"Stopped after {skipped} failed conversations. Last error: {last_error}") from err
        exp.skipped = skipped
        if skipped:
            exp.error = f"Skipped {skipped} conversation(s) after provider errors. Last: {last_error}"[:500]
        exp.status = ExperimentStatus.done
        tracing.flush()
    except Exception as err:  # keep the failure visible on the page
        session.rollback()
        exp.status = ExperimentStatus.failed
        exp.error = str(err)[:500]
    exp.finished_at = datetime.now()
    session.commit()
    return exp


@dataclass
class ExperimentSummary:
    conversations: int             # messaged shoppers (treatment arm)
    purchases: int
    claimed_revenue_cents: int     # every purchase after a message
    anyway_revenue_cents: int      # what those shoppers would have paid with no message
    caused_revenue_cents: int      # true lift = claimed - anyway (simulator ground truth)
    handoffs: int
    discounted: int
    tokens: int
    avg_quality: float | None        # 0-100, rules + AI judge
    policy_violations: int           # chats that failed a rule check
    guardrail_fixes: int             # agent messages the guardrails changed
    holdout_size: int = 0
    estimate: LiftEstimate | None = None  # what a real store would measure
    agent_cost_micros: int = 0            # agent model calls at list price

    @property
    def agent_cost_usd(self) -> float:
        return self.agent_cost_micros / 1_000_000

    @property
    def lift_per_dollar(self) -> float | None:
        """True lift (in dollars) for every $1 spent on the agent's model calls."""
        if self.agent_cost_micros <= 0:
            return None
        return (self.caused_revenue_cents / 100) / self.agent_cost_usd

    @property
    def claimed_vs_true(self) -> float | None:
        """How many times bigger the claimed number is than the real effect."""
        if self.caused_revenue_cents <= 0:
            return None
        return self.claimed_revenue_cents / self.caused_revenue_cents


def done_cache(session: Session, name: str) -> dict:
    """A per-database cache for facts about finished experiments, which never change again.

    Kept on the engine object, so each database (and each test's fresh database) has its own.
    """
    bind = session.get_bind()
    caches = bind.__dict__.setdefault("_thread_done_caches", {})
    return caches.setdefault(name, {})


def summarize(session: Session, exp: Experiment) -> ExperimentSummary:
    """Totals and the lift estimate for one experiment. A finished experiment is summarized
    once and then served from memory: loading every conversation is the slow part."""
    done = exp.status == ExperimentStatus.done
    cache = done_cache(session, "summaries")
    if done and exp.id in cache:
        return cache[exp.id]
    summary = _summarize(session, exp)
    if done:
        cache[exp.id] = summary
    return summary


def _summarize(session: Session, exp: Experiment) -> ExperimentSummary:
    # Plain rows of just the columns needed, not full Conversation objects: much faster for
    # experiments with thousands of conversations.
    rows = session.execute(
        select(
            Conversation.arm, Conversation.revenue_cents, Conversation.baseline_revenue_cents,
            Conversation.purchased, Conversation.handed_off, Conversation.discount_pct,
            Conversation.input_tokens, Conversation.output_tokens, Conversation.guardrail_fixes,
            Conversation.agent_cost_micros, ConversationScore.quality, ConversationScore.correct_handoff,
            ConversationScore.discount_within_policy, ConversationScore.no_pressure,
        )
        .outerjoin(ConversationScore, ConversationScore.conversation_id == Conversation.id)
        .where(Conversation.experiment_id == exp.id)
    ).all()
    treated = [r for r in rows if r.arm == "treatment"]
    held = [r for r in rows if r.arm == "holdout"]
    scores = [r for r in treated if r.quality is not None]
    claimed = sum(x.revenue_cents for x in treated)
    anyway = sum(x.baseline_revenue_cents for x in treated)
    estimate = estimate_lift(
        [c.revenue_cents for c in treated],
        [c.revenue_cents for c in held],
        [c.purchased for c in treated],
        [c.purchased for c in held],
    )
    return ExperimentSummary(
        conversations=len(treated),
        purchases=sum(1 for x in treated if x.purchased),
        claimed_revenue_cents=claimed,
        anyway_revenue_cents=anyway,
        caused_revenue_cents=claimed - anyway,
        handoffs=sum(1 for x in treated if x.handed_off),
        discounted=sum(1 for x in treated if x.discount_pct > 0),
        tokens=sum(x.input_tokens + x.output_tokens for x in treated),
        avg_quality=round(sum(s.quality for s in scores) / len(scores), 1) if scores else None,
        policy_violations=sum(
            1 for s in scores if not (s.correct_handoff and s.discount_within_policy and s.no_pressure)
        ),
        guardrail_fixes=sum(x.guardrail_fixes for x in treated),
        holdout_size=len(held),
        estimate=estimate,
        agent_cost_micros=sum(x.agent_cost_micros for x in treated),
    )
