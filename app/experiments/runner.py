"""Running an agent against a sample of abandoned checkouts.

For each checkout: the agent opens, the simulated shopper replies, and they
go back and forth for up to MAX_ROUNDS (or until the agent hands off). The
outcome model then decides whether the shopper bought.
"""
import random
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents import AgentContext, get_agent
from app.llm import LLM, Message
from app.models import (
    Checkout,
    CheckoutItem,
    CheckoutStatus,
    Conversation,
    ConversationMessage,
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


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def create_experiment(session: Session, store_id: int, agent_key: str, sample_size: int) -> Experiment:
    exp = Experiment(
        store_id=store_id,
        agent_key=agent_key,
        sample_size=sample_size,
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


def _converse(session: Session, exp: Experiment, checkout: Checkout, llm: LLM, store: Store) -> None:
    agent = get_agent(exp.agent_key, llm)
    persona = persona_for(checkout.id)
    ctx = AgentContext(
        store_name=store.name,
        first_name=checkout.customer.first_name,
        cart_lines=[f"{i.quantity} x {i.product.title} ({_money(i.unit_price_cents)})" for i in checkout.items],
        cart_total=_money(checkout.total_cents),
        store_facts=STORE_FACTS,
    )
    shopper = ShopperBot(persona, llm, ctx.first_name, ", ".join(ctx.cart_lines))

    history: list[Message] = []
    transcript: list[tuple[str, str]] = []
    agent_text, handed_off, discount = [], False, 0
    tokens_in = tokens_out = 0

    for _ in range(MAX_ROUNDS):
        turn = agent.respond(history, ctx)
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
    session.add(
        Conversation(
            experiment_id=exp.id,
            checkout_id=checkout.id,
            persona_key=persona.key,
            purchased=outcome.purchased,
            would_buy_anyway=outcome.would_buy_anyway,
            handed_off=handed_off,
            discount_pct=discount,
            revenue_cents=revenue,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            messages=[
                ConversationMessage(seq=i, sender=sender, text=text)
                for i, (sender, text) in enumerate(transcript)
            ],
        )
    )


def run_experiment(session: Session, exp: Experiment, llm: LLM) -> Experiment:
    exp.status = ExperimentStatus.running
    exp.started_at = datetime.now()
    session.commit()
    try:
        store = session.get(Store, exp.store_id)
        for checkout in _sample(session, exp):
            _converse(session, exp, checkout, llm, store)
        exp.status = ExperimentStatus.done
    except Exception as err:  # keep the failure visible on the page
        session.rollback()
        exp.status = ExperimentStatus.failed
        exp.error = str(err)[:500]
    exp.finished_at = datetime.now()
    session.commit()
    return exp


@dataclass
class ExperimentSummary:
    conversations: int
    purchases: int
    claimed_revenue_cents: int     # every purchase after a message
    anyway_revenue_cents: int      # purchases that would have happened anyway
    caused_revenue_cents: int      # the true effect (simulator ground truth)
    handoffs: int
    discounted: int
    tokens: int


def summarize(session: Session, exp: Experiment) -> ExperimentSummary:
    convs = list(session.scalars(select(Conversation).where(Conversation.experiment_id == exp.id)))
    claimed = sum(x.revenue_cents for x in convs)
    anyway = sum(x.revenue_cents for x in convs if x.would_buy_anyway)
    return ExperimentSummary(
        conversations=len(convs),
        purchases=sum(1 for x in convs if x.purchased),
        claimed_revenue_cents=claimed,
        anyway_revenue_cents=anyway,
        caused_revenue_cents=claimed - anyway,
        handoffs=sum(1 for x in convs if x.handed_off),
        discounted=sum(1 for x in convs if x.discount_pct > 0),
        tokens=sum(x.input_tokens + x.output_tokens for x in convs),
    )
