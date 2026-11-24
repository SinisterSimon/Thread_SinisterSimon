"""Phase 2: agents, shoppers, experiments. All offline, no API key needed."""
import random

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import db
from app.agents import AgentContext, get_agent
from app.agents.base import parse_turn
from app.experiments import create_experiment, run_experiment, summarize
from app.llm import Message
from app.llm.client import OfflineLLM
from app.models import ExperimentStatus
from app.shoppers import PERSONAS, persona_for
from app.shoppers.outcome import decide
from app.simulator import SimConfig, run_simulation

CTX = AgentContext(store_name="Demo", first_name="Liis", cart_lines=["1 x Tee"], cart_total="$32.00")
BY_KEY = {p.key: p for p in PERSONAS}


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_parse_turn_reads_handoff_and_discount():
    t = parse_turn("Here's 10% off. [HANDOFF]", 5, 3)
    assert t.handoff and t.discount_pct == 10 and "[HANDOFF]" not in t.text


def test_baseline_hands_off_an_upset_shopper():
    agent = get_agent("baseline-haiku", OfflineLLM())
    turn = agent.respond([Message("user", "I want to talk to a real person.")], CTX)
    assert turn.handoff


def test_baseline_never_exceeds_ten_percent():
    agent = get_agent("baseline-haiku", OfflineLLM())
    turn = agent.respond([Message("user", "It's a bit pricey, any deal?")], CTX)
    assert 0 < turn.discount_pct <= 10


def test_persona_assignment_is_stable():
    assert persona_for(123) == persona_for(123)


def test_browsers_never_buy_and_returners_always_do():
    rng = random.Random(0)
    browse = decide(persona=BY_KEY["browsing"], agent_text="10% off", handed_off=False,
                    discount_pct=10, persuadability=100, returns_naturally=False, rng=rng)
    anyway = decide(persona=BY_KEY["browsing"], agent_text="", handed_off=False,
                    discount_pct=0, persuadability=0, returns_naturally=True, rng=rng)
    assert not browse.purchased
    assert anyway.purchased and anyway.would_buy_anyway


def test_upset_shopper_is_only_resolved_by_a_handoff():
    args = dict(persona=BY_KEY["angry"], agent_text="our team will help", discount_pct=0,
                persuadability=50, returns_naturally=False, rng=random.Random(0))
    assert not decide(handed_off=False, **args).resolved
    assert decide(handed_off=True, **args).resolved


def test_experiment_runs_end_to_end(session):
    sim = run_simulation(session, SimConfig(seed=11, days=10))
    exp = create_experiment(session, sim.store_id, "baseline-haiku", 40)
    run_experiment(session, exp, OfflineLLM())
    assert exp.status == ExperimentStatus.done
    s = summarize(session, exp)
    assert s.conversations == min(40, sim.reachable_abandoned)
    assert s.caused_revenue_cents == s.claimed_revenue_cents - s.anyway_revenue_cents
    assert s.tokens > 0


def test_same_behaviour_gives_same_outcomes(session):
    sim = run_simulation(session, SimConfig(seed=5, days=10))
    a = create_experiment(session, sim.store_id, "baseline-haiku", 60)
    b = create_experiment(session, sim.store_id, "baseline-sonnet", 60)
    for e in (a, b):
        run_experiment(session, e, OfflineLLM())
    assert summarize(session, a).caused_revenue_cents == summarize(session, b).caused_revenue_cents


def test_browsers_get_one_goodbye(session):
    agent = get_agent("baseline-haiku", OfflineLLM())
    assert agent.respond([Message("user", "Oh I was just looking around.")], CTX).end


def test_baseline_beats_pushy_on_true_lift(session):
    sim = run_simulation(session, SimConfig(seed=21, days=30))
    results = {}
    for key in ("baseline-haiku", "pushy-haiku"):
        exp = create_experiment(session, sim.store_id, key, 200)
        run_experiment(session, exp, OfflineLLM())
        results[key] = summarize(session, exp)
    # Same shoppers, same "anyway" buyers; the better strategy causes more sales.
    assert results["baseline-haiku"].anyway_revenue_cents > 0
    assert results["baseline-haiku"].caused_revenue_cents > results["pushy-haiku"].caused_revenue_cents
    assert results["baseline-haiku"].handoffs > 0 == results["pushy-haiku"].handoffs
