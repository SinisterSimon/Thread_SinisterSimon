"""Phase 4: holdout groups and lift estimation. All offline."""
import math

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.experiments import create_experiment, run_experiment, summarize
from app.experiments.runner import in_holdout
from app.experiments.stats import estimate_lift
from app.llm.client import OfflineLLM
from app.models import Conversation
from app.simulator import SimConfig, run_simulation


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


# --- statistics ---------------------------------------------------------------


def test_estimate_matches_hand_calculation():
    est = estimate_lift([10, 20, 30, 40], [0, 10, 20, 10], [True] * 4, [False, True, True, True])
    # means 25 and 10 -> 15 per shopper, x4 messaged shoppers = 60
    assert est.lift_per_shopper == 15 and est.total_lift == 60
    se = math.sqrt((500 / 3) / 4 + (200 / 3) / 4)
    assert math.isclose(est.ci_low, (15 - 1.96 * se) * 4)
    assert est.conversion_holdout == 0.75


def test_needs_two_shoppers_per_group():
    assert estimate_lift([5, 6], [1], [True, True], [True]) is None


def test_interval_including_zero_is_not_significant():
    est = estimate_lift([0, 100, 0, 100], [100, 0, 0, 100], [False, True, False, True], [True, False, False, True])
    assert est.total_lift == 0 and not est.significant


# --- assignment ---------------------------------------------------------------


def test_holdout_assignment_is_stable_and_close_to_the_share():
    ids = range(1, 4001)
    share = sum(in_holdout(i, 20) for i in ids) / len(ids)
    assert 0.17 < share < 0.23
    assert [in_holdout(i, 20) for i in ids] == [in_holdout(i, 20) for i in ids]
    assert not any(in_holdout(i, 0) for i in ids)


def test_holdout_shoppers_get_no_messages_and_pay_full_price():
    with _session() as s:
        sim = run_simulation(s, SimConfig(seed=3, days=30))
        exp = create_experiment(s, sim.store_id, "pushy-haiku", 500, holdout_pct=50)
        run_experiment(s, exp, OfflineLLM())
        held = list(s.scalars(select(Conversation).where(Conversation.arm == "holdout")))
        assert held
        assert all(not c.messages and c.score is None and c.discount_pct == 0 for c in held)
        assert all(c.purchased == c.would_buy_anyway for c in held)
        assert all(c.revenue_cents == c.baseline_revenue_cents for c in held)


# --- the point of Phase 4 -----------------------------------------------------


def test_holdout_estimate_recovers_the_true_lift():
    """On a big store, the 95% range should contain the truth for every agent."""
    with _session() as s:
        sim = run_simulation(s, SimConfig(seed=42, days=180))
        for key in ("baseline-haiku", "pushy-haiku"):
            exp = create_experiment(s, sim.store_id, key, 5000, holdout_pct=50)
            run_experiment(s, exp, OfflineLLM())
            r = summarize(s, exp)
            assert r.estimate.ci_low <= r.caused_revenue_cents <= r.estimate.ci_high, key
            assert r.estimate.significant
            assert r.claimed_vs_true > 1.5  # claiming every sale overstates the effect


def test_true_lift_charges_discounts_given_to_shoppers_who_would_buy_anyway():
    with _session() as s:
        sim = run_simulation(s, SimConfig(seed=21, days=30))
        exp = create_experiment(s, sim.store_id, "pushy-haiku", 150)
        run_experiment(s, exp, OfflineLLM())
        anyway = [c for c in s.scalars(select(Conversation)) if c.would_buy_anyway]
        # The pushy agent gave them 20% off: they paid less than they would have without a message.
        assert all(c.revenue_cents < c.baseline_revenue_cents for c in anyway)
        r = summarize(s, exp)
        assert r.caused_revenue_cents == r.claimed_revenue_cents - sum(c.baseline_revenue_cents for c in anyway)
