"""Comparing one prompt across models: list-price costs and lift per dollar."""
import os

os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db  # noqa: E402
from app.experiments.runner import ExperimentSummary  # noqa: E402
from app.llm.pricing import PRICES_PER_MTOK, cost_micros, has_price  # noqa: E402
from app.models import Conversation, Experiment  # noqa: E402


def test_cost_is_exact_micro_dollars_at_list_price():
    # 1M input + 1M output tokens on Opus = $4 + $20 = $24
    assert cost_micros("claude-opus-5-5", 1_000_000, 1_000_000) == 24_000_000
    assert cost_micros("offline:claude-haiku-5-5", 1000, 100) == 150  # $0.00015
    assert cost_micros("some-unknown-model", 1000, 1000) == 0 and not has_price("some-unknown-model")
    # Every agent in the registry has a price, so no agent silently shows $0.
    from app.agents.registry import AGENTS

    assert all(has_price(model) for _, model, _, _ in AGENTS.values())
    assert set(PRICES_PER_MTOK) >= {model for _, model, _, _ in AGENTS.values()}


def test_lift_per_dollar():
    s = ExperimentSummary(conversations=1, purchases=1, claimed_revenue_cents=5000, anyway_revenue_cents=1000,
                          caused_revenue_cents=4000, handoffs=0, discounted=0, tokens=0, avg_quality=None,
                          policy_violations=0, guardrail_fixes=0, agent_cost_micros=2_000_000)
    assert s.agent_cost_usd == 2.0 and s.lift_per_dollar == 20.0  # $40 true lift / $2
    s.agent_cost_micros = 0
    assert s.lift_per_dollar is None


def test_compare_models_runs_one_prompt_on_six_models_on_the_same_shoppers():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    with TestClient(main.app) as client:
        client.post("/simulate", data={"days": 20})
        page = client.post("/experiments/compare-models", follow_redirects=True).text
        assert "Compare 6 models" in page and "Model cost" in page and "Lift / $1" in page

        with db.SessionLocal() as session:
            exps = session.scalars(select(Experiment).order_by(Experiment.id)).all()
            assert [e.agent_key for e in exps] == list(main.MODEL_COMPARISON)
            assert all(e.status.value == "done" and e.holdout_pct == 50 for e in exps)
            arms = [
                sorted((c.checkout_id, c.arm) for c in session.scalars(
                    select(Conversation).where(Conversation.experiment_id == e.id)))
                for e in exps
            ]
            assert all(a == arms[0] for a in arms)  # identical shoppers and identical split

            cost = {e.agent_key: main.summarize(session, e).agent_cost_micros for e in exps}
            # Offline scripts are identical, so the only difference is the price per token.
            assert 0 < cost["baseline-haiku"] < cost["baseline-gemini-flash"] < cost["baseline-sonnet"]
            assert cost["baseline-sonnet"] < cost["baseline-opus"] < cost["baseline-fable"]
            assert cost["baseline-gemini-argon"] == cost["baseline-opus"]  # same list price from Dec 1
            held = session.scalars(select(Conversation).where(Conversation.arm == "holdout")).all()
            assert held and all(c.agent_cost_micros == 0 for c in held)  # no message, no cost
