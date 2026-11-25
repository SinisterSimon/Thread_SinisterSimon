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


def test_compare_models_runs_one_prompt_on_every_model_on_the_same_shoppers():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    with TestClient(main.app) as client:
        client.post("/simulate", data={"days": 20})
        page = client.post("/experiments/compare-models", follow_redirects=True).text
        assert "Compare 10 models" in page and "Lift / $1" in page and "Price /1M in·out" in page

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
            # One GPT model per Claude price tier.
            assert cost["baseline-gpt-luna"] == cost["baseline-haiku"]
            assert cost["baseline-gpt-sol"] == cost["baseline-sonnet"]
            assert cost["baseline-gpt-astra"] == cost["baseline-fable"]
            assert cost["baseline-gemini-flash"] < cost["baseline-gpt-terra"] < cost["baseline-sonnet"]
            held = session.scalars(select(Conversation).where(Conversation.arm == "holdout")).all()
            assert held and all(c.agent_cost_micros == 0 for c in held)  # no message, no cost


def test_model_table_shows_prices_and_never_scores_a_model_from_offline_runs():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main
    from app.models import ConversationScore

    def model_table(page: str) -> str:
        return page.split('id="model-comparison"', 1)[1].split("<h2>Leaderboard</h2>", 1)[0]

    with TestClient(main.app) as client:
        client.post("/simulate", data={"days": 20})
        table = model_table(client.post("/experiments/compare-models", follow_redirects=True).text)
        # Cheapest first, list prices per 1M tokens (input / output).
        names = ["Haiku", "GPT Luna", "Gemini Flash", "GPT Terra", "Sonnet", "GPT Sol", "Opus", "Gemini Argon",
                 "Fable", "GPT Astra"]
        where = [table.index(f'<strong style="font-weight:500">{n}</strong>') for n in names]
        assert where == sorted(where)
        for price in ("$0.10 / $0.50", "$0.75 / $3.75", "$1 / $5", "$2 / $10", "$4 / $20", "$8 / $40"):
            assert f'<td class="num muted">{price}</td>' in table
        # The strategy rows show the lab's own runs, marked offline. Offline runs never score a
        # model: all ten model rows stay "Not measured yet" (plus pushy, which has not run).
        assert "· offline" in table
        assert table.count("Not measured yet") == 11

        # A run whose judge used a real model counts as that model's measurement.
        with db.SessionLocal() as session:
            sonnet = session.scalars(select(Experiment).where(Experiment.agent_key == "baseline-sonnet")).one()
            for score in session.scalars(select(ConversationScore).join(Conversation).where(
                    Conversation.experiment_id == sonnet.id)):
                score.judge_model = "claude-sonnet-5-5"
            session.commit()
            sonnet_id = sonnet.id
            # Finished runs are cached as offline/real; this test rewrites history, so forget it.
            from app.experiments.runner import done_cache
            done_cache(session, "real_models").clear()
        table = model_table(client.get("/experiments").text)
        assert table.count("Not measured yet") == 10
        assert f'href="/experiments/{sonnet_id}" style="text-decoration:none">run #{sonnet_id}</a>' in table
        assert "same true lift" in table and "Policy breaks" in table


def test_attribution_lens_shows_real_claimed_and_true_lift_for_the_three_agents():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    with TestClient(main.app) as client:
        assert "No runs yet" in client.get("/").text
        client.post("/simulate", data={"days": 20})
        client.post("/experiments/run-all", data={"holdout_pct": 50})
        page = client.get("/").text
        assert "Attribution lens" in page and 'type="range"' in page
        money = main.templates.env.filters["money"]
        with db.SessionLocal() as session:
            exps = session.scalars(select(Experiment).order_by(Experiment.id)).all()
            for name, exp in zip(["Baseline", "Pushy", "Pushy + guardrails"], exps, strict=True):
                s = main.summarize(session, exp)
                assert f'aria-label="{name} claimed revenue">{money(s.claimed_revenue_cents)}' in page
                assert f'aria-label="{name} true lift">{money(s.caused_revenue_cents)}' in page
