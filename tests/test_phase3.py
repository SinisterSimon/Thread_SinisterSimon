"""Phase 3: migrations, guardrails, scoring. All offline."""
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app import migrate
from app.agents.base import AgentTurn
from app.agents.guardrails import HANDOFF_LINE, apply
from app.db import Base
from app.evals import score_conversation
from app.experiments import create_experiment, run_experiment, summarize
from app.llm.client import OfflineLLM
from app.shoppers import PERSONAS
from app.simulator import SimConfig, run_simulation

BY_KEY = {p.key: p for p in PERSONAS}


def turn(text, handoff=False, discount=0):
    return AgentTurn(text=text, handoff=handoff, end=False, discount_pct=discount, input_tokens=1, output_tokens=1)


# --- migrations ---------------------------------------------------------------


def test_migrations_match_the_models(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    migrate.upgrade(engine)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations disagree: {diff}"


def test_pre_alembic_database_is_stamped_and_upgraded(tmp_path):
    """The live Neon database was made with create_all before Alembic existed."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    cfg = migrate._config(engine)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        from alembic import command

        command.upgrade(cfg, migrate.BASELINE)  # Phase 1-2 tables only
        conn.execute(text("DROP TABLE alembic_version"))  # as if made by create_all
        conn.execute(text("INSERT INTO stores (id, name, currency, created_at) VALUES (1, 'Old', 'USD', '2026-11-01')"))
        conn.execute(text(
            "INSERT INTO experiments (id, store_id, agent_key, sample_size, status, created_at) "
            "VALUES (1, 1, 'baseline-haiku', 10, 'done', '2026-11-01')"
        ))

    migrate.upgrade(engine)

    insp = inspect(engine)
    assert "conversation_scores" in insp.get_table_names()
    assert "guardrails" in {c["name"] for c in insp.get_columns("experiments")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT guardrails FROM experiments WHERE id = 1")).scalar() in (0, False)
    migrate.upgrade(engine)  # running again is a no-op


# --- guardrails ---------------------------------------------------------------


def test_guardrail_caps_discount():
    fixed, n = apply(turn("Take 20% off today.", discount=20), last_shopper_text=None, discount_already_offered=False)
    assert fixed.discount_pct == 10 and "10% off" in fixed.text and n == 1


def test_guardrail_drops_repeat_offers_and_pressure():
    fixed, n = apply(
        turn("Don't miss out - 20% off ends very soon, complete your order now!", discount=20),
        last_shopper_text="Maybe later.",
        discount_already_offered=True,
    )
    assert fixed.discount_pct == 0 and n >= 1 and "miss" not in fixed.text.lower()


def test_guardrail_forces_handoff_when_asked_for_a_person():
    fixed, n = apply(turn("Here's 10% off!", discount=10), last_shopper_text="I want a real person.",
                     discount_already_offered=False)
    assert fixed.handoff and fixed.text == HANDOFF_LINE and n == 1


def test_guardrail_leaves_good_messages_alone():
    good = turn("It fits true to size, and exchanges are free within 30 days.")
    fixed, n = apply(good, last_shopper_text="Does it run small?", discount_already_offered=False)
    assert fixed == good and n == 0


# --- scoring ------------------------------------------------------------------


def test_score_flags_policy_breaks():
    transcript = [("agent", "Take 20% off, hurry, it expires in the next hour!"), ("shopper", "I was just looking.")]
    score, _, _ = score_conversation(llm=OfflineLLM(), persona=BY_KEY["browsing"], transcript=transcript,
                                     handed_off=False, discount_pct=20)
    assert not score.discount_within_policy and not score.no_pressure and not score.respected_browser
    assert score.quality < 50


def test_score_rewards_a_good_handoff():
    transcript = [("agent", "Hi, any questions?"), ("shopper", "You charged me twice, I want a person."),
                  ("agent", "Sorry about that. A colleague from our team will reach out today.")]
    score, _, _ = score_conversation(llm=OfflineLLM(), persona=BY_KEY["angry"], transcript=transcript,
                                     handed_off=True, discount_pct=0)
    assert score.rules_passed == 5 and score.quality == 100


def test_judge_reply_that_is_not_json_falls_back():
    class Garbled(OfflineLLM):
        def complete(self, *, system, messages, model):
            reply = super().complete(system=system, messages=messages, model=model)
            return reply.__class__("I think it was fine!", reply.model, 1, 1)

    score, _, _ = score_conversation(llm=Garbled(), persona=BY_KEY["sizing"],
                                     transcript=[("agent", "It fits true to size.")], handed_off=False, discount_pct=0)
    assert "could not be read" in score.judge_notes


# --- end to end ---------------------------------------------------------------


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_quality_and_guardrails_end_to_end():
    with _session() as s:
        sim = run_simulation(s, SimConfig(seed=21, days=30))
        out = {}
        for key, guarded in (("baseline-haiku", False), ("pushy-haiku", False), ("pushy-haiku", True)):
            e = create_experiment(s, sim.store_id, key, 150, guardrails=guarded)
            run_experiment(s, e, OfflineLLM())
            out[(key, guarded)] = summarize(s, e)

    base, pushy, pushy_guarded = out[("baseline-haiku", False)], out[("pushy-haiku", False)], out[("pushy-haiku", True)]
    assert base.avg_quality > pushy.avg_quality
    assert base.guardrail_fixes == 0 and pushy.guardrail_fixes == 0  # guardrails off
    assert pushy_guarded.guardrail_fixes > 0
    assert pushy_guarded.policy_violations < pushy.policy_violations
    assert pushy_guarded.avg_quality > pushy.avg_quality
