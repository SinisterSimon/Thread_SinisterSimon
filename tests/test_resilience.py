"""A busy provider drops a conversation, not the whole run (up to a limit)."""
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db import Base
from app.experiments import create_experiment, run_experiment
from app.experiments.runner import _too_many_skipped
from app.llm.client import LLMError, OfflineLLM
from app.models import Conversation, ExperimentStatus
from app.simulator import SimConfig, run_simulation


class FlakyLLM(OfflineLLM):
    def __init__(self, fail_calls):
        self.calls, self.fail_calls = 0, fail_calls

    def complete(self, **kw):
        self.calls += 1
        if self.fail_calls == "all" or self.calls in self.fail_calls:
            raise LLMError("Provider busy (503).")
        return super().complete(**kw)


def _run(llm, n=100):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    sim = run_simulation(s, SimConfig(seed=3))
    exp = create_experiment(s, sim.store_id, "baseline-haiku", n, holdout_pct=50)
    run_experiment(s, exp, llm)
    rows = s.scalar(select(func.count()).select_from(Conversation).where(Conversation.experiment_id == exp.id))
    return exp, rows


def test_one_failed_conversation_is_skipped_and_the_run_finishes():
    clean, clean_rows = _run(OfflineLLM())
    exp, rows = _run(FlakyLLM({3}))
    assert exp.status == ExperimentStatus.done
    assert exp.skipped == 1
    assert rows == clean_rows - 1
    assert "Skipped 1" in exp.error


def test_a_provider_that_keeps_failing_stops_the_run():
    exp, _ = _run(FlakyLLM("all"))
    assert exp.status == ExperimentStatus.failed
    assert "Stopped after" in exp.error


def test_skip_limit_is_five_percent_with_a_floor_of_one():
    assert not _too_many_skipped(1, 10)
    assert _too_many_skipped(2, 10)
    assert not _too_many_skipped(2, 50)
    assert _too_many_skipped(3, 50)
