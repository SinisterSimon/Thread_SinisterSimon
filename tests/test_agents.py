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


class _FakeAnthropic:
    """Stands in for the Anthropic SDK client, so no network or key is needed."""

    def __init__(self, text: str, stop_reason: str = "end_turn"):
        from types import SimpleNamespace as NS

        self.calls = []
        self._resp = NS(
            content=[NS(type="thinking", thinking="..."), NS(type="text", text=text)] if text else [],
            stop_reason=stop_reason,
            usage=NS(input_tokens=10, output_tokens=5),
        )
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def _real_llm(fake):
    import pytest

    pytest.importorskip("anthropic")
    from app.llm.client import AnthropicLLM

    llm = AnthropicLLM(api_key="test")
    llm._client = fake
    return llm


def test_real_model_runs_without_thinking_and_keeps_only_text():
    from app.llm import Message

    fake = _FakeAnthropic("Hi Winnie, your snowboard is still in your cart.")
    reply = _real_llm(fake).complete(system="rules <<OFFLINE_REPLY>> script", messages=[Message("user", "go")],
                                     model="claude-haiku-5-5")
    assert reply.text == "Hi Winnie, your snowboard is still in your cart."
    sent = fake.calls[0]
    assert sent["thinking"] == {"type": "disabled"} and sent["max_tokens"] >= 1024
    assert "OFFLINE_REPLY" not in sent["system"]


def test_real_model_never_returns_an_empty_or_cut_off_message():
    import pytest

    from app.llm import Message
    from app.llm.client import LLMError

    for text, stop, why in [
        ("Orders ship within 1", "max_tokens", "token limit"),
        ("", "refusal", "declined"),
        ("", "end_turn", "no text"),
    ]:
        with pytest.raises(LLMError, match=why):
            _real_llm(_FakeAnthropic(text, stop)).complete(system="s", messages=[Message("user", "go")], model="m")


def test_opus_and_fable_use_adaptive_thinking_at_low_effort():
    from app.llm import Message

    fake = _FakeAnthropic("Hi Winnie, your snowboard is still in your cart.")
    llm = _real_llm(fake)
    for model in ("claude-opus-5-5", "claude-fable-5-2"):
        llm.complete(system="s", messages=[Message("user", "go")], model=model)
        sent = fake.calls[-1]
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["extra_body"] == {"output_config": {"effort": "low"}} and sent["max_tokens"] >= 4096
    llm.complete(system="s", messages=[Message("user", "go")], model="claude-haiku-5-5")
    assert fake.calls[-1]["thinking"] == {"type": "disabled"}


def test_a_model_that_rejects_disabled_thinking_is_retried_and_remembered():
    from app.llm import Message

    fake = _FakeAnthropic("Hello there, your cart is saved.")
    ok = fake._create

    def picky(**kwargs):
        fake.calls.append(kwargs)
        if kwargs["thinking"] == {"type": "disabled"}:
            raise RuntimeError('Error code: 400 - "thinking.type.disabled" is not supported for this model.')
        return ok.__wrapped__(**kwargs) if hasattr(ok, "__wrapped__") else fake._resp

    fake.messages.create = picky
    llm = _real_llm(fake)
    assert llm.complete(system="s", messages=[Message("user", "go")], model="claude-new-7").text.startswith("Hello")
    assert [c["thinking"]["type"] for c in fake.calls] == ["disabled", "adaptive"]
    llm.complete(system="s", messages=[Message("user", "go")], model="claude-new-7")
    assert fake.calls[-1]["thinking"]["type"] == "adaptive" and len(fake.calls) == 3  # no second failed try


def test_a_model_that_asks_for_between_tools_gets_it_and_it_is_remembered():
    """Anthropic's Nov 2026 wording: the error names the setting to send instead."""
    from app.llm import Message

    fake = _FakeAnthropic("Hi Sinister, your Snowboard is saved.")
    error = ('Error code: 400 - {\'type\': \'error\', \'error\': {\'type\': \'invalid_request_error\', '
             '\'message\': \'To turn thinking off on this model, send "thinking": {"type": "between_tools"} '
             'instead of {"type": "disabled"}.\'}}')

    def picky(**kwargs):
        fake.calls.append(kwargs)
        if kwargs["thinking"] == {"type": "disabled"}:
            raise RuntimeError(error)
        return fake._resp

    fake.messages.create = picky
    llm = _real_llm(fake)
    for _ in range(2):
        assert llm.complete(system="s", messages=[Message("user", "go")], model="claude-haiku-5-5").text
    assert [c["thinking"]["type"] for c in fake.calls] == ["disabled", "between_tools", "between_tools"]
    assert fake.calls[-1]["max_tokens"] == 1024  # thinking stays off, so the normal limit


def test_an_unrelated_error_is_not_retried():
    from app.llm import Message

    fake = _FakeAnthropic("x")

    def broken(**kwargs):
        fake.calls.append(kwargs)
        raise RuntimeError("Error code: 401 - invalid x-api-key")

    fake.messages.create = broken
    import pytest

    with pytest.raises(RuntimeError, match="invalid x-api-key"):
        _real_llm(fake).complete(system="s", messages=[Message("user", "go")], model="claude-haiku-5-5")
    assert len(fake.calls) == 1


def test_a_judge_from_any_provider_is_read_even_inside_a_code_block():
    """Gemini tends to wrap JSON in ```json fences; the judge reply must still be read."""
    from app.evals.scorer import _parse

    fenced = 'Here is my verdict:\n```json\n{"helpfulness": 4, "accuracy": 5, "tone": 9, "notes": "ok"}\n```'
    assert _parse(fenced) == {"helpfulness": 4, "accuracy": 5, "tone": 5, "notes": "ok"}
    assert _parse("no json here") is None
