"""Gemini behind the same LLM contract, and routing calls to the provider that owns the model.

All offline: a fake stands in for the google-genai client, so no test calls Google.
"""
from types import SimpleNamespace as NS

import pytest

from app.llm import Message
from app.llm.client import LLMError, OfflineLLM, RoutedLLM, get_llm
from app.llm.pricing import cost_micros


def _resp(text, reason="STOP", thought=None, usage=(120, 20, 30)):
    parts = [NS(text=thought, thought=True)] if thought else []
    parts.append(NS(text=text, thought=False))
    prompt, out, thinking = usage
    return NS(
        candidates=[NS(finish_reason=NS(name=reason), content=NS(parts=parts))],
        usage_metadata=NS(prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thinking),
    )


class _FakeGenai:
    def __init__(self, resp):
        self.calls = []
        self._resp = resp
        self.models = NS(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def _gemini(fake):
    pytest.importorskip("google.genai")
    from app.llm.client import GeminiLLM

    llm = GeminiLLM(api_key="test")
    llm._client = fake
    return llm


def test_gemini_keeps_only_the_answer_and_bills_thinking_as_output():
    fake = _FakeGenai(_resp("Hi Winnie, your snowboard is still in your cart.", thought="Plan: be brief."))
    history = [Message("user", "go"), Message("assistant", "Hi!"), Message("user", "Is it waterproof?")]
    reply = _gemini(fake).complete(system="rules <<OFFLINE_REPLY>> script", messages=history,
                                   model="gemini-3.8-flash")
    assert reply.text == "Hi Winnie, your snowboard is still in your cart."  # the thought part is dropped
    assert (reply.input_tokens, reply.output_tokens) == (120, 50)  # 20 answer + 30 thinking tokens
    sent = fake.calls[0]
    assert "OFFLINE_REPLY" not in sent["config"].system_instruction
    assert sent["config"].thinking_config.thinking_level.name == "LOW"
    assert [c.role for c in sent["contents"]] == ["user", "model", "user"]  # Gemini's name for the assistant


def test_gemini_never_returns_an_empty_cut_off_or_blocked_message():
    for text, reason, why in [
        ("Orders ship within 1", "MAX_TOKENS", "token limit"),
        ("", "SAFETY", "declined"),
        ("", "STOP", "no text"),
    ]:
        with pytest.raises(LLMError, match=why):
            _gemini(_FakeGenai(_resp(text, reason))).complete(system="s", messages=[Message("user", "go")],
                                                              model="gemini-3.8-flash")


def test_a_gemini_model_that_rejects_thinking_level_is_retried_without_it_and_remembered():
    fake = _FakeGenai(_resp("Your cart is saved."))

    def picky(**kwargs):
        fake.calls.append(kwargs)
        if kwargs["config"].thinking_config is not None:
            raise RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported for this model.")
        return fake._resp

    fake.models.generate_content = picky
    llm = _gemini(fake)
    assert llm.complete(system="s", messages=[Message("user", "go")], model="gemini-old").text == "Your cart is saved."
    llm.complete(system="s", messages=[Message("user", "go")], model="gemini-old")
    assert [c["config"].thinking_config is None for c in fake.calls] == [False, True, True]


def test_calls_go_to_the_provider_that_owns_the_model(monkeypatch):
    import app.llm.client as client

    seen = []

    class Fake:
        def __init__(self, key):
            self.key = key

        def complete(self, *, system, messages, model):
            seen.append((model, self.key))
            return client.LLMReply("ok", model, 1, 1)

    monkeypatch.setattr(client, "PROVIDERS", {"claude-": (Fake, "ANTHROPIC_API_KEY"),
                                              "gemini-": (Fake, "GEMINI_API_KEY")})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    llm = RoutedLLM()
    for model in ("claude-haiku-5-5", "gemini-3.8-flash", "claude-opus-5-5"):
        llm.complete(system="s", messages=[Message("user", "go")], model=model)
    assert seen == [("claude-haiku-5-5", "a-key"), ("gemini-3.8-flash", "g-key"), ("claude-opus-5-5", "a-key")]

    # A missing key fails only the calls that need it, as an LLMError (drafts show it, nothing crashes).
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        RoutedLLM().complete(system="s", messages=[Message("user", "go")], model="gemini-3.8-flash")
    with pytest.raises(LLMError, match="no provider"):
        RoutedLLM().complete(system="s", messages=[Message("user", "go")], model="gpt-9")


def test_live_mode_and_its_old_name_both_route_and_offline_stays_free(monkeypatch):
    monkeypatch.delenv("DRAFT_LLM_MODE", raising=False)
    for mode in ("live", "anthropic"):
        monkeypatch.setenv("LLM_MODE", mode)
        assert isinstance(get_llm(), RoutedLLM)
    monkeypatch.setenv("LLM_MODE", "offline")
    assert isinstance(get_llm(), OfflineLLM)


def test_gemini_is_priced():
    # 1M input + 1M output on Gemini Flash = $0.75 + $3.75
    assert cost_micros("gemini-3.8-flash", 1_000_000, 1_000_000) == 4_500_000
