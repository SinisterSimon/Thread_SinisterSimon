"""LLM clients: a free offline stand-in and the real Anthropic API.

Pick one with the LLM_MODE environment variable:
- "offline" (default): scripted replies, no network, no cost
- "anthropic": real Claude models, needs ANTHROPIC_API_KEY
"""
import os
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    text: str


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply: ...


def traced(name: str, llm: LLM, *, system: str, messages: list[Message], model: str) -> LLMReply:
    """Call the model and record the call in Langfuse (a no-op without keys)."""
    from app.tracing import record_generation

    reply = llm.complete(system=system, messages=messages, model=model)
    record_generation(
        name=name,
        model=reply.model,
        system=system.split(OfflineLLM.MARKER, 1)[0],
        messages=[{"role": m.role, "content": m.text} for m in messages],
        output=reply.text,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )
    return reply


class OfflineLLM:
    """Deterministic stand-in for a real model.

    It never "thinks": callers pass a `scripted` reply through the system prompt
    marker below, so the conversation logic can be exercised end to end for
    free. Token counts are estimated (about 4 characters per token) so cost
    views still have something to show.
    """

    MARKER = "<<OFFLINE_REPLY>>"

    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply:
        text = system.split(self.MARKER, 1)[1].strip() if self.MARKER in system else "Okay."
        prompt_chars = len(system) + sum(len(m.text) for m in messages)
        return LLMReply(
            text=text,
            model=f"offline:{model}",
            input_tokens=prompt_chars // 4,
            output_tokens=max(1, len(text) // 4),
        )


class LLMError(RuntimeError):
    """The model gave no usable answer; the message says why."""


class AnthropicLLM:
    # Agents write one or two text-message sentences, so thinking adds nothing.
    # Newer Claude models think by default, and thinking tokens count toward
    # max_tokens: with a small limit the visible reply came back empty or cut
    # off mid-sentence ("Orders ship within 1"). So: thinking off, roomier limit,
    # and an error instead of a silently broken message.
    MAX_TOKENS = 1024
    # Opus 5.5 and Fable 5.2 refuse thinking={"type": "disabled"} (HTTP 400). They take
    # adaptive thinking with an effort level instead: "low" keeps thinking short, and the
    # roomier limit leaves space for the reply after it. Other models that reject
    # "disabled" are learned on the first 400 and get the same treatment.
    ADAPTIVE_ONLY = {"claude-opus-5-5", "claude-fable-5-2"}
    ADAPTIVE_MAX_TOKENS = 4096

    def __init__(self, api_key: str):
        import anthropic  # imported lazily so offline mode needs no SDK

        self._client = anthropic.Anthropic(api_key=api_key)
        self._adaptive = set(self.ADAPTIVE_ONLY)

    def _thinking(self, model: str) -> dict:
        if model in self._adaptive:
            return {
                "max_tokens": self.ADAPTIVE_MAX_TOKENS,
                "thinking": {"type": "adaptive"},
                "extra_body": {"output_config": {"effort": "low"}},
            }
        return {"max_tokens": self.MAX_TOKENS, "thinking": {"type": "disabled"}}

    def _create(self, model: str, system: str, messages: list[Message]):
        args = {
            "model": model,
            "system": system,
            "messages": [{"role": m.role, "content": m.text} for m in messages],
        }
        try:
            return self._client.messages.create(**args, **self._thinking(model)), model in self._adaptive
        except Exception as err:
            if "thinking.type.disabled" not in str(err) or model in self._adaptive:
                raise
            self._adaptive.add(model)
            return self._client.messages.create(**args, **self._thinking(model)), True

    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply:
        system = system.split(OfflineLLM.MARKER, 1)[0]  # never send offline scripts
        resp, adaptive = self._create(model, system, messages)
        text = "".join(block.text for block in resp.content if block.type == "text")
        if resp.stop_reason == "max_tokens":
            limit = self.ADAPTIVE_MAX_TOKENS if adaptive else self.MAX_TOKENS
            raise LLMError(f"{model} hit the {limit}-token limit; the reply would be cut off")
        if resp.stop_reason == "refusal":
            raise LLMError(f"{model} declined to write this message (stop reason: refusal)")
        if not text.strip():
            raise LLMError(f"{model} returned no text (stop reason: {resp.stop_reason})")
        return LLMReply(
            text=text.strip(),
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


def mode_for(shadow: bool = False) -> str:
    """Which model mode a run uses.

    LLM_MODE drives the simulated lab, where one click can mean thousands of
    calls. DRAFT_LLM_MODE drives shadow drafts on the Live store: one short
    message per real checkout. Keeping them apart lets the public site write
    real drafts while the lab stays offline and free. DRAFT_LLM_MODE falls back
    to LLM_MODE, so a single setting still works locally.
    """
    lab = os.getenv("LLM_MODE", "offline")
    return os.getenv("DRAFT_LLM_MODE", lab) if shadow else lab


def get_llm(shadow: bool = False) -> LLM:
    mode = mode_for(shadow)
    if mode == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("anthropic mode needs ANTHROPIC_API_KEY")
        return AnthropicLLM(key)
    return OfflineLLM()
