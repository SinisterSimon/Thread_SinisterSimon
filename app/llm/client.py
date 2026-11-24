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


class AnthropicLLM:
    def __init__(self, api_key: str):
        import anthropic  # imported lazily so offline mode needs no SDK

        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply:
        system = system.split(OfflineLLM.MARKER, 1)[0]  # never send offline scripts
        resp = self._client.messages.create(
            model=model,
            max_tokens=300,
            system=system,
            messages=[{"role": m.role, "content": m.text} for m in messages],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMReply(
            text=text.strip(),
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


def get_llm() -> LLM:
    mode = os.getenv("LLM_MODE", "offline")
    if mode == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("LLM_MODE=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(key)
    return OfflineLLM()
