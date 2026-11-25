"""LLM clients: a free offline stand-in and real models from Anthropic and Google.

Pick one with the LLM_MODE environment variable:
- "offline" (default): scripted replies, no network, no cost
- "live" (or its older name "anthropic"): real models. Each call goes to the
  provider that owns the model name: claude-* to Anthropic (ANTHROPIC_API_KEY),
  gemini-* to Google (GEMINI_API_KEY). A key is only needed for the models used.
"""
import os
import time
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


class GeminiLLM:
    """Google's Gemini models through the google-genai SDK.

    Same contract as AnthropicLLM: one short text, or an LLMError saying why not.
    Gemini 3.x models think before answering; thinking_level "low" keeps that short.
    Thinking tokens are billed as output, so they are counted in output_tokens and
    show up in the cost. A model that rejects thinking_level is retried without it
    and remembered, like the Anthropic retry.
    """

    MAX_TOKENS = 4096  # room for short thinking plus a one- or two-sentence reply
    # Busy (503) and per-minute limits (429) usually clear within seconds: wait, then try again.
    RETRY_DELAYS = (2, 5)
    BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

    def __init__(self, api_key: str):
        from google import genai  # imported lazily so offline mode needs no SDK

        self._client = genai.Client(api_key=api_key)
        self._no_thinking_level: set[str] = set()
        self._sleep = time.sleep  # swapped out in tests

    def _config(self, model: str, system: str):
        from google.genai import types

        thinking = None if model in self._no_thinking_level else types.ThinkingConfig(thinking_level="low")
        return types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=self.MAX_TOKENS, thinking_config=thinking
        )

    def _generate(self, model: str, system: str, messages: list[Message]):
        from google.genai import types

        # Gemini calls the assistant side "model".
        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.text)])
            for m in messages
        ]
        retries = 0
        while True:
            try:
                return self._client.models.generate_content(
                    model=model, contents=contents, config=self._config(model, system)
                )
            except Exception as err:
                text = str(err)
                if "thinking" in text.lower() and model not in self._no_thinking_level:
                    self._no_thinking_level.add(model)  # retry without thinking_level, remembered
                    continue
                if retries < len(self.RETRY_DELAYS) and _gemini_transient(text):
                    self._sleep(self.RETRY_DELAYS[retries])
                    retries += 1
                    continue
                # One plain sentence instead of Google's raw error, and an LLMError so a failed
                # draft shows "(No draft: ...)" while the rest of the run carries on.
                raise LLMError(_gemini_explain(model, text)) from err

    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply:
        system = system.split(OfflineLLM.MARKER, 1)[0]  # never send offline scripts
        resp = self._generate(model, system, messages)
        candidate = resp.candidates[0] if resp.candidates else None
        reason = getattr(getattr(candidate, "finish_reason", None), "name", None) or "none"
        parts = (candidate.content.parts or []) if candidate and candidate.content else []
        text = "".join(p.text or "" for p in parts if not getattr(p, "thought", False))
        if reason == "MAX_TOKENS":
            raise LLMError(f"{model} hit the {self.MAX_TOKENS}-token limit; the reply would be cut off")
        if reason in self.BLOCKED or candidate is None:
            raise LLMError(f"{model} declined to write this message (finish reason: {reason})")
        if not text.strip():
            raise LLMError(f"{model} returned no text (finish reason: {reason})")
        usage = resp.usage_metadata
        return LLMReply(
            text=text.strip(),
            model=model,
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        )


def _gemini_transient(text: str) -> bool:
    """Errors worth one more try: Google busy or a short rate limit (not a model the plan lacks)."""
    if "limit: 0" in text:
        return False  # the free tier has no quota for this model; waiting won't help
    return any(code in text for code in ("503", "UNAVAILABLE", "500", "INTERNAL", "429", "RESOURCE_EXHAUSTED"))


def _gemini_explain(model: str, text: str) -> str:
    if "limit: 0" in text:
        return f"{model} isn't available on the Gemini free tier; turn on billing for the Gemini API key"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return f"{model} hit a Gemini rate limit (the free tier allows a few requests a minute and a day); try later"
    if "503" in text or "UNAVAILABLE" in text:
        return f"{model} is busy at Google right now (503); try again in a minute"
    first_line = text.splitlines()[0] if text else "no details"
    return f"{model} call failed: {first_line[:160]}"


PROVIDERS = {
    # model-name prefix: (client class, env var holding its key)
    "claude-": (AnthropicLLM, "ANTHROPIC_API_KEY"),
    "gemini-": (GeminiLLM, "GEMINI_API_KEY"),
}


class RoutedLLM:
    """Sends each call to the provider that owns the model, creating clients on first use.

    A missing key fails only the calls that need it, as an LLMError: a Gemini
    draft without GEMINI_API_KEY shows "(No draft: ...)" while Claude drafts work.
    """

    def __init__(self):
        self._clients: dict[str, LLM] = {}

    def _client_for(self, model: str) -> LLM:
        prefix = next((p for p in PROVIDERS if model.startswith(p)), None)
        if prefix is None:
            raise LLMError(f"no provider for model {model!r}; known prefixes: {', '.join(PROVIDERS)}")
        if prefix not in self._clients:
            cls, env = PROVIDERS[prefix]
            key = os.getenv(env)
            if not key:
                raise LLMError(f"{model} needs {env}")
            self._clients[prefix] = cls(key)
        return self._clients[prefix]

    def complete(self, *, system: str, messages: list[Message], model: str) -> LLMReply:
        return self._client_for(model).complete(system=system, messages=messages, model=model)


LIVE_MODES = {"live", "anthropic"}  # "anthropic" is the older name, kept so existing settings work


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
    return RoutedLLM() if mode_for(shadow) in LIVE_MODES else OfflineLLM()
