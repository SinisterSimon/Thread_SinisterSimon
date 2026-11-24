"""Optional Langfuse tracing.

With LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY set, every conversation becomes
a trace in Langfuse: each model call (agent, shopper, judge) is a generation
with its prompt, reply, model and token usage, and the judge's scores are
attached to the trace. Without the keys every function here does nothing, so
local runs and tests never need an account.
"""
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()  # reads keys and LANGFUSE_HOST from the environment
    except Exception:  # tracing must never break the app
        log.exception("Langfuse disabled: could not start client")
        return None


def enabled() -> bool:
    return _client() is not None


@contextmanager
def conversation_trace(name: str, metadata: dict[str, Any]) -> Iterator[None]:
    lf = _client()
    if lf is None:
        yield
        return
    with lf.start_as_current_observation(name=name, as_type="agent", metadata=metadata):
        yield


def record_generation(
    *, name: str, model: str, system: str, messages: list[dict], output: str, input_tokens: int, output_tokens: int
) -> None:
    lf = _client()
    if lf is None:
        return
    try:
        with lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input={"system": system, "messages": messages},
            output=output,
            usage_details={"input": input_tokens, "output": output_tokens},
        ):
            pass
    except Exception:
        log.exception("Langfuse: failed to record generation")


def score_trace(name: str, value: float, comment: str | None = None) -> None:
    lf = _client()
    if lf is None:
        return
    try:
        lf.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        log.exception("Langfuse: failed to record score")


def flush() -> None:
    lf = _client()
    if lf is not None:
        lf.flush()
