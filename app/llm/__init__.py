"""One interface for every language model Thread talks to.

Everything else in the app calls `get_llm()` and never imports a provider SDK
directly. That keeps providers swappable (Anthropic today, OpenAI later) and
lets tests and demos run in an offline mode that costs nothing.
"""
from app.llm.client import LLM, LLMReply, Message, get_llm

__all__ = ["LLM", "LLMReply", "Message", "get_llm"]
