"""The simulated shopper on the other end of the conversation."""
from app.llm import LLM, Message
from app.llm.client import OfflineLLM
from app.shoppers.personas import Persona

SHOPPER_MODEL = "claude-haiku-5-5"


class ShopperBot:
    """Plays a persona. Offline it reads scripted lines; online an LLM role-plays."""

    def __init__(self, persona: Persona, llm: LLM, first_name: str, cart_summary: str):
        self.persona = persona
        self.llm = llm
        self.first_name = first_name
        self.cart_summary = cart_summary
        self._turn = 0

    def reply(self, history: list[Message]) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). History is from the agent's view."""
        scripted = self.persona.lines[min(self._turn, len(self.persona.lines) - 1)]
        self._turn += 1
        system = (
            f"You are {self.first_name}, a shopper who left items in an online cart: "
            f"{self.cart_summary}. Persona: {self.persona.description} "
            "Reply to the store's text message in one short, casual sentence. "
            "Stay in character; never say you are an AI."
            f"\n{OfflineLLM.MARKER} {scripted}"
        )
        # From the shopper's side, the agent's messages are the "user" turns.
        flipped = [
            Message(role="user" if m.role == "assistant" else "assistant", text=m.text)
            for m in history
        ]
        out = self.llm.complete(system=system, messages=flipped, model=SHOPPER_MODEL)
        return out.text, out.input_tokens, out.output_tokens
