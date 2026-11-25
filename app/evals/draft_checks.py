"""Rule checks for a drafted first text on the Live store.

Plain rules, not an AI judge: free, instant and the same every time, so drafts from
different models can be compared fairly. Each check mirrors a rule in the baseline
prompt's "first text" section, plus SMS length. The score is the share of checks passed.
"""
import re
from dataclasses import dataclass

from app.experiments.runner import clean_name, clean_title

MAX_BODY_CHARS = 120  # the prompt's limit; the checkout link is added after it
FILLER = ("we noticed", "just checking in", "glad to help", "happy to help", "any questions",
          "don't hesitate", "do not hesitate", "feel free to")
PRESSURE = ("hurry", "expire", "last chance", "act now", "don't miss", "ends soon", "limited time",
            "only a few left", "before it's gone")
DISCOUNT = re.compile(r"\d+\s*%|\bdiscount\b|\bpromo\b|\bcode\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
# GSM-7 basic characters: a text using only these fits 160 per SMS, otherwise 70.
GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "ÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")


def sms_segments(text: str) -> int:
    """How many SMS a text is billed as (each segment costs the same)."""
    if not text:
        return 0
    if set(text) <= GSM7:
        return 1 if len(text) <= 160 else -(-len(text) // 153)
    return 1 if len(text) <= 70 else -(-len(text) // 67)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class DraftScore:
    checks: tuple[Check, ...]
    body_chars: int  # the agent's own text, without the link
    segments: int  # SMS segments of the whole message, link included

    @property
    def score(self) -> int:
        return round(100 * sum(c.passed for c in self.checks) / len(self.checks))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def score_draft(text: str, *, first_name: str, products: list[str], link: str | None) -> DraftScore | None:
    """Score one drafted first text. None for a draft that failed to generate."""
    if not text or text.startswith("(No draft"):
        return None
    body = text[: -len(link)].rstrip() if link and text.endswith(link) else text
    lower = body.lower()
    checks = [Check("short", len(body) <= MAX_BODY_CHARS, f"{len(body)} chars")]
    if link:
        checks.append(Check("link", text.endswith(link), "checkout link at the end"))
    checks.append(Check("no invented link", "http" not in lower and "www." not in lower))
    name = clean_name(first_name)
    if name != "there":
        checks.append(Check("first name", re.search(rf"\b{re.escape(name)}\b", body) is not None, name))
    words = {w.lower() for p in products for w in clean_title(p).split() if len(w) >= 4}
    titles = [clean_title(p).lower() for p in products]
    checks.append(Check("product", any(t in lower for t in titles) or any(w in lower for w in words)))
    filler = next((f for f in FILLER if f in lower), None)
    checks.append(Check("no filler", filler is None, f'"{filler}"' if filler else ""))
    pressure = next((p for p in PRESSURE if p in lower), None) or ("!" if "!" in body else None)
    checks.append(Check("no pressure", pressure is None, f'"{pressure}"' if pressure else ""))
    checks.append(Check("no discount", DISCOUNT.search(body) is None))
    checks.append(Check("no emoji", EMOJI.search(body) is None))
    return DraftScore(tuple(checks), len(body), sms_segments(text))


@dataclass(frozen=True)
class RunScore:
    drafts: int
    avg_score: int
    avg_chars: int
    avg_segments: float


def summarize_scores(scores: list[DraftScore | None]) -> RunScore | None:
    ok = [s for s in scores if s is not None]
    if not ok:
        return None
    n = len(ok)
    return RunScore(n, round(sum(s.score for s in ok) / n), round(sum(s.body_chars for s in ok) / n),
                    round(sum(s.segments for s in ok) / n, 1))
