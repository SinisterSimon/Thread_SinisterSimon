"""Sending real messages, safely.

Every send goes through `send_sms`, which refuses unless all of these hold:
- the number is on SEND_ALLOWLIST (your own phone; comma-separated, +372... form),
- the number hasn't replied STOP,
- fewer than SEND_PER_DAY messages went out in the last 24 hours (default 10).

SEND_MODE picks the carrier: "console" (default) logs the message and sends nothing;
"twilio" sends through Twilio (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM).
Every attempt, blocked ones included, is written to sent_messages.
"""
import base64
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import OptOut, SentMessage

OPT_OUT_LINE = "Reply STOP to opt out."
STOP_WORDS = {"stop", "stopall", "unsubscribe", "cancel", "end", "quit"}
START_WORDS = {"start", "unstop", "yes"}


@dataclass(frozen=True)
class SendResult:
    status: str  # sent | logged | blocked | failed
    detail: str


def normalize(number: str) -> str:
    """'+372 5153 479' -> '+3725153479'. Only international (+country) numbers are accepted."""
    digits = re.sub(r"[^\d+]", "", number or "")
    return digits if re.fullmatch(r"\+\d{7,15}", digits) else ""


def allowlist() -> set[str]:
    return {n for n in (normalize(x) for x in os.getenv("SEND_ALLOWLIST", "").split(",")) if n}


def with_opt_out(body: str) -> str:
    return body if OPT_OUT_LINE.lower() in body.lower() else f"{body.rstrip()} {OPT_OUT_LINE}"


def _twilio(to: str, body: str) -> SendResult:
    sid, token, sender = (os.getenv(k, "").strip() for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))
    if not (sid and token and sender):
        return SendResult("failed", "Twilio needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM")
    try:
        r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                       auth=(sid, token), data={"To": to, "From": sender, "Body": body}, timeout=15)
    except httpx.HTTPError as err:
        return SendResult("failed", f"Twilio unreachable: {err}"[:300])
    if r.status_code >= 400:
        message = r.json().get("message", r.text) if r.headers.get("content-type", "").startswith("application/json") \
            else r.text
        return SendResult("failed", f"Twilio {r.status_code}: {message}"[:300])
    return SendResult("sent", r.json().get("sid", ""))


def send_sms(session: Session, to: str, body: str, *, conversation_id: int | None = None) -> SendResult:
    """Send one text if every safety rule allows it, and log the attempt either way."""
    mode = os.getenv("SEND_MODE", "console").strip().lower()
    number = normalize(to)
    body = with_opt_out(body)
    per_day = int(os.getenv("SEND_PER_DAY", "10"))
    since = datetime.now() - timedelta(days=1)
    sent_today = session.scalar(select(func.count(SentMessage.id)).where(
        SentMessage.created_at >= since, SentMessage.status.in_(("sent", "logged")))) or 0

    if not number:
        result = SendResult("blocked", "not an international number (+country code)")
    elif number not in allowlist():
        result = SendResult("blocked", "not on SEND_ALLOWLIST")
    elif session.get(OptOut, number) is not None:
        result = SendResult("blocked", "this number replied STOP")
    elif sent_today >= per_day:
        result = SendResult("blocked", f"daily limit of {per_day} reached")
    elif mode == "twilio":
        result = _twilio(number, body)
    else:
        result = SendResult("logged", "console mode: nothing left Thread")

    session.add(SentMessage(conversation_id=conversation_id, to_number=number or (to or "")[:20], body=body[:1600],
                            channel="twilio" if mode == "twilio" else "console", status=result.status,
                            detail=result.detail[:300], created_at=datetime.now()))
    session.commit()
    return result


def twilio_signature_ok(url: str, params: dict[str, str], signature: str | None, token: str) -> bool:
    """Twilio signs inbound requests: base64 HMAC-SHA1 of the URL plus sorted key+value pairs."""
    if not signature or not token:
        return False
    payload = url + "".join(k + params[k] for k in sorted(params))
    expected = base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, signature)


def handle_reply(session: Session, sender: str, text: str) -> str:
    """Record STOP / START from an inbound text. Returns what happened."""
    number = normalize(sender)
    word = (text or "").strip().lower().strip(".!")
    if not number:
        return "ignored"
    if word in STOP_WORDS:
        if session.get(OptOut, number) is None:
            session.add(OptOut(number=number, created_at=datetime.now()))
            session.commit()
        return "opted out"
    if word in START_WORDS:
        row = session.get(OptOut, number)
        if row is not None:
            session.delete(row)
            session.commit()
        return "opted in"
    return "reply"
