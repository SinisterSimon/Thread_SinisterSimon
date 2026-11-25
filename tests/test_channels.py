"""Safe sending: allowlist, STOP, daily cap, console by default, signed Twilio replies. No network."""
import base64
import hashlib
import hmac
import os
import sys

os.environ["DATABASE_URL"] = "sqlite://"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db  # noqa: E402
from app.channels import handle_reply, normalize, send_sms, twilio_signature_ok  # noqa: E402
from app.models import OptOut, SentMessage  # noqa: E402

ME = "+3725153479"


@pytest.fixture()
def session(monkeypatch):
    monkeypatch.setenv("SEND_ALLOWLIST", "+372 5153 479")
    monkeypatch.delenv("SEND_MODE", raising=False)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_console_mode_by_default_logs_and_sends_nothing(session, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("console mode must not call out"))
    r = send_sms(session, ME, "Hi Sinister, your Snowboard is saved.")
    assert r.status == "logged"
    row = session.scalars(select(SentMessage)).one()
    assert row.body.endswith("Reply STOP to opt out.") and row.channel == "console"


def test_only_allowlisted_numbers_never_opted_out_and_within_the_daily_cap(session, monkeypatch):
    assert send_sms(session, "+3725550000", "hi").detail == "not on SEND_ALLOWLIST"
    assert send_sms(session, "5153479", "hi").detail.startswith("not an international number")
    monkeypatch.setenv("SEND_PER_DAY", "2")
    assert [send_sms(session, ME, "hi").status for _ in range(3)] == ["logged", "logged", "blocked"]
    session.add(OptOut(number=ME, created_at=__import__("datetime").datetime.now()))
    session.commit()
    monkeypatch.setenv("SEND_PER_DAY", "10")
    assert send_sms(session, ME, "hi").detail == "this number replied STOP"
    statuses = [m.status for m in session.scalars(select(SentMessage).order_by(SentMessage.id))]
    assert statuses.count("blocked") == 4  # every attempt is logged, blocked ones too


def test_twilio_mode_sends_through_twilio_with_the_opt_out_line(session, monkeypatch):
    for k, v in {"SEND_MODE": "twilio", "TWILIO_ACCOUNT_SID": "AC1", "TWILIO_AUTH_TOKEN": "tok",
                 "TWILIO_FROM": "+15550001111"}.items():
        monkeypatch.setenv(k, v)
    seen = {}

    def post(url, auth, data, timeout):
        seen.update(url=url, auth=auth, data=data)
        return httpx.Response(201, json={"sid": "SM123"})

    monkeypatch.setattr(httpx, "post", post)
    r = send_sms(session, ME, "Your Snowboard is saved.")
    assert (r.status, r.detail) == ("sent", "SM123")
    assert seen["url"].endswith("/Accounts/AC1/Messages.json") and seen["auth"] == ("AC1", "tok")
    assert seen["data"] == {"To": ME, "From": "+15550001111", "Body": "Your Snowboard is saved. Reply STOP to opt out."}


def test_stop_and_start_replies(session):
    assert normalize("+372 5153-479") == ME and normalize("abc") == ""
    assert handle_reply(session, ME, "STOP") == "opted out" and session.get(OptOut, ME)
    assert handle_reply(session, ME, "Start") == "opted in" and session.get(OptOut, ME) is None
    assert handle_reply(session, ME, "is it waterproof?") == "reply"


def _twilio_sig(url: str, params: dict, token: str) -> str:
    payload = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()


def test_twilio_inbound_needs_a_valid_signature_and_records_stop(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://thread.example")
    params = {"From": ME, "Body": "STOP"}
    sig = _twilio_sig("https://thread.example/webhooks/twilio", params, "tok")
    assert twilio_signature_ok("https://thread.example/webhooks/twilio", params, sig, "tok")
    with TestClient(main.app) as client:
        assert client.post("/webhooks/twilio", data=params, headers={"X-Twilio-Signature": "forged"}).status_code == 401
        assert client.post("/webhooks/twilio", data=params, headers={"X-Twilio-Signature": sig}).status_code == 200
    with db.SessionLocal() as s:
        assert s.get(OptOut, ME) is not None


def test_send_to_my_phone_goes_to_send_test_to_never_the_shopper(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    sys.path.insert(0, os.path.dirname(__file__))
    from test_shopify import fake_shopify

    monkeypatch.setenv("SEND_ALLOWLIST", ME)
    monkeypatch.setenv("SEND_TEST_TO", ME)
    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    with TestClient(main.app) as client:
        client.post("/shopify/sync")
        page = client.post("/shopify/draft", data={"agent_key": "baseline-haiku"}, follow_redirects=True).text
        assert "Log a test send" in page
        page = client.post("/shopify/send-test/1", follow_redirects=True).text
        assert "Logged only" in page and "Send log" in page
    with db.SessionLocal() as s:
        assert [m.to_number for m in s.scalars(select(SentMessage))] == [ME]  # SEND_TEST_TO, nobody else
