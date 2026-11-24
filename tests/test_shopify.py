"""v2: the Shopify connector and shadow mode, against a fake Shopify (no network)."""
import json
import os

os.environ["DATABASE_URL"] = "sqlite://"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db  # noqa: E402
from app.metrics import latest_store  # noqa: E402
from app.models import Checkout, CheckoutStatus, Customer, Store  # noqa: E402
from app.shopify import ShopifyClient, ShopifyError, sync_store  # noqa: E402
from app.simulator import SimConfig, run_simulation  # noqa: E402


def _checkout(n: int, email: str, phone: str | None = None, completed: bool = False) -> dict:
    return {
        "id": f"gid://shopify/AbandonedCheckout/{n}",
        "name": f"#{n}",
        "createdAt": "2026-11-24T20:18:00Z",
        "updatedAt": "2026-11-24T20:20:00Z",
        "completedAt": "2026-11-24T21:00:00Z" if completed else None,
        "totalPriceSet": {"shopMoney": {"amount": "794.22", "currencyCode": "EUR"}},
        "customer": {
            "id": f"gid://shopify/Customer/{n}",
            "firstName": f"Shopper{n}",
            "defaultEmailAddress": {"emailAddress": email},
            "defaultPhoneNumber": None,
        },
        "shippingAddress": {"firstName": f"Shopper{n}", "phone": phone},
        "billingAddress": None,
        "lineItems": {"nodes": [{
            "title": "The Compare at Price Snowboard",
            "quantity": 1,
            "originalUnitPriceSet": {"shopMoney": {"amount": "785.95"}},
            "product": {
                "id": "gid://shopify/Product/1", "title": "The Compare at Price Snowboard", "productType": "snowboard",
            },
        }]},
    }


CHECKOUTS = [
    _checkout(1, "test1@example.com"),
    _checkout(2, "test2@example.com"),
    _checkout(3, "test3@example.com", phone="+3725153479"),
    _checkout(4, "paid@example.com", completed=True),
]


def fake_shopify(token_status: int = 200, graphql_errors: list | None = None):
    calls = {"token": 0, "graphql": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/oauth/access_token":
            calls["token"] += 1
            assert b"grant_type=client_credentials" in request.content
            if token_status != 200:
                return httpx.Response(token_status, text='{"error":"invalid_client"}')
            return httpx.Response(200, json={"access_token": "tok", "scope": "read_orders", "expires_in": 86399})
        assert request.headers["X-Shopify-Access-Token"] == "tok"
        calls["graphql"] += 1
        if graphql_errors:
            return httpx.Response(200, json={"errors": graphql_errors})
        query = json.loads(request.content)["query"]
        if "shop {" in query:
            return httpx.Response(200, json={"data": {"shop": {"name": "thread-dev", "currencyCode": "EUR"}}})
        return httpx.Response(200, json={"data": {"abandonedCheckouts": {
            "nodes": CHECKOUTS, "pageInfo": {"hasNextPage": False, "endCursor": None}}}})

    client = ShopifyClient("thread-dev.myshopify.com", "id", "secret", transport=httpx.MockTransport(handler))
    return client, calls


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_sync_imports_real_checkouts_and_is_safe_to_repeat(session):
    client, calls = fake_shopify()
    first = sync_store(session, client)
    assert (first.fetched, first.new_checkouts) == (4, 4)

    store = session.get(Store, first.store_id)
    assert (store.source, store.currency, store.name) == ("shopify", "EUR", "thread-dev")
    abandoned = session.scalars(select(Checkout).where(Checkout.status == CheckoutStatus.abandoned)).all()
    assert len(abandoned) == 3 and abandoned[0].total_cents == 79422
    phone_customer = session.scalars(select(Customer).where(Customer.phone.is_not(None))).one()
    assert phone_customer.preferred_channel == "sms"
    # Consent is unknown for real shoppers, so nobody is marked as textable.
    assert not session.scalar(select(func.count()).select_from(Customer).where(Customer.sms_consent))

    again = sync_store(session, client)
    assert (again.new_checkouts, again.updated_checkouts) == (0, 0)
    assert session.scalar(select(func.count(Checkout.id))) == 4
    assert calls["token"] == 1  # the 24-hour token is reused


def test_bad_credentials_and_missing_access_explain_themselves(session):
    client, _ = fake_shopify(token_status=400)
    with pytest.raises(ShopifyError, match="SHOPIFY_CLIENT_ID"):
        sync_store(session, client)
    client, _ = fake_shopify(graphql_errors=[{"message": "Access denied", "extensions": {"code": "ACCESS_DENIED"}}])
    with pytest.raises(ShopifyError, match="Protected customer data"):
        sync_store(session, client)


def test_simulated_pages_never_pick_up_the_real_store(session):
    sim = run_simulation(session, SimConfig(days=2))
    sync_store(session, fake_shopify()[0])
    assert latest_store(session).id == sim.store_id
    assert latest_store(session, source="shopify").source == "shopify"


def test_shadow_mode_drafts_without_sending(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    with TestClient(main.app) as client:
        assert "Sync from Shopify" in client.get("/shopify").text
        page = client.post("/shopify/sync", follow_redirects=True).text
        assert "4 checkouts from Shopify, 4 new" in page and "Abandoned checkouts (3)" in page
        assert "t***@example.com" in page and "test1@example.com" not in page  # contact details masked

        page = client.post("/shopify/draft", data={"agent_key": "pushy-haiku"}, follow_redirects=True).text
        assert "Shopper1" in page and "unprompted" in page
        exp = client.get("/api/experiments/1").json()
        assert exp["shadow"] and exp["status"] == "done" and exp["conversations"] == 0  # no treatment arm, no outcome

        page = client.post("/shopify/draft", data={"agent_key": "pushy-haiku", "guardrails": "true"},
                           follow_redirects=True).text
        assert "guardrails changed" in page
        # The simulated lab is untouched.
        assert "Simulate a store first" in client.get("/experiments").text


def test_page_explains_setup_when_not_configured(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "client_from_env", lambda: None)
    with TestClient(main.app) as client:
        assert "No Shopify store connected" in client.get("/shopify").text


def test_one_failed_model_reply_does_not_sink_the_other_drafts(session):
    from app.experiments import create_experiment, run_experiment
    from app.llm.client import LLMError, OfflineLLM
    from app.models import Conversation

    class Flaky(OfflineLLM):
        calls = 0

        def complete(self, **kw):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise LLMError("claude-haiku-5-5 returned no text (stop reason: end_turn)")
            return super().complete(**kw)

    store_id = sync_store(session, fake_shopify()[0]).store_id
    exp = create_experiment(session, store_id, "baseline-haiku", 50, shadow=True)
    run_experiment(session, exp, Flaky())
    assert exp.status.value == "done"
    texts = [c.messages[0].text for c in session.scalars(select(Conversation)).all()]
    assert len(texts) == 3 and sum(t.startswith("(No draft: ") for t in texts) == 1


def test_drafts_can_use_real_claude_while_the_lab_stays_offline(monkeypatch):
    from app.llm import mode_for

    monkeypatch.setenv("LLM_MODE", "offline")
    monkeypatch.delenv("DRAFT_LLM_MODE", raising=False)
    assert (mode_for(), mode_for(shadow=True)) == ("offline", "offline")  # one setting still drives both
    monkeypatch.setenv("DRAFT_LLM_MODE", "anthropic")
    assert (mode_for(), mode_for(shadow=True)) == ("offline", "anthropic")


def test_real_model_drafts_are_capped_per_day(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main
    from app.llm.client import OfflineLLM

    asked = []
    monkeypatch.setenv("DRAFT_LLM_MODE", "anthropic")
    monkeypatch.setattr(main, "DRAFT_RUNS_PER_DAY", 1)
    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    monkeypatch.setattr(main, "get_llm", lambda shadow=False: asked.append(shadow) or OfflineLLM())
    with TestClient(main.app) as client:
        client.post("/shopify/sync")
        assert "1 run left today" in client.get("/shopify").text
        client.post("/shopify/draft", data={"agent_key": "baseline-haiku"})
        assert asked == [True]  # the draft run asked for the draft model, not the lab's
        page = client.post("/shopify/draft", data={"agent_key": "baseline-haiku"}, follow_redirects=True).text
        assert "Daily limit" in page and asked == [True]  # the second run never started
