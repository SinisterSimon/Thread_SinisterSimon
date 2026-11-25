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
        "abandonedCheckoutUrl": f"https://thread-dev.myshopify.com/checkouts/{n}/recover?key=k{n}",
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
        # The form keeps the last run's agent and guardrails, instead of snapping back to Haiku.
        assert '<option value="pushy-haiku" selected>' in page
        assert 'name="guardrails" value="true" checked>' in page
        page = client.post("/shopify/draft", data={"agent_key": "baseline-gemini-flash"}, follow_redirects=True).text
        assert '<option value="baseline-gemini-flash" selected>' in page
        assert 'name="guardrails" value="true" checked>' not in page
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


def test_live_store_shows_what_each_draft_cost(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    with TestClient(main.app) as client:
        client.post("/shopify/sync")
        page = client.post("/shopify/draft", data={"agent_key": "baseline-fable"}, follow_redirects=True).text
        assert "Baseline · Fable" in page and "$0.00" in page  # fractions of a cent, shown with enough digits


def test_drafts_end_with_the_real_checkout_link_and_use_clean_names(session, monkeypatch):
    """The link comes from Shopify and is appended by Thread, so the model can't invent or mistype it."""
    from app.experiments import create_experiment, run_experiment
    from app.llm.client import OfflineLLM
    from app.models import Conversation

    nodes = [dict(c) for c in CHECKOUTS]
    nodes[0]["customer"] = dict(nodes[0]["customer"], firstName="sinister")
    client = fake_shopify()[0]
    monkeypatch.setattr("app.shopify.sync.fetch_abandoned_checkouts", lambda _c: nodes)
    store_id = sync_store(session, client).store_id
    exp = create_experiment(session, store_id, "baseline-haiku", 50, shadow=True)
    run_experiment(session, exp, OfflineLLM())
    ext = dict(session.execute(select(Checkout.id, Checkout.external_id)).all())
    drafts = {ext[c.checkout_id]: c.messages[0].text for c in session.scalars(select(Conversation))}
    first = drafts["gid://shopify/AbandonedCheckout/1"]
    assert first.startswith("Hi Sinister, your Compare at Price Snowboard is still saved at thread-dev.")
    assert first.endswith(" https://thread-dev.myshopify.com/checkouts/1/recover?key=k1")


def test_cleaning_names_titles_and_store_facts(monkeypatch):
    from app.experiments.runner import STORE_FACTS, clean_name, clean_title, store_facts

    assert [clean_name(n) for n in ("sinister", "WINNIE", "McKay", "", "there")] == [
        "Sinister", "Winnie", "McKay", "there", "there"]
    assert clean_title("The Multi-location Snowboard") == "Multi-location Snowboard"
    assert clean_title("Theory Jacket") == "Theory Jacket"
    shop, lab = Store(source="shopify"), Store(source="simulated")
    monkeypatch.setenv("SHOPIFY_STORE_FACTS", "Free shipping over $100 | Ships from Tallinn")
    assert store_facts(shop) == ["Free shipping over $100", "Ships from Tallinn"]
    assert store_facts(lab) == STORE_FACTS  # the lab always uses its own facts


def test_opening_rules_and_link_note_reach_the_model():
    from app.agents.base import AgentContext
    from app.agents.registry import BASELINE_RULES, _context_block

    assert "At most 120 characters" in BASELINE_RULES and "never write a URL" in BASELINE_RULES
    ctx = AgentContext("Thread Snow Co.", "Sinister", ["1 x Snowboard ($700)"], "$700", ["Free returns"],
                       checkout_url="https://x/recover")
    assert "Thread adds it after your first text" in _context_block(ctx)
    assert "Checkout link" not in _context_block(AgentContext("S", "A", [], "$0"))


def test_live_store_page_scores_every_draft_and_compares_agents(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    with TestClient(main.app) as client:
        client.post("/shopify/sync")
        client.post("/shopify/draft", data={"agent_key": "baseline-haiku"})
        page = client.post("/shopify/draft", data={"agent_key": "pushy-haiku"}, follow_redirects=True).text
    table = page.split("Draft scores by model", 1)[1].split("</table>", 1)[0]
    # The baseline follows the rules; the pushy agent's "20% off ... expire!" fails them.
    assert table.index("Baseline · Haiku") < table.index("Pushy discounter · Haiku")
    assert "✓ all checks" in page and "✗ no pressure" in page and "✗ no discount" in page


def _signed(body: bytes, secret: str = "shh") -> dict:
    import base64
    import hashlib
    import hmac

    sig = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return {"X-Shopify-Hmac-Sha256": sig, "Content-Type": "application/json"}


def test_webhook_rejects_unsigned_events_and_syncs_then_drafts_on_signed_ones(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    from datetime import datetime

    import app.main as main

    monkeypatch.setenv("SHOPIFY_CLIENT_SECRET", "shh")
    monkeypatch.setenv("AUTO_DRAFT_AGENT", "baseline-haiku")
    monkeypatch.setattr(main, "client_from_env", lambda: fake_shopify()[0])
    main._webhook_state.update(last_sync=datetime.min, last_draft=datetime.min)
    body = b'{"id": 1}'
    with TestClient(main.app) as client:
        def post(content, headers):
            return client.post("/webhooks/shopify", content=content, headers=headers).status_code

        assert post(body, {"X-Shopify-Hmac-Sha256": "forged"}) == 401
        assert post(b'{"id": 2}', _signed(body)) == 401  # signature for a different body
        assert post(body, _signed(body)) == 200
        page = client.get("/shopify").text
        assert "Abandoned checkouts (3)" in page and "Baseline · Haiku" in page  # synced, then drafted
        # A burst of events within seconds doesn't sync or draft again.
        client.post("/webhooks/shopify", content=body, headers=_signed(body))
        assert client.get("/api/experiments/2").status_code == 404


def test_register_subscribes_both_checkout_topics_once():
    from app.shopify.webhooks import register

    calls = []

    class Stub:
        def graphql(self, query, variables=None):
            calls.append(variables)
            if "webhookSubscriptions" in query:
                return {"webhookSubscriptions": {"nodes": [
                    {"id": "1", "topic": "CHECKOUTS_CREATE", "uri": "https://t.example/webhooks/shopify"}]}}
            return {"webhookSubscriptionCreate": {"webhookSubscription": {"id": "2"}, "userErrors": []}}

    report = register(Stub(), "https://t.example/webhooks/shopify")
    assert report == ["CHECKOUTS_CREATE: already subscribed", "CHECKOUTS_UPDATE: subscribed"]
    uri = "https://t.example/webhooks/shopify"
    assert calls[-1] == {"topic": "CHECKOUTS_UPDATE", "sub": {"uri": uri, "format": "JSON"}}


def test_bots_refuse_anything_but_a_dev_store(monkeypatch, capsys):
    import sys

    from bots import abandon_cart

    monkeypatch.setattr(sys, "argv", ["abandon_cart", "--store", "www.realshop.com"])
    assert abandon_cart.main() == 2
    assert "only run against a development store" in capsys.readouterr().out
