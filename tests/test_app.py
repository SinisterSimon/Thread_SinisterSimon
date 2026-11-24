"""Tests run against a throwaway in-memory SQLite database."""
import os

os.environ["DATABASE_URL"] = "sqlite://"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db  # noqa: E402
from app.models import Checkout, CheckoutStatus, Order  # noqa: E402
from app.simulator import SimConfig, run_simulation  # noqa: E402


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_simulation_is_reproducible(session):
    a = run_simulation(session, SimConfig(seed=7, days=5))
    b = run_simulation(session, SimConfig(seed=7, days=5))
    assert (a.checkouts, a.abandoned, a.natural_returns) == (
        b.checkouts, b.abandoned, b.natural_returns,
    )


def test_abandonment_rate_is_close_to_config(session):
    s = run_simulation(session, SimConfig(seed=1, days=30, abandon_rate=0.4))
    assert 0.33 < s.abandoned / s.checkouts < 0.47


def test_every_completed_checkout_has_an_order(session):
    s = run_simulation(session, SimConfig(seed=3, days=5))
    completed = session.scalar(
        select(func.count(Checkout.id)).where(
            Checkout.store_id == s.store_id, Checkout.status == CheckoutStatus.completed
        )
    )
    same_session = session.scalar(
        select(func.count(Order.id)).where(
            Order.store_id == s.store_id, Order.source == "same_session"
        )
    )
    assert completed == same_session == s.completed


def test_natural_returns_only_come_from_abandoned_checkouts(session):
    s = run_simulation(session, SimConfig(seed=5, days=10))
    bad = session.scalar(
        select(func.count(Order.id))
        .join(Checkout, Checkout.id == Order.checkout_id)
        .where(Order.source == "returned", Checkout.status != CheckoutStatus.abandoned)
    )
    assert bad == 0
    assert s.natural_returns > 0


def test_dashboard_flow():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "test lab for AI sales agents" in client.get("/").text
        assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg+xml")
        assert "No store data yet" in client.get("/store").text
        r = client.post("/simulate", data={"days": 3}, follow_redirects=True)
        assert r.status_code == 200 and "Abandoned" in r.text
        assert client.get("/api/metrics").json()["checkouts"] > 0


def test_demo_and_run_all(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    monkeypatch.setattr(main, "DEMO_DAYS", 10)
    with TestClient(main.app) as client:
        r = client.post("/demo", follow_redirects=True)
        assert r.status_code == 200 and "Leaderboard" in r.text
        runs = [client.get(f"/api/experiments/{i}").json() for i in (1, 2, 3)]
        assert [(x["agent"], x["guardrails"], x["holdout_pct"], x["status"]) for x in runs] == [
            ("baseline-haiku", False, 50, "done"),
            ("pushy-haiku", False, 50, "done"),
            ("pushy-haiku", True, 50, "done"),
        ]
        assert all(x["estimate"] is not None for x in runs)

        client.post("/experiments/run-all", data={"holdout_pct": 20})
        assert client.get("/api/experiments/6").json()["holdout_pct"] == 20


def test_simulation_runs_after_the_response_and_clears_the_building_flag(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.engine = engine
    db.SessionLocal.configure(bind=engine)
    import app.main as main

    with TestClient(main.app) as client:
        # The request itself only schedules the work, so it returns at once.
        r = client.post("/simulate", data={"days": 3}, follow_redirects=False)
        assert r.status_code == 303
        assert not main.is_building()
        assert client.get("/api/metrics").json()["checkouts"] > 0

        def broken(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(main, "run_simulation", broken)
        client.post("/simulate", data={"days": 3})
        assert not main.is_building()  # a failed run must not leave pages refreshing forever
        assert "Building a new store" not in client.get("/store").text


def test_building_banner_shows_while_a_store_is_simulating():
    import app.main as main

    main._mark_building(1)
    try:
        with TestClient(main.app) as client:
            page = client.get("/store").text
            assert "Building a new store" in page and 'http-equiv="refresh"' in page
    finally:
        main._mark_building(-1)
