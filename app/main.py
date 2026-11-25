"""Web app: dashboard, experiments, transcripts and a small JSON API."""
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.agents import AGENTS
from app.config import APP_NAME
from app.db import SessionLocal, get_session, init_db
from app.experiments import create_experiment, run_experiment, summarize
from app.llm import get_llm, mode_for
from app.metrics import latest_store, store_metrics
from app.models import Checkout, CheckoutItem, CheckoutStatus, Conversation, Experiment
from app.shopify import ShopifyError, client_from_env, sync_store
from app.shoppers import PERSONAS
from app.simulator import SimConfig, run_simulation

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["money"] = lambda cents: f"{'−' if cents < 0 else ''}${abs(cents) / 100:,.0f}"
templates.env.globals.update(
    app_name=APP_NAME,
    agents=AGENTS,
    personas={p.key: p for p in PERSONAS},
    repo_url=os.getenv("REPO_URL", "https://github.com/SinisterSimon/Thread_SinisterSimon"),
)

# The agents compared by "Run all agents" and the demo: the good one, the bad one,
# and the bad one behind guardrails.
COMPARISON = (("baseline-haiku", False), ("pushy-haiku", False), ("pushy-haiku", True))
# One prompt on four models, same shoppers, same holdout split. 100 checkouts with
# a 50% holdout means about 50 messaged shoppers per model: around $4 in total
# with real models (mostly Opus and Fable), free offline.
MODEL_COMPARISON = ("baseline-haiku", "baseline-sonnet", "baseline-opus", "baseline-fable")
MODEL_COMPARISON_CHECKOUTS = 100
MODEL_COMPARISON_HOLDOUT_PCT = 50
DEMO_DAYS = 180
DEMO_HOLDOUT_PCT = 50

FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#3c3489"/>'
    '<path d="M14 40c8-14 14 6 22-8s10-8 14-8" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>'
    "</svg>"
)

# With a separate worker service running (Render), experiments wait in the
# queue for it. Without one (local dev, free tier), the web app runs them
# itself in the background right after the request.
WORKER_ENABLED = os.getenv("WORKER_ENABLED", "0") == "1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)


log = logging.getLogger("thread")

# Simulations run after the response is sent, so a slow host never times the
# request out. This counter lets the pages say "building" and refresh
# themselves until the new store is in. One web process, so in-memory is enough.
_building = 0
_building_lock = threading.Lock()


def is_building() -> bool:
    return _building > 0


def _mark_building(delta: int) -> None:
    global _building
    with _building_lock:
        _building += delta


def _simulate_in_background(config: SimConfig, holdout_pct: int | None = None) -> None:
    """Build a store; with holdout_pct, then queue and run the three-agent comparison."""
    experiment_ids: list[int] = []
    try:
        with SessionLocal() as session:
            summary = run_simulation(session, config)
            if holdout_pct is not None:
                experiment_ids = _create_comparison(session, summary.store_id, holdout_pct)
    except Exception:
        log.exception("simulation failed")
    finally:
        _mark_building(-1)
    if not WORKER_ENABLED:
        for exp_id in experiment_ids:
            _run_in_background(exp_id)


def _run_in_background(experiment_id: int) -> None:
    with SessionLocal() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is not None:
            run_experiment(session, exp, get_llm(shadow=exp.shadow))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/favicon.svg")
@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(FAVICON, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(request, "landing.html", {})


@app.get("/store", response_class=HTMLResponse)
def dashboard(request: Request, session: Session = Depends(get_session)):
    store = latest_store(session)
    metrics = store_metrics(session, store) if store else None
    return templates.TemplateResponse(request, "dashboard.html", {"m": metrics, "building": is_building()})


@app.post("/simulate")
def simulate(background: BackgroundTasks, days: int = Form(30), seed: int = Form(42)):
    _mark_building(1)
    background.add_task(_simulate_in_background, SimConfig(seed=seed, days=max(1, min(days, 365))))
    return RedirectResponse("/store", status_code=303)


def _create_comparison(session: Session, store_id: int, holdout_pct: int) -> list[int]:
    return [
        create_experiment(session, store_id, agent_key, 5000, guardrails=guardrails, holdout_pct=holdout_pct).id
        for agent_key, guardrails in COMPARISON
    ]


def _start_comparison(session: Session, background: BackgroundTasks, store_id: int, holdout_pct: int) -> None:
    for exp_id in _create_comparison(session, store_id, holdout_pct):
        if not WORKER_ENABLED:
            background.add_task(_run_in_background, exp_id)


@app.post("/experiments/run-all")
def run_all(
    background: BackgroundTasks, holdout_pct: int = Form(DEMO_HOLDOUT_PCT), session: Session = Depends(get_session)
):
    """Baseline, pushy, and pushy with guardrails on every textable checkout of the latest store."""
    store = latest_store(session)
    if store is None:
        raise HTTPException(400, "simulate a store first")
    _start_comparison(session, background, store.id, holdout_pct)
    return RedirectResponse("/experiments", status_code=303)


@app.post("/experiments/compare-models")
def compare_models(background: BackgroundTasks, session: Session = Depends(get_session)):
    store = latest_store(session)
    if store is None:
        return RedirectResponse("/store", status_code=303)
    for agent_key in MODEL_COMPARISON:
        exp = create_experiment(
            session, store.id, agent_key, MODEL_COMPARISON_CHECKOUTS, holdout_pct=MODEL_COMPARISON_HOLDOUT_PCT
        )
        if not WORKER_ENABLED:
            background.add_task(_run_in_background, exp.id)
    return RedirectResponse("/experiments", status_code=303)


@app.post("/demo")
def demo(background: BackgroundTasks):
    """One click: a fresh 180-day store, then the three-agent comparison with a 50% holdout."""
    _mark_building(1)
    background.add_task(_simulate_in_background, SimConfig(days=DEMO_DAYS), DEMO_HOLDOUT_PCT)
    return RedirectResponse("/experiments", status_code=303)


def _range_scale(rows) -> tuple[float, float]:
    """One shared axis (in cents) for every leaderboard range bar, always including $0."""
    values = [0]
    for _, s in rows:
        values += [s.claimed_revenue_cents, s.caused_revenue_cents]
        if s.estimate:
            values += [s.estimate.ci_low, s.estimate.ci_high]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.04 or 1
    return lo - pad, hi + pad


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request, session: Session = Depends(get_session)):
    store = latest_store(session)
    exps = (
        list(session.scalars(select(Experiment).where(Experiment.store_id == store.id))) if store else []
    )
    rows = [(e, summarize(session, e)) for e in exps]
    # Leaderboard: what a real store could measure first (holdout estimate), then the truth.
    rows.sort(
        key=lambda r: (r[1].estimate.total_lift if r[1].estimate else float("-inf"), r[1].caused_revenue_cents),
        reverse=True,
    )
    reachable = store_metrics(session, store).reachable_abandoned if store else 0
    return templates.TemplateResponse(
        request,
        "experiments.html",
        {
            "store": store,
            "rows": rows,
            "scale": _range_scale(rows),
            "worker": WORKER_ENABLED,
            "reachable": reachable,
            "building": is_building(),
            "running": any(e.status.value in ("queued", "running") for e in exps),
            "lab_mode": mode_for(),
        },
    )


@app.post("/experiments")
def start_experiment(
    background: BackgroundTasks,
    agent_key: str = Form(...),
    sample_size: int = Form(50),
    guardrails: bool = Form(False),
    holdout_pct: int = Form(0),
    session: Session = Depends(get_session),
):
    if agent_key not in AGENTS:
        raise HTTPException(400, "unknown agent")
    store = latest_store(session)
    if store is None:
        raise HTTPException(400, "simulate a store first")
    exp = create_experiment(
        session, store.id, agent_key, max(1, min(sample_size, 5000)), guardrails=guardrails, holdout_pct=holdout_pct
    )
    if not WORKER_ENABLED:
        background.add_task(_run_in_background, exp.id)
    return RedirectResponse(f"/experiments/{exp.id}", status_code=303)


@app.get("/experiments/{exp_id}", response_class=HTMLResponse)
def experiment_detail(exp_id: int, request: Request, session: Session = Depends(get_session)):
    exp = session.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404)
    convs = list(
        session.scalars(
            select(Conversation)
            .where(Conversation.experiment_id == exp.id)
            .options(selectinload(Conversation.score))
            .order_by(Conversation.id)
        )
    )
    return templates.TemplateResponse(
        request, "experiment.html", {"e": exp, "s": summarize(session, exp), "convs": convs}
    )


@app.get("/conversations/{conv_id}", response_class=HTMLResponse)
def conversation_detail(conv_id: int, request: Request, session: Session = Depends(get_session)):
    conv = session.scalars(
        select(Conversation)
        .where(Conversation.id == conv_id)
        .options(selectinload(Conversation.messages), selectinload(Conversation.score))
    ).first()
    if conv is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "conversation.html", {"c": conv})


# --- v2: a real Shopify store, in shadow mode --------------------------------


def _mask_email(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    name, domain = email.split("@", 1)
    return f"{name[:1]}***@{domain}"


def _mask_phone(phone: str | None) -> str:
    return f"•••{phone[-2:]}" if phone else ""


def _usd_small(dollars: float) -> str:
    """Model costs are often fractions of a cent: show enough digits to compare them."""
    if dollars == 0:
        return "$0"
    if dollars < 0.01:
        return f"${dollars:.4f}"
    return f"${dollars:.3f}" if dollars < 1 else f"${dollars:,.2f}"


templates.env.filters["usd_small"] = _usd_small
templates.env.filters["mask_email"] = _mask_email
templates.env.filters["mask_phone"] = _mask_phone


# With a real model behind the public Live store, anyone can click "Draft".
# These caps keep that to cents: a run drafts at most DRAFT_MAX_CHECKOUTS
# messages, and at most DRAFT_RUNS_PER_DAY runs start in 24 hours.
DRAFT_MAX_CHECKOUTS = int(os.getenv("DRAFT_MAX_CHECKOUTS", "25"))
DRAFT_RUNS_PER_DAY = int(os.getenv("DRAFT_RUNS_PER_DAY", "30"))


def _draft_runs_left(session: Session) -> int | None:
    """Runs left today, or None when drafts are offline (free, so no cap)."""
    if mode_for(shadow=True) == "offline":
        return None
    since = datetime.now() - timedelta(days=1)
    used = session.scalar(
        select(func.count(Experiment.id)).where(Experiment.shadow, Experiment.created_at >= since)
    ) or 0
    return max(0, DRAFT_RUNS_PER_DAY - used)


@app.get("/shopify", response_class=HTMLResponse)
def shopify_page(request: Request, msg: str = "", error: str = "", session: Session = Depends(get_session)):
    client = client_from_env()
    store = latest_store(session, source="shopify")
    checkouts, shadows = [], []
    if store is not None:
        checkouts = list(
            session.scalars(
                select(Checkout)
                .where(Checkout.store_id == store.id, Checkout.status == CheckoutStatus.abandoned)
                .options(
                    selectinload(Checkout.customer),
                    selectinload(Checkout.items).selectinload(CheckoutItem.product),
                )
                .order_by(Checkout.started_at.desc())
            )
        )
        shadows = list(
            session.scalars(
                select(Experiment)
                .where(Experiment.store_id == store.id, Experiment.shadow)
                .options(selectinload(Experiment.conversations).selectinload(Conversation.messages))
                .order_by(Experiment.id.desc())
                .limit(3)
            )
        )
    by_id = {c.id: c for c in checkouts}
    return templates.TemplateResponse(
        request,
        "shopify.html",
        {
            "connected": client is not None,
            "domain": client.shop_domain if client else "",
            "store": store,
            "checkouts": checkouts,
            "by_id": by_id,
            "shadows": shadows,
            "running": any(e.status.value in ("queued", "running") for e in shadows),
            "draft_mode": mode_for(shadow=True),
            "runs_left": _draft_runs_left(session),
            "msg": msg,
            "error": error,
        },
    )


@app.post("/shopify/sync")
def shopify_sync(session: Session = Depends(get_session)):
    client = client_from_env()
    if client is None:
        return RedirectResponse("/shopify?error=" + quote("Shopify isn't configured yet."), status_code=303)
    try:
        r = sync_store(session, client)
    except ShopifyError as err:
        return RedirectResponse("/shopify?error=" + quote(str(err)), status_code=303)
    note = f"Synced: {r.fetched} checkouts from Shopify, {r.new_checkouts} new, {r.updated_checkouts} updated."
    return RedirectResponse("/shopify?msg=" + quote(note), status_code=303)


@app.post("/shopify/draft")
def shopify_draft(
    background: BackgroundTasks,
    agent_key: str = Form(...),
    guardrails: bool = Form(False),
    session: Session = Depends(get_session),
):
    if agent_key not in AGENTS:
        raise HTTPException(400, "unknown agent")
    store = latest_store(session, source="shopify")
    if store is None:
        raise HTTPException(400, "sync a Shopify store first")
    if _draft_runs_left(session) == 0:
        note = "Daily limit for real-model drafts reached. Try again tomorrow."
        return RedirectResponse("/shopify?error=" + quote(note), status_code=303)
    exp = create_experiment(session, store.id, agent_key, DRAFT_MAX_CHECKOUTS, guardrails=guardrails, shadow=True)
    if not WORKER_ENABLED:
        background.add_task(_run_in_background, exp.id)
    return RedirectResponse("/shopify", status_code=303)


@app.get("/api/metrics")
def api_metrics(session: Session = Depends(get_session)) -> dict:
    store = latest_store(session)
    return asdict(store_metrics(session, store)) if store else {}


@app.get("/api/experiments/{exp_id}")
def api_experiment(exp_id: int, session: Session = Depends(get_session)) -> dict:
    exp = session.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404)
    return {
        "id": exp.id,
        "agent": exp.agent_key,
        "guardrails": exp.guardrails,
        "holdout_pct": exp.holdout_pct,
        "shadow": exp.shadow,
        "status": exp.status.value,
        **asdict(summarize(session, exp)),
    }
