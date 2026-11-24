"""Web app: dashboard, experiments, transcripts and a small JSON API."""
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents import AGENTS
from app.config import APP_NAME
from app.db import SessionLocal, get_session, init_db
from app.experiments import create_experiment, run_experiment, summarize
from app.llm import get_llm
from app.metrics import latest_store, store_metrics
from app.models import Conversation, Experiment
from app.shoppers import PERSONAS
from app.simulator import SimConfig, run_simulation

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["money"] = lambda cents: f"{'−' if cents < 0 else ''}${abs(cents) / 100:,.0f}"
templates.env.globals.update(app_name=APP_NAME, agents=AGENTS, personas={p.key: p for p in PERSONAS})

# With a separate worker service running (Render), experiments wait in the
# queue for it. Without one (local dev, free tier), the web app runs them
# itself in the background right after the request.
WORKER_ENABLED = os.getenv("WORKER_ENABLED", "0") == "1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)


def _run_in_background(experiment_id: int) -> None:
    with SessionLocal() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is not None:
            run_experiment(session, exp, get_llm())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, session: Session = Depends(get_session)):
    store = latest_store(session)
    metrics = store_metrics(session, store) if store else None
    return templates.TemplateResponse(request, "dashboard.html", {"m": metrics})


@app.post("/simulate")
def simulate(days: int = Form(30), seed: int = Form(42), session: Session = Depends(get_session)):
    run_simulation(session, SimConfig(seed=seed, days=max(1, min(days, 365))))
    return RedirectResponse("/", status_code=303)


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
        {"store": store, "rows": rows, "worker": WORKER_ENABLED, "reachable": reachable},
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
        "status": exp.status.value,
        **asdict(summarize(session, exp)),
    }
