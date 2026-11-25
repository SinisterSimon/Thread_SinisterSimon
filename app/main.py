"""Web app: dashboard, experiments, transcripts and a small JSON API."""
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload
from starlette.middleware.gzip import GZipMiddleware

from app.agents import AGENTS
from app.config import APP_NAME
from app.db import SessionLocal, get_session, init_db
from app.evals.draft_checks import score_draft, summarize_scores
from app.experiments import create_experiment, run_experiment, summarize
from app.experiments.runner import done_cache
from app.llm import get_llm, mode_for
from app.llm.pricing import PRICES_PER_MTOK
from app.metrics import latest_store, store_metrics
from app.models import (
    Checkout,
    CheckoutItem,
    CheckoutStatus,
    Conversation,
    ConversationScore,
    Experiment,
    ExperimentStatus,
)
from app.shopify import ShopifyError, client_from_env, sync_store
from app.shoppers import PERSONAS
from app.simulator import SimConfig, run_simulation

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
# The emblem and tab icon live in app/static. Until they are added, the header keeps the "T"
# mark and the tab keeps the SVG icon, so a missing file never shows as a broken image.
STATIC_DIR = Path(__file__).parent / "static"
templates.env.filters["money"] = lambda cents: f"{'−' if cents < 0 else ''}${abs(cents) / 100:,.0f}"
# List price per 1M tokens: "$0.10", "$0.75", "$2", "$40".
templates.env.filters["per_mtok"] = lambda usd: f"${usd:,.0f}" if usd == int(usd) else f"${usd:,.2f}"
templates.env.globals.update(
    app_name=APP_NAME,
    agents=AGENTS,
    personas={p.key: p for p in PERSONAS},
    repo_url=os.getenv("REPO_URL", "https://github.com/SinisterSimon/Thread_SinisterSimon"),
    has_logo=(STATIC_DIR / "thread-logo.png").is_file(),
    has_favicon_png=(STATIC_DIR / "favicon.png").is_file(),
)

# The agents compared by "Run all agents" and the demo: the good one, the bad one,
# and the bad one behind guardrails.
COMPARISON = (("baseline-haiku", False), ("pushy-haiku", False), ("pushy-haiku", True))
# One prompt on four models, same shoppers, same holdout split. 100 checkouts with
# a 50% holdout means about 50 messaged shoppers per model: around $4 in total
# with real models (mostly Opus and Fable), free offline.
MODEL_COMPARISON = (
    "baseline-haiku", "baseline-sonnet", "baseline-opus", "baseline-fable",
    "baseline-gemini-flash", "baseline-gemini-argon",
)
MODEL_COMPARISON_CHECKOUTS = 100
CONVERSATIONS_PER_PAGE = 50  # a run page lists conversations 50 at a time, not all ~1,600
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


class CachedStatic(StaticFiles):
    """Static files (logo, tab icon) cached by browsers for a week."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=604800"
        return response


app = FastAPI(title=APP_NAME, lifespan=lifespan)
# HTML compresses about 6-8x; pages are mostly repeated markup.
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", CachedStatic(directory=STATIC_DIR), name="static")


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
def landing(request: Request, session: Session = Depends(get_session)):
    return templates.TemplateResponse(request, "landing.html", {"lens": _attribution_lens(session)})


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


def _ran_on_real_models(session: Session, exp: Experiment) -> bool:
    """Whether a run used real models. Offline runs leave "offline:" judge models on their scores."""
    cache = done_cache(session, "real_models")
    if exp.id in cache:
        return cache[exp.id]
    judge = session.scalar(
        select(ConversationScore.judge_model)
        .join(Conversation, Conversation.id == ConversationScore.conversation_id)
        .where(Conversation.experiment_id == exp.id)
        .limit(1)
    )
    real = judge is not None and not judge.startswith("offline:")
    if exp.status.value == "done":
        cache[exp.id] = real  # a finished run never changes
    return real


def _latest_done(session: Session, rows, key: str, guardrails: bool, *, real_only=False, need_holdout=True):
    """The newest finished run of one agent setup, as (experiment, summary), or None."""
    for e, s in sorted(rows, key=lambda r: r[0].id, reverse=True):
        if e.agent_key != key or e.guardrails != guardrails or e.status.value != "done":
            continue
        if need_holdout and not e.holdout_pct:
            continue
        if real_only and not _ran_on_real_models(session, e):
            continue
        return e, s
    return None


@dataclass
class CompareRow:
    title: str  # "Baseline", "Pushy discounter", or "" for a bare model row
    model: str  # "Haiku", "Gemini Flash"
    price_in: float  # list price per 1M tokens (USD)
    price_out: float
    experiment: Experiment | None = None
    summary: object = None
    real: bool = False  # ran on real models (offline runs are scripted)
    status: str = "none"  # "measured" | "running" | "none"

    @property
    def lift_per_dollar(self):
        # Offline token counts are estimates, so a cost-based ratio only means something for real runs.
        return self.summary.lift_per_dollar if self.summary and self.real else None


def _price(key: str) -> tuple[float, float]:
    return PRICES_PER_MTOK[AGENTS[key][1]]


def _model_name(key: str) -> str:
    return AGENTS[key][0].split(" · ", 1)[-1]


def _row(session: Session, title: str, key: str, found) -> CompareRow:
    exp, summary = found or (None, None)
    real = bool(exp) and _ran_on_real_models(session, exp)
    return CompareRow(title, _model_name(key), *_price(key), exp, summary, real, "measured" if exp else "none")


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[int]:
    """Round axis ticks (cents) inside [lo, hi]; 0 is always one of them when in range."""
    raw = (hi - lo) / count or 1
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    return [round(first + i * step) for i in range(int((hi - first) // step) + 1)]


def _axis(rows: list[CompareRow]) -> dict:
    """A shared lift axis (cents, always including $0) for one group of rows."""
    shown = [r for r in rows if r.summary and r.summary.estimate]
    values = [0]
    for r in shown:
        est = r.summary.estimate
        values += [est.ci_low, est.ci_high, est.total_lift]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.04 or 1
    ticks = _nice_ticks(lo - pad, hi + pad) if shown else []
    return {"range": (lo - pad, hi + pad), "ticks": ticks, "shown": len(shown)}


STRATEGY_NAMES = {("baseline-haiku", False): "Baseline", ("pushy-haiku", False): "Pushy",
                  ("pushy-haiku", True): "Pushy + guardrails"}


def _strategy_batch(session: Session, rows, *, need_holdout: bool) -> dict:
    """The three lab agents from one batch (same sample size and holdout), so they compare
    like with like. Picks the batch with the most of the three agents, newest first."""
    best: dict = {}
    for setup in {(e.sample_size, e.holdout_pct) for e, _ in rows}:
        same = [(e, s) for e, s in rows if (e.sample_size, e.holdout_pct) == setup]
        found = {(k, g): f for k, g in COMPARISON
                 if (f := _latest_done(session, same, k, g, need_holdout=need_holdout))}
        def rank(batch):
            return len(batch), max((e.id for e, _ in batch.values()), default=0)
        if found and rank(found) > rank(best):
            best = found
    return best


def _model_comparison(session: Session, rows) -> dict:
    """The model comparison on the Experiments page, from stored results only, in two groups
    that are each one batch on one axis: lift depends on how many shoppers were messaged, so
    runs of different sizes never share an axis.

    Strategies: the lab's three Haiku agents from one batch, marked when offline.
    Models: the same baseline prompt per model, measured only from finished runs on real
    models. Offline runs answer from one script whatever the model, so showing them as a
    model's score would be invented.
    """
    batch = _strategy_batch(session, rows, need_holdout=True)
    baseline = _row(session, "Baseline", "baseline-haiku", batch.get(("baseline-haiku", False)))
    pushy = _row(session, "Pushy discounter", "pushy-haiku", batch.get(("pushy-haiku", False)))
    guarded = _row(session, "Pushy discounter", "pushy-haiku", batch.get(("pushy-haiku", True)))
    models = []
    for key in MODEL_COMPARISON:
        row = _row(session, "", key, _latest_done(session, rows, key, False, real_only=True))
        busy = any(e.agent_key == key and e.status.value in ("queued", "running") for e, _ in rows)
        if row.status == "none" and busy:
            row.status = "running"
        models.append(row)
    models.sort(key=lambda r: (r.price_in, r.price_out))  # stable: equal prices keep MODEL_COMPARISON order
    strategies = _axis([baseline, pushy, guarded])
    model_axis = _axis(models)
    first = baseline.summary or pushy.summary or guarded.summary
    return {
        "baseline": baseline, "pushy": pushy, "guarded": guarded, "models": models,
        "strategy_axis": strategies, "model_axis": model_axis,
        "strategy_size": (first.conversations, first.holdout_size) if first else None,
        "runs": strategies["shown"] + model_axis["shown"],
    }


def _attribution_lens(session: Session) -> dict | None:
    """Claimed vs true lift for the three lab agents (one batch) on the latest simulated store."""
    store = latest_store(session)
    if store is None:
        return None
    # Only the lab agents' finished runs: the lens never needs the other experiments.
    exps = list(session.scalars(select(Experiment).where(
        Experiment.store_id == store.id,
        Experiment.agent_key.in_({k for k, _ in COMPARISON}),
        Experiment.status == ExperimentStatus.done,
    )))
    rows = [(e, summarize(session, e)) for e in exps]
    batch = _strategy_batch(session, rows, need_holdout=False)
    if not batch:
        return None
    agents = [{"name": STRATEGY_NAMES[k], "id": e.id, "claimed": s.claimed_revenue_cents,
               "truth": s.caused_revenue_cents} for k, (e, s) in sorted(batch.items(), key=lambda kv: kv[1][0].id)]
    top = max(max(a["claimed"], a["truth"]) for a in agents) or 1
    for a in agents:
        a["claimed_pct"] = round(max(a["claimed"], 0) / top * 100, 2)
        a["truth_pct"] = round(max(a["truth"], 0) / top * 100, 2)
    ids = sorted(a["id"] for a in agents)
    return {"agents": agents, "runs": f"run {ids[0]}" if len(ids) == 1 else f"run {ids[0]}–{ids[-1]}"}


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
    comparison = _model_comparison(session, rows)
    return templates.TemplateResponse(
        request,
        "experiments.html",
        {
            "store": store,
            "rows": rows,
            "scale": _range_scale(rows),
            "cmp": comparison,
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
def experiment_detail(exp_id: int, request: Request, page: int = 1, session: Session = Depends(get_session)):
    exp = session.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404)
    s = summarize(session, exp)
    total = s.conversations + s.holdout_size
    pages = max(1, -(-total // CONVERSATIONS_PER_PAGE))
    page = min(max(page, 1), pages)
    convs = list(
        session.scalars(
            select(Conversation)
            .where(Conversation.experiment_id == exp.id)
            .options(selectinload(Conversation.score))
            .order_by(Conversation.id)
            .offset((page - 1) * CONVERSATIONS_PER_PAGE)
            .limit(CONVERSATIONS_PER_PAGE)
        )
    )
    judge = session.scalar(
        select(ConversationScore.judge_model)
        .join(Conversation, Conversation.id == ConversationScore.conversation_id)
        .where(Conversation.experiment_id == exp.id)
        .limit(1)
    )
    return templates.TemplateResponse(
        request, "experiment.html",
        {"e": exp, "s": s, "convs": convs, "page": page, "pages": pages, "total": total,
         "first": (page - 1) * CONVERSATIONS_PER_PAGE + 1, "judge": judge},
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


def _score_run(exp: Experiment, by_id: dict) -> dict:
    """Rule-check every draft in a Live store run: {conversation id: DraftScore or None}."""
    out = {}
    for conv in exp.conversations:
        co = by_id.get(conv.checkout_id)
        text = conv.messages[0].text if conv.messages else ""
        out[conv.id] = score_draft(
            text,
            first_name=co.customer.first_name if co else "",
            products=[i.product.title for i in co.items] if co else [],
            link=co.recovery_url if co else None,
        ) if co else None
    return out


def _draft_scores(session: Session, store, by_id: dict, shadows: list[Experiment]):
    """Scores for the runs on the page, and the latest finished run of every agent, for comparing models."""
    scores, run_scores = {}, {}
    for e in shadows:
        scores.update(s := _score_run(e, by_id))
        run_scores[e.id] = summarize_scores(list(s.values()))
    by_model = []
    if store is not None:
        done = session.scalars(
            select(Experiment)
            .where(Experiment.store_id == store.id, Experiment.shadow, Experiment.status == ExperimentStatus.done)
            .options(selectinload(Experiment.conversations).selectinload(Conversation.messages))
            .order_by(Experiment.id.desc())
        )
        seen = set()
        for e in done:
            if (e.agent_key, e.guardrails) in seen or e.agent_key not in AGENTS:
                continue
            seen.add((e.agent_key, e.guardrails))
            summary = summarize_scores(list(_score_run(e, by_id).values()))
            if summary:
                drafts = [c for c in e.conversations if c.agent_cost_micros]
                cost = sum(c.agent_cost_micros for c in drafts) / len(drafts) / 1_000_000 if drafts else None
                by_model.append({"exp": e, "label": AGENTS[e.agent_key][0], "s": summary, "cost": cost})
        by_model.sort(key=lambda r: (-r["s"].avg_score, r["cost"] if r["cost"] is not None else 0))
    return scores, run_scores, by_model


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
    scores, run_scores, by_model = _draft_scores(session, store, by_id, shadows)
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
            "scores": scores,
            "run_scores": run_scores,
            "by_model": by_model,
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
