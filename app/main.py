"""Web app: dashboard pages and a small JSON API."""
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import APP_NAME
from app.db import get_session, init_db
from app.metrics import latest_store, store_metrics
from app.simulator import SimConfig, run_simulation

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def money(cents: int) -> str:
    return f"${cents / 100:,.0f}"


templates.env.filters["money"] = money


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, session: Session = Depends(get_session)):
    store = latest_store(session)
    metrics = store_metrics(session, store) if store else None
    return templates.TemplateResponse(
        request, "dashboard.html", {"app_name": APP_NAME, "m": metrics}
    )


@app.post("/simulate")
def simulate(seed: int = 42, days: int = 30, session: Session = Depends(get_session)):
    run_simulation(session, SimConfig(seed=seed, days=days))
    return RedirectResponse("/", status_code=303)


@app.get("/api/metrics")
def api_metrics(session: Session = Depends(get_session)) -> dict:
    store = latest_store(session)
    return asdict(store_metrics(session, store)) if store else {}
