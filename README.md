# Thread

**A conversational cart-recovery platform that measures what it actually causes.**

Thread recovers abandoned checkouts with AI-written, two-way text conversations, and
uses a randomized holdout group to show how much of the "recovered" revenue is real
lift rather than shoppers who would have come back anyway.

> Status: **Phase 1 of 5** — data model, store simulator and dashboard.

## Why

Roughly 7 in 10 online carts are abandoned, and recovery tools usually report every
purchase that follows a message as "recovered." Some of those shoppers were coming
back regardless. Thread treats that as the core product question: every claim on
the dashboard should be backed by a comparison against a group that got no message.

Because real store data is hard to get, Thread ships with a **simulator** that
generates realistic traffic *with ground truth*: it knows which abandoned shoppers
would return on their own. That lets us check whether the measurement method finds
the true effect.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Data model, store simulator, dashboard, CI, deploy | ✅ |
| 2 | Recovery engine: scheduled follow-ups, LLM-drafted messages, draft ranking | ⏳ |
| 3 | Two-way conversations: AI replies from store facts, human handoff rules, transcripts | ⏳ |
| 4 | Experiments: randomized holdout, A/B message styles, incremental lift dashboard | ⏳ |
| 5 | Polish: architecture write-up, demo video, cost-per-conversation view | ⏳ |

## Architecture (target)

```
Store events ──▶ Backend (FastAPI + Postgres) ──▶ Conversation engine (LLM drafts → ranker)
                       ▲                                   │
   Merchant dashboard ─┘                                   ▼
                       ◀── outcomes (replies, orders) ── Messaging (SMS / iMessage / WhatsApp, simulated)
```

## Data model

- `stores`, `products`, `customers` — the merchant's catalog and audience
- `checkouts` + `checkout_items` — every checkout, its status (`completed` / `abandoned`) and its future experiment arm
- `orders` — purchases, tagged `same_session` or `returned`
- Fields prefixed `sim_` are simulator ground truth a real store would never see

Money is stored in integer cents.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000 and click **Simulate 30 days**. With no `DATABASE_URL`
set, the app uses a local SQLite file. To use Postgres locally:

```bash
docker compose up -d
cp .env.example .env
```

Run the tests:

```bash
pytest
```

## Deploy

1. Create a free Postgres database on [Neon](https://neon.tech) and copy its connection string.
2. On [Render](https://render.com), create a **Blueprint** from this repo (it reads `render.yaml`).
3. Set `DATABASE_URL` to the Neon string in the Render dashboard.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard |
| POST | `/simulate?seed=42&days=30` | Generate a new simulated store |
| GET | `/api/metrics` | Dashboard numbers as JSON |
| GET | `/health` | Health check |
