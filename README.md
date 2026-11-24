# Thread

**A test lab for AI sales agents: simulated shoppers, scored conversations, measured lift.**

Thread plays simulated shoppers with real personalities against pluggable AI sales agents,
scores every conversation, and measures each agent's true effect on sales — separating
real lift from shoppers who would have bought anyway.

> Status: **Phase 4 of 5** — holdout groups estimate real lift the way a real store must, checked against the truth.

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
| 2 | Shopper personas, agent interface, baseline + comparison agents, experiment runs, worker | ✅ |
| 2.5 | n8n workflows around the API (Slack summaries, scheduled runs) | ⏳ |
| 3 | Conversation scoring (rules + AI judge), guardrails, Langfuse tracing, Alembic migrations | ✅ |
| 4 | Randomized holdout groups, estimated vs true lift, agent leaderboard | ✅ |
| 5 | Polish: findings write-up, demo video, custom domain, optional real SMS | ⏳ |

## How an experiment works

1. **Sample:** take the store's abandoned checkouts whose shopper can be texted, in a fixed order, so every agent meets the same shoppers.
2. **Persona:** each checkout is played by one of five shopper types — price-sensitive, unsure about fit, worried about delivery, just browsing, or frustrated. Each has an objection (`app/shoppers/personas.py`).
3. **Conversation:** the agent texts first, the shopper replies, up to three rounds. The agent can end the chat (`[END]`) or hand off to a human (`[HANDOFF]`).
4. **Outcome:** a small, documented model decides whether the shopper bought (`app/shoppers/outcome.py`). Shoppers the simulator marked as returning anyway always buy — the agent gets no real credit for them.
5. **Summary:** claimed revenue (every purchase after a message) vs revenue that would have come anyway vs revenue the agent actually caused.

Every agent uses the same random draw per checkout ("common random numbers"), so two agents that behave the same get the same results and any difference is real.

### Agents

| Key | Strategy | Model |
|---|---|---|
| `baseline-haiku` | Answers the real objection, max 10% off for price only, hands off upset shoppers | Claude Haiku |
| `baseline-sonnet` | Same rules, bigger model — is it worth the cost? | Claude Sonnet |
| `pushy-haiku` | 20% off every message, never hands off — the strategy to beat | Claude Haiku |

Add an agent by adding an entry to `app/agents/registry.py`.

## Measuring lift with a holdout group

A real store never knows which shoppers would have bought anyway. What it can do is randomly keep a share of abandoned checkouts message-free (the **holdout**) and compare revenue per shopper between the two groups (`app/experiments/stats.py`):

```
lift per shopper     = mean revenue (messaged) − mean revenue (holdout)
estimated total lift = lift per shopper × messaged shoppers
95% range            = difference ± 1.96 × Welch standard error, scaled the same way
```

Assignment is random but stable per checkout, so every agent run with the same holdout share is compared on the same split. Because the simulator knows the truth, each experiment page shows three numbers side by side: what's **claimed** (every sale after a message), what the holdout **estimates**, and the **true lift**.

**True lift** = what messaged shoppers paid − what they would have paid with no message. Shoppers coming back anyway would have paid full price, so a discount given to them counts against the agent.

### Findings (offline, default seed, 50% holdout)

| Store | Agent | Claimed | Estimated (95% range) | True | Claimed ÷ true |
|---|---|---|---|---|---|
| 30 days, 273 textable | Baseline | $3,153 | $1,477 ($71 to $2,882) | $1,859 | 1.7× |
| 30 days, 273 textable | Pushy | $2,021 | $345 (−$884 to $1,573) | $727 | 2.8× |
| 180 days, 1,587 textable | Baseline | $24,178 | $13,873 ($10,171 to $17,576) | $11,157 | 2.2× |
| 180 days, 1,587 textable | Pushy | $16,500 | $6,195 ($2,957 to $9,433) | $3,479 | 4.7× |

- **The holdout estimate works**: the 95% range contained the true lift in every run.
- **Claimed revenue overstates the effect by roughly 2–5×**, and more for agents that discount people who were buying anyway.
- **Small stores can't measure small effects.** With 273 textable shoppers, the pushy agent's range includes $0: a real store that size could not tell whether its messages helped at all. Sample size, not cleverness, sets what you can know.

## Scoring every conversation

Each finished chat gets a quality score from 0 to 100 (`app/evals/scorer.py`):

- **Five rule checks** (60% of the score), exact and free: dealt with the shopper's real concern; handed off only when needed; discount within policy (max 10%, price concerns only); no pressure tactics; left "just browsing" shoppers alone.
- **An AI judge** (40%) rates helpfulness, accuracy and tone from 1 to 5. It runs on `JUDGE_MODEL` (Sonnet by default), a different model from the Haiku agents, because models tend to rate writing like their own more kindly. If its reply can't be parsed, the rule-based estimate is used and the note says so.

A **policy break** is a chat that fails the handoff, discount or pressure rule.

## Guardrails

Tick **Guardrails on** when starting an experiment to put the agent behind checks that run outside the model (`app/agents/guardrails.py`): discounts capped at 10% and offered once, pressure sentences removed, and a forced handoff when a shopper asks for a person or reports a billing problem. Every change counts as a fix.

Finding so far (offline): guardrails took the pushy agent from 150 policy breaks to 0 and its quality from 38 to 81, but it caused *less* revenue ($587 vs $1,312), because its stripped-down messages still don't answer anyone's question. Guardrails make a bad agent safe, not good.

## Tracing

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` and every conversation becomes a Langfuse trace: each agent, shopper and judge call with prompt, reply, model and tokens, plus the quality score. Without keys, tracing is off and nothing else changes (`app/tracing.py`).

## Database migrations

Schema changes are Alembic migrations in `migrations/versions/`, applied automatically when the app or worker starts (`app/migrate.py`). Databases created before Alembic was added are detected and stamped at the baseline revision first, so only newer migrations run. Tests check that the migrations match the models exactly.

### Offline vs real models

`LLM_MODE=offline` (default) uses scripted replies: free, deterministic, and what the tests use.
`LLM_MODE=anthropic` with `ANTHROPIC_API_KEY` set uses real Claude models for both the agents and the shoppers.
All model calls go through `app/llm/`, so adding OpenAI later means one new class.

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
- `experiments` — one agent run over a sample of checkouts, with status
- `conversations` + `conversation_messages` — each chat, its transcript and outcome (kept apart from `orders` so repeated runs never double-count revenue)
- `conversation_scores` — rule results, judge ratings and the 0-100 quality score per chat
- `conversations.arm` (`treatment` / `holdout`) and `baseline_revenue_cents` (what the shopper would have paid with no message) power the lift estimate
- Fields prefixed `sim_` are simulator ground truth a real store would never see

Money is stored in integer cents.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000, pick a period and click **Simulate a new store**, then go to **Experiments** and run each agent with a holdout share. With no `DATABASE_URL`
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
| POST | `/simulate` (form: `days`, `seed`) | Generate a new simulated store |
| GET | `/api/metrics` | Dashboard numbers as JSON |
| GET | `/experiments` | Run agents, compare results |
| POST | `/experiments` | Start a run (`agent_key`, `sample_size`, `holdout_pct`, `guardrails`) |
| GET | `/experiments/{id}` | One run's results and conversations |
| GET | `/conversations/{id}` | Full transcript and outcome |
| GET | `/api/experiments/{id}` | One run's summary as JSON (for n8n) |
| GET | `/health` | Health check |

## Worker

By default the web app runs experiments itself. With a paid Render worker, set
`WORKER_ENABLED=1` and run `python -m worker.run` as a separate service; it claims
queued experiments safely (`SELECT … FOR UPDATE SKIP LOCKED` on Postgres).
Locally: `python -m worker.run --once`.
