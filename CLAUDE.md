# Thread: notes for AI coding assistants

A test lab for AI sales agents. Read `README.md` for what it does and `docs/findings.md` for results.

## Commands

- Run: `uvicorn app.main:app --reload` (http://localhost:8000)
- Test: `pytest` (all offline, no API key needed)
- Lint: `ruff check .` (CI runs both on every push)
- New migration: `alembic revision --autogenerate -m "what changed"`, then review the file by hand

## Layout

- `app/simulator.py`: synthetic store; `sim_` fields are ground truth the measurement must never read
- `app/shoppers/`: personas, shopper bot, outcome model
- `app/agents/`: agent contract, registry, guardrails
- `app/llm/`: the only place that talks to model providers (offline + Anthropic)
- `app/evals/`: rule checks + AI judge
- `app/experiments/`: runner, holdout assignment, lift statistics
- `migrations/versions/`: Alembic; applied at startup by `app/migrate.py`
- `worker/run.py`: background worker

## Rules

- Never import a provider SDK outside `app/llm/`. Call `traced(...)` so calls show in Langfuse.
- Lift estimates use only observable data (revenue, arm). Ground truth (`sim_*`, `would_buy_anyway`, `baseline_revenue_cents`) is for checking estimates, never for producing them.
- Keep common random numbers: outcome randomness is seeded by checkout id only, never by agent.
- Money is integer cents.
- Every schema change gets a migration, and `tests/test_phase3.py::test_migrations_match_the_models` must pass.
- Offline mode must stay free and deterministic; tests never call a real API.
- Work on a branch per change and open a pull request into `main`; Render deploys `main`.
