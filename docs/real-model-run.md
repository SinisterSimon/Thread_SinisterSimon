# Running the real model comparison

The lab runs **offline** by default: every model answers from the same script, so
offline runs can't say anything about a model. This is how to measure the six models
for real, about $5 in total, on your own Mac. The live site stays offline, so nobody
can spend your credit by clicking a button.

## 1. Before you start

- `ANTHROPIC_API_KEY` in `.env` (credit bought, spend limit set).
- `GEMINI_API_KEY` in `.env`. Gemini Argon needs billing turned on for that key;
  without it, the Argon row fails with "turn on billing" and the other five still run.
- Latest code: `git checkout main && git pull && uv pip install -r requirements.txt`.

## 2. The main run (Claude judge)

1. In `.env`, set `LLM_MODE=live`.
2. Start the app: `uvicorn app.main:app --reload`.
3. Open http://localhost:8000/experiments and click **Compare 6 models**.
   Each model messages ~50 simulated shoppers; a 50% holdout gets nothing.
4. Wait until all six runs say `done` (the page refreshes itself; a few minutes).
5. Screenshot the **Model comparison** and the leaderboard.

Shoppers are played by Claude Haiku and the judge is Claude Sonnet (`JUDGE_MODEL`).

## 3. The fairness check (Gemini judge)

A Claude judge might prefer Claude-style replies. Run the same comparison once more
with a Gemini judge and see whether the ranking holds.

1. In `.env`, add `JUDGE_MODEL=gemini-3.8-flash` (keep `LLM_MODE=live`).
2. Restart the app (Ctrl+C, then start it again), so the new judge is picked up.
3. Click **Compare 6 models** again. Each run page shows "judged by gemini-3.8-flash".
4. Screenshot again.

Lift doesn't depend on the judge (it comes from purchases), only the **quality**
score does. So compare the quality ranking between the two runs.

## 4. Afterwards

- Set `LLM_MODE=offline` again and remove `JUDGE_MODEL` (back to Claude Sonnet).
- Check the real spend in the Anthropic console (Usage) and Google AI Studio.
- Send the screenshots; the results go into `docs/findings.md` and the README.
