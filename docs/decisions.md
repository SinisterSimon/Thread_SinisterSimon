# Decisions

Why Thread is built the way it is. Each entry gives the decision, the reason, what was
rejected, and the question an interviewer is likely to ask about it. The README has the short
list; this file has the reasoning.

## Measurement

### 1. Measure lift with a randomized holdout, not attributed sales
- **Decision:** a share of shoppers (50% by default) gets no message. Lift is the difference in
  revenue per shopper between the messaged and held-out groups, times the messaged count, with a
  95% Welch interval.
- **Why:** "every sale after a message" counts people who would have bought anyway. In the lab
  that overstated the real effect by 1.6–4.7×.
- **Rejected:** last-touch attribution (what most recovery tools report); comparing to last
  month (seasonality and traffic changes swamp the effect).
- **Expect:** *"Isn't holding back half your shoppers expensive?"* Yes, during the test. Once the
  answer is clear the holdout can shrink to 5–10% to keep monitoring. Without a holdout you
  can't know whether the tool earns its fee.

### 2. A simulator with ground truth, which the estimate may never read
- **Decision:** each simulated shopper records whether they would have bought with no message
  (`would_buy_anyway`, `baseline_revenue_cents`, `sim_*` fields). Only checks use it; the lift
  estimate uses observable data only (revenue and arm).
- **Why:** a real store never knows the truth, so a real store can't test whether its
  measurement works. The simulator can: the 95% range contained the true lift in 9 of 10 runs,
  which is what a working method should do.
- **Rejected:** testing only on real data (no way to check the answer).
- **Expect:** *"How do you know the simulator is realistic?"* It isn't fully, and that's the
  point. It tests the measurement method, not the market. The shopper reactions are simple
  rules, which is also why lift ties across capable models (see 9).

### 3. Common random numbers
- **Decision:** outcome randomness is seeded by checkout id only, never by agent.
- **Why:** two agents that behave the same get the same result, so every difference comes from
  behaviour, not luck. Comparisons need far fewer shoppers.
- **Expect:** *"Why not just use more shoppers?"* Real models cost money per shopper; reducing
  noise for free is better than paying for it.

### 4. Hold every comparison on the same shoppers
- **Decision:** the holdout assignment is stable per checkout, and comparisons only put runs side
  by side when they share sample size and holdout.
- **Why:** an early version of the chart compared a 100-shopper run with a 1,587-shopper run,
  which made the bigger run look better simply because it was bigger. Fixed by matching batches.

### 5. Small stores can't measure small effects, and the app says so
- **Decision:** results whose 95% range includes $0 are labelled "Could be $0" instead of ranked
  as wins.
- **Why:** on a 273-shopper store the range ran from −$884 to $1,573. Honest uncertainty is a
  feature, not a bug.

## Agents and models

### 6. Guardrails outside the model
- **Decision:** code checks every message after the model writes it: discount cap (10%), one
  offer, no pressure sentences, forced handoff when asked, link added by Thread.
- **Why:** a prompt is a request; a guardrail is a guarantee. Haiku broke policy 8 times in 47
  conversations with the same prompt that Sonnet followed perfectly.
- **Finding:** guardrails made the pushy agent safe (0 breaks) but not effective (lift fell).
  Safety and quality are separate problems.

### 7. The checkout link is added by code, never written by the model
- **Why:** a model can invent or mistype a URL. Thread appends Shopify's real recovery link, and
  draft scoring fails any message containing another link.

### 8. One LLM layer, many providers, with an offline mode
- **Decision:** everything goes through `app/llm/`. Model names route by prefix (`claude-`,
  `gemini-`, `gpt-`). An offline mode returns scripted replies.
- **Why:** tests and CI cost nothing and are deterministic; adding a provider is one class;
  every call is priced and traced.
- **Story:** every real run first failed with a 400 asking to "send thinking between tools".
  Some models need a different thinking mode, so the client now learns the right mode per model
  from the error and retries.

### 9. Compare models on quality, rule breaks and cost, not lift
- **Decision:** the model table shows quality (0–100), rule breaks and cost per message.
- **Why:** all eight real models reached nearly the same true lift ($574–$590). The simulated
  shopper buys when the agent addresses the concern and makes the right call, and every capable
  model clears that bar. A lift chart that shows eight identical bars says nothing.
- **Expect:** *"So which model would you ship?"* Sonnet: best quality (97.4) and 0 breaks. GPT
  Sol is the runner-up at similar cost. Haiku is cheapest but broke rules 8 times, so only
  behind guardrails.

### 10. LLM-as-judge plus rule checks
- **Decision:** quality = 60% rule checks + 40% AI judge (helpfulness, accuracy, tone). The judge
  model is shown on every run.
- **Why:** rules catch hard failures reliably; a judge catches tone and helpfulness that rules
  can't. Using both limits the judge's bias.
- **Known weakness:** Claude Sonnet judged its own family's messages. A fair next step is
  rotating judges across providers.

### 11. A run survives a few provider failures, but not many
- **Decision:** providers retry for about a minute (2, 5, 15, 30 s). A conversation that still
  fails is skipped and counted. More than 5% skipped stops the run.
- **Why:** one busy moment shouldn't throw away a paid run, but skipped conversations aren't
  random (long, tricky ones fail more), so many skips would bias the result.
- **Story:** Gemini Flash on Google's free tier returned 503 "busy" for nearly every call. The
  rule correctly stopped the run instead of reporting numbers from a handful of survivors.

## Real data and sending

### 12. Real stores start in shadow mode
- **Decision:** with a real Shopify store, Thread reads abandoned checkouts with read-only scopes
  and drafts messages without sending. Real customers are never marked as consenting.
- **Why:** you can judge real drafts at near-zero cost and risk before anything reaches a
  customer.

### 13. Sending is locked down by default
- **Decision:** `SEND_MODE=console` by default; real sends need an allowlist, respect STOP/START
  opt-outs, have a daily cap and add "Reply STOP to opt out". Twilio and Shopify webhooks are
  signature-verified.
- **Why:** a recovery message is marketing. A bug must never be able to text a stranger.

### 14. Two model switches
- **Decision:** `LLM_MODE` controls the lab (one click can be thousands of calls);
  `DRAFT_LLM_MODE` controls live-store drafts (one call per checkout, capped per day).
- **Why:** the public site can show real drafts without anyone being able to run up a bill by
  clicking "Compare models".
- **Story:** Render's blueprint sync once reset `DRAFT_LLM_MODE` back to offline. The setting is
  now `sync: false` so it lives only in Render.

## Engineering

### 15. Money in integer cents
No floating-point rounding errors in revenue.

### 16. Every schema change gets a migration, checked by a test
Migrations run at startup, and a test fails if the models and migrations ever disagree.

### 17. Fast pages without a frontend framework
Server-rendered Jinja, gzip, week-long static caching, cached summaries for finished runs
(they never change), paging at 50 conversations. No React needed for a results site.

### 18. Plain labels over jargon
"Extra sales (estimate)" instead of "Estimated lift (95%)". If the author has to stop and decode
a label, a merchant will too.

## Two-minute version

> Recovery tools claim credit for every sale after a message. I built a lab to test that: a
> simulated store where I know who would have bought anyway, AI sales agents, and a randomized
> no-message group. Attributed sales overstated the real effect 1.6–4.7×, and the holdout
> estimate caught the truth 9 times out of 10. Then I ran eight real models from Anthropic and
> OpenAI on the same 100 shoppers for about $3.60. Lift tied, because they all made the right
> call, so the difference was quality and rule-following: Sonnet was best, Haiku cheapest but
> broke rules 8 times. That's why the guardrails live in code, not the prompt. It also connects
> to a real Shopify store in shadow mode, with locked-down SMS sending for when it goes live.
