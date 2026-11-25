# Findings

**Counting every sale after a message overstated what the messages caused by 1.6–4.7×, and a randomized holdout group recovered the real effect.** A polite agent that answers the shopper's actual question caused about three times the real revenue of one that offers 20% off to everyone. Guardrails stopped the bad agent from breaking policy but did not make it effective.

All results come from Thread's offline mode: simulated shoppers and scripted agents, so every number is reproducible with `pytest` and the seeds below. Real models (`LLM_MODE=live`) will change the scores; the measurement method stays the same.

## Setup

- **Stores:** five simulated 180-day stores (seeds 42, 1, 2, 3, 4), each with about 1,500 abandoned checkouts whose shoppers gave a phone number and consented to texts.
- **Split:** each checkout is assigned at random to be messaged or held out (50% each). The assignment is stable per checkout, so every agent meets the same shoppers.
- **Shoppers:** five types, each with a hidden reason for leaving: price-sensitive, unsure about fit, worried about delivery, just browsing, frustrated. About 15% of abandoned shoppers come back and buy on their own, message or not.
- **Agents:**
  - *Baseline*: answers the real objection, offers at most 10% off and only for price concerns, hands frustrated shoppers to a person, leaves browsers alone.
  - *Pushy*: 20% off in every message, urgency, never hands off.
  - *Pushy + guardrails*: the same agent behind checks that cap discounts at 10%, allow one offer, remove pressure sentences and force a handoff when asked.
- **Metrics:**
  - *Claimed revenue*: every purchase by a messaged shopper. What a recovery tool typically reports.
  - *True lift*: what messaged shoppers paid minus what they would have paid with no message. Known only because the simulator records it.
  - *Estimated lift*: (mean revenue per messaged shopper − mean revenue per held-out shopper) × messaged shoppers, with a 95% Welch interval. What a real store can measure.
  - *Quality*: 0–100 per conversation, 60% from five rule checks and 40% from an AI judge's ratings.

## Results

| Store seed | Agent | Messaged / held out | Claimed | True lift | Estimated (95% range) | Range contains truth | Claimed ÷ true |
|---|---|---|---|---|---|---|---|
| 42 | Baseline | 786 / 801 | $24,178 | $11,157 | $13,873 ($10,171 to $17,576) | yes | 2.2× |
| 42 | Pushy | 786 / 801 | $16,500 | $3,479 | $6,195 ($2,957 to $9,433) | yes | 4.7× |
| 1 | Baseline | 760 / 740 | $19,749 | $12,339 | $8,993 ($5,556 to $12,429) | yes | 1.6× |
| 1 | Pushy | 760 / 740 | $11,559 | $4,149 | $802 (−$2,070 to $3,674) | **no** | 2.8× |
| 2 | Baseline | 720 / 749 | $20,092 | $12,299 | $11,512 ($8,079 to $14,946) | yes | 1.6× |
| 2 | Pushy | 720 / 749 | $11,886 | $4,093 | $3,306 ($501 to $6,111) | yes | 2.9× |
| 3 | Baseline | 873 / 906 | $25,280 | $14,022 | $13,891 ($10,098 to $17,685) | yes | 1.8× |
| 3 | Pushy | 873 / 906 | $15,246 | $3,988 | $3,857 ($738 to $6,976) | yes | 3.8× |
| 4 | Baseline | 797 / 800 | $21,189 | $11,288 | $10,356 ($6,842 to $13,871) | yes | 1.9× |
| 4 | Pushy | 797 / 800 | $12,597 | $2,696 | $1,764 (−$1,156 to $4,684) | yes | 4.7× |

Guardrails on the pushy agent, same stores:

| Store seed | True lift without → with guardrails | Policy breaks without → with | Quality without → with |
|---|---|---|---|
| 42 | $3,479 → $2,350 | 786 → 0 | 39.0 → 80.2 |
| 1 | $4,149 → $2,062 | 760 → 0 | 38.4 → 80.2 |
| 2 | $4,093 → $2,241 | 720 → 0 | 38.4 → 80.3 |
| 3 | $3,988 → $3,461 | 873 → 0 | 38.4 → 79.9 |
| 4 | $2,696 → $1,374 | 797 → 0 | 38.5 → 79.6 |

The baseline scored 100 on quality with zero policy breaks in every store; offline, its scripted replies are written to pass the checks.

## What this shows

1. **Claimed revenue is the wrong number.** It overstated the real effect in every run, by 1.6–2.2× for the good agent and 2.8–4.7× for the pushy one. The gap is bigger for agents that discount shoppers who were coming back anyway: those sales count as "recovered" while the discount is pure loss.
2. **A holdout group recovers the real effect.** The 95% range contained the true lift in 9 of 10 runs. A 95% range is expected to miss about 1 time in 20, so one miss in ten is consistent with the method working; seed 1's pushy run is that miss.
3. **Answering the actual question beats discounting.** The pushy agent caused 66–76% less real lift than the baseline in every store, despite offering twice the discount.
4. **Guardrails make an agent safe, not good.** They removed every policy break and roughly doubled quality scores, but real lift fell further in all five stores. Stripping pressure and repeat discounts left messages that still didn't address anyone's concern.
5. **Small stores can't measure small effects.** On a 30-day store (273 textable shoppers), the pushy agent's 95% range ran from −$884 to $1,573: a store that size could not tell whether its messages helped at all. Sample size, not cleverness, sets what can be known.

## Limitations

- **Simulated shoppers.** Personas, purchase probabilities and the 15% natural return rate are assumptions (`app/shoppers/`, `app/simulator.py`). The point is not the exact numbers but that the measurement method finds whatever the truth is.
- **Scripted agents in offline mode.** Real models will be less perfect than the baseline script and less consistent than the pushy one.
- **The outcome model rewards keywords** that show an objection was addressed. A real shopper responds to meaning, not keywords; with real models the AI judge's ratings carry more weight.
- **One holdout per experiment.** Repeated looks at the same experiment, or many experiments chosen after the fact, would need corrections this lab doesn't yet apply.

## Reproduce

```bash
pytest tests/test_phase4.py -k recovers
```

Or run the demo on the live site: it simulates a 180-day store (seed 42) and runs all three agents with a 50% holdout.
