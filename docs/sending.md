# Sending real texts, safely

Thread drafts in shadow mode by default: nothing leaves it. This adds one narrow way out,
built for testing on **your own phone**, never for messaging shoppers.

## The rules (in `app/channels.py`, every send goes through them)

| Rule | Setting |
|---|---|
| Only numbers on the allowlist can receive anything | `SEND_ALLOWLIST=+372…` |
| A number that replied STOP gets nothing until it replies START | automatic |
| At most N texts in 24 hours | `SEND_PER_DAY=10` |
| Every text ends with "Reply STOP to opt out." | automatic |
| Every attempt is logged, blocked ones too | Live store → Send log |
| Console mode logs and sends nothing | `SEND_MODE=console` (default) |

"Send to my phone" on a Live store draft sends that draft to `SEND_TEST_TO`, which must be
on the allowlist. It never sends to the shopper in the checkout.

## Try it without Twilio

Set `SEND_ALLOWLIST` and `SEND_TEST_TO` to your number, keep `SEND_MODE=console`, and click
**Log a test send** under a draft. The Send log shows what would have gone out.

## With Twilio (when you're ready)

1. Twilio trial: verify your own number (trial accounts can only text verified numbers).
2. Buy or use the trial number; set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`.
3. Set `SEND_MODE=twilio`. The button now says **Send to my phone**.
4. For STOP replies: in Twilio, set the number's "A message comes in" webhook to
   `https://thread-fszp.onrender.com/webhooks/twilio`, and set `PUBLIC_BASE_URL` to
   `https://thread-fszp.onrender.com`. Thread only accepts requests signed with your auth token.

About $0.10 per text segment to Estonia; the daily cap keeps a mistake cheap.
