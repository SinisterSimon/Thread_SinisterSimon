# The bot-shopper loop

Bots leave real abandoned checkouts on your Shopify **development** store; Shopify tells
Thread straight away; Thread syncs and drafts the recovery text. Nothing is sent to anyone.

```
bot fills checkout ─▶ Shopify ─(signed webhook)─▶ Thread /webhooks/shopify ─▶ sync ─▶ draft
```

## One-time setup

1. **Webhook.** On your Mac, with `SHOPIFY_*` in `.env`:
   ```
   python -m scripts.register_webhooks https://thread-fszp.onrender.com/webhooks/shopify
   ```
   It subscribes to `CHECKOUTS_CREATE` and `CHECKOUTS_UPDATE` (needs the `read_orders`
   scope your app already has). Safe to run again.
2. **Auto-draft (optional).** In Render → Environment, set `AUTO_DRAFT_AGENT`, e.g.
   `baseline-haiku`. Empty means the webhook only syncs. Auto-drafts use the same daily cap
   as the Draft button, and at most one starts per 10 minutes.
3. **Bots.** Install the browser once:
   ```
   uv pip install playwright && playwright install chromium
   ```
   and put the storefront password in `.env` as `SHOPIFY_STOREFRONT_PASSWORD`
   (Shopify admin → Online Store → Preferences → Password protection).

## Run it

```
python -m bots.abandon_cart --count 3            # three different bot shoppers
python -m bots.abandon_cart --count 1 --headed   # watch one in a browser window
```

Then open the Live store page: the new checkouts are there, and with `AUTO_DRAFT_AGENT`
set, a draft run starts on its own.

## Safety

- Bots refuse any store that isn't `*.myshopify.com`, use `@example.com` emails (a domain
  reserved for testing, never delivered) and never pay.
- Thread only accepts webhooks signed with the app's client secret (HMAC-SHA256); a forged
  or altered delivery gets 401.
- On Render's free plan the site may be asleep when Shopify calls. Shopify retries failed
  deliveries later; the **Sync** button always works too.
- Checkout pages change over time. If a bot stops, run it with `--headed` to see where.
