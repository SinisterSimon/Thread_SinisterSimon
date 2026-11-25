"""Bot shoppers that leave real checkouts on your Shopify *development* store.

Each bot opens the store, picks a product, starts checkout, types a contact email and
a shipping name, and leaves without paying: a real abandoned checkout. Shopify then
tells Thread (webhook), Thread syncs and drafts the recovery text.

Safety:
- Runs only against a *.myshopify.com store (your dev store), never a custom domain.
- Emails are @example.com, a domain reserved for testing: nothing is ever delivered.
- Never pays, never enters card details.

Setup (Mac, once):
    uv pip install playwright && playwright install chromium
Run:
    python -m bots.abandon_cart --count 3            # uses SHOPIFY_STORE from .env
    python -m bots.abandon_cart --count 1 --headed   # watch it in a browser window
The dev store's storefront password goes in SHOPIFY_STOREFRONT_PASSWORD
(Shopify admin > Online Store > Preferences > Password protection).

Checkout pages change over time; the selectors below try several known field names.
If a bot stops, run with --headed to see where.
"""
import argparse
import os
import random
import sys
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Bot:
    key: str
    first_name: str
    last_name: str
    fills_address: bool  # leaves after the contact step, or after typing an address too


BOTS = [
    Bot("browser", "Mia", "Tamm", False),
    Bot("comparer", "Oliver", "Saar", True),
    Bot("hesitant", "Sofia", "Mägi", True),
    Bot("busy", "Rasmus", "Kask", False),
]

EMAIL_FIELDS = ["input#email", "input[name='email']", "input[autocomplete='email']", "input[type='email']"]
FIRST_NAME_FIELDS = ["input[name='firstName']", "input[autocomplete='shipping given-name']", "#TextField0"]
LAST_NAME_FIELDS = ["input[name='lastName']", "input[autocomplete='shipping family-name']"]
ADDRESS_FIELDS = ["input[name='address1']", "input[autocomplete='shipping address-line1']"]
CITY_FIELDS = ["input[name='city']", "input[autocomplete='shipping address-level2']"]
ZIP_FIELDS = ["input[name='postalCode']", "input[autocomplete='shipping postal-code']"]


def _fill(page, selectors: list[str], value: str) -> bool:
    for sel in selectors:
        field = page.locator(sel).first
        if field.count() and field.is_visible():
            field.fill(value)
            field.press("Tab")  # Shopify saves the field when focus leaves it
            return True
    return False


def run_bot(page, store: str, password: str, bot: Bot, n: int, rng: random.Random) -> str:
    base = f"https://{store}"
    page.goto(base, wait_until="domcontentloaded")
    if "/password" in page.url:
        if not password:
            raise SystemExit("The storefront is password-protected: set SHOPIFY_STOREFRONT_PASSWORD.")
        page.fill("input[type='password']", password)
        page.press("input[type='password']", "Enter")
        page.wait_for_load_state("domcontentloaded")

    products = page.request.get(f"{base}/products.json?limit=50").json()["products"]
    in_stock = [p for p in products if any(v.get("available", True) for v in p["variants"])]
    product = rng.choice(in_stock or products)
    variant = next((v for v in product["variants"] if v.get("available", True)), product["variants"][0])

    # A cart permalink opens checkout with this item already in the cart.
    page.goto(f"{base}/cart/{variant['id']}:1", wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    email = f"thread.bot+{bot.key}{n}@example.com"
    if not _fill(page, EMAIL_FIELDS, email):
        raise RuntimeError(f"no email field found on {page.url}")
    _fill(page, FIRST_NAME_FIELDS, bot.first_name)
    _fill(page, LAST_NAME_FIELDS, bot.last_name)
    if bot.fills_address:
        _fill(page, ADDRESS_FIELDS, "Narva mnt 5")
        _fill(page, CITY_FIELDS, "Tallinn")
        _fill(page, ZIP_FIELDS, "10117")
    page.wait_for_timeout(2500)  # give Shopify time to save the checkout, then walk away
    return f"{bot.first_name} ({bot.key}) left {product['title']} in checkout as {email}"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Leave abandoned checkouts on your Shopify dev store.")
    parser.add_argument("--count", type=int, default=3, help="how many bots (max 10)")
    parser.add_argument("--store", default=os.getenv("SHOPIFY_STORE", ""), help="xxx.myshopify.com")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    store = args.store.strip().removeprefix("https://").rstrip("/")
    if not store.endswith(".myshopify.com"):
        print("Refusing: bots only run against a development store (*.myshopify.com).")
        return 2
    from playwright.sync_api import sync_playwright  # dev-only dependency

    rng = random.Random(args.seed)
    password = os.getenv("SHOPIFY_STOREFRONT_PASSWORD", "")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        for n in range(1, min(args.count, 10) + 1):
            bot = BOTS[(n - 1) % len(BOTS)]
            context = browser.new_context()  # a fresh shopper each time: own cookies, own cart
            try:
                print(run_bot(context.new_page(), store, password, bot, n, rng))
            except Exception as err:  # keep going with the next bot
                print(f"{bot.first_name} ({bot.key}) stopped: {err}")
            finally:
                context.close()
        browser.close()
    print("Done. Thread syncs within seconds if the webhook is registered; otherwise click Sync on the Live store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
