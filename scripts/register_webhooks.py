"""Subscribe Thread to the dev store's checkout events.

Usage (with SHOPIFY_* set in .env):
    python -m scripts.register_webhooks https://thread-fszp.onrender.com/webhooks/shopify

Safe to run again. Shopify then POSTs to that URL whenever a checkout is created or
updated, and Thread syncs (and, if AUTO_DRAFT_AGENT is set, drafts) right away.
"""
import sys

from dotenv import load_dotenv

from app.shopify import client_from_env
from app.shopify.webhooks import register


def main() -> int:
    load_dotenv()
    if len(sys.argv) != 2 or not sys.argv[1].startswith("https://"):
        print("usage: python -m scripts.register_webhooks https://<your-site>/webhooks/shopify")
        return 2
    client = client_from_env()
    if client is None:
        print("Set SHOPIFY_STORE, SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET first.")
        return 1
    for line in register(client, sys.argv[1]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
