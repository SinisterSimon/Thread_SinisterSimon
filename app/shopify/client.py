"""A small read-only Shopify Admin API client.

Authentication uses the client credentials grant: the app's client ID and
secret are swapped for an access token that lasts 24 hours. This works because
the app (made in the Dev Dashboard) and the dev store belong to the same
Shopify organization, so no merchant login screen is needed.

Only GraphQL queries are sent. The app's scopes are read-only, so Thread could
not change the store even by mistake.
"""
import os
import time
from dataclasses import dataclass, field

import httpx

API_VERSION = "2026-10"


class ShopifyError(RuntimeError):
    """Something the person running Thread can act on; the message says what."""


@dataclass
class ShopifyClient:
    shop_domain: str  # e.g. thread-dev-yct4r2dk.myshopify.com
    client_id: str
    client_secret: str
    transport: httpx.BaseTransport | None = None  # tests pass a fake one
    _token: str | None = field(default=None, repr=False)
    _token_expires: float = 0.0

    def _http(self) -> httpx.Client:
        return httpx.Client(base_url=f"https://{self.shop_domain}", timeout=20, transport=self.transport)

    def _access_token(self) -> str:
        # Refresh a minute before the 24-hour token runs out.
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        with self._http() as http:
            resp = http.post(
                "/admin/oauth/access_token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
        if resp.status_code != 200:
            raise ShopifyError(
                f"Shopify refused the app's credentials ({resp.status_code}): {resp.text[:200]}. "
                "Check SHOPIFY_STORE, SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET, "
                "and that the app is installed on the store."
            )
        body = resp.json()
        self._token = body["access_token"]
        self._token_expires = time.time() + int(body.get("expires_in", 86399))
        return self._token

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        with self._http() as http:
            resp = http.post(
                f"/admin/api/{API_VERSION}/graphql.json",
                json={"query": query, "variables": variables or {}},
                headers={"X-Shopify-Access-Token": self._access_token()},
            )
        if resp.status_code != 200:
            raise ShopifyError(f"Shopify API error {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if body.get("errors"):
            text = "; ".join(str(e.get("message", e)) for e in body["errors"])
            if "ACCESS_DENIED" in str(body["errors"]) or "access" in text.lower():
                text += (
                    " — the app may need protected customer data access: Dev Dashboard → your app → "
                    "Settings → Access requests → Protected customer data."
                )
            raise ShopifyError(f"Shopify query failed: {text[:600]}")
        return body["data"]


def client_from_env() -> ShopifyClient | None:
    """The configured store, or None when the SHOPIFY_* settings are missing."""
    shop = os.getenv("SHOPIFY_STORE", "").strip()
    cid = os.getenv("SHOPIFY_CLIENT_ID", "").strip()
    secret = os.getenv("SHOPIFY_CLIENT_SECRET", "").strip()
    if not (shop and cid and secret):
        return None
    shop = shop.removeprefix("https://").rstrip("/")
    return ShopifyClient(shop_domain=shop, client_id=cid, client_secret=secret)
