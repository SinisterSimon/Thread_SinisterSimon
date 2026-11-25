"""Shopify webhooks: Thread hears about a new or changed checkout as it happens.

Shopify signs every delivery with the app's client secret (HMAC-SHA256 of the raw body,
base64, in the X-Shopify-Hmac-Sha256 header). Thread only acts on deliveries whose
signature matches; anything else is rejected, so nobody can fake a checkout event.

Apps made in the Shopify admin register webhooks through the Admin API, which is what
`register` does (see scripts/register_webhooks.py).
"""
import base64
import hashlib
import hmac

from app.shopify.client import ShopifyClient

TOPICS = ("CHECKOUTS_CREATE", "CHECKOUTS_UPDATE")  # both need the read_orders scope

LIST_QUERY = """
query { webhookSubscriptions(first: 50) { nodes { id topic uri } } }
"""

CREATE_MUTATION = """
mutation Create($topic: WebhookSubscriptionTopic!, $sub: WebhookSubscriptionInput!) {
  webhookSubscriptionCreate(topic: $topic, webhookSubscription: $sub) {
    webhookSubscription { id topic uri }
    userErrors { field message }
  }
}
"""


def verify_hmac(body: bytes, signature: str | None, secret: str) -> bool:
    """True only if Shopify signed exactly this body with our app's secret."""
    if not signature or not secret:
        return False
    expected = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(expected, signature.strip())


def register(client: ShopifyClient, uri: str) -> list[str]:
    """Subscribe Thread's endpoint to checkout events. Safe to run again: existing ones are kept."""
    existing = {(n["topic"], n["uri"]) for n in client.graphql(LIST_QUERY)["webhookSubscriptions"]["nodes"]}
    report = []
    for topic in TOPICS:
        if (topic, uri) in existing:
            report.append(f"{topic}: already subscribed")
            continue
        out = client.graphql(CREATE_MUTATION, {"topic": topic, "sub": {"uri": uri, "format": "JSON"}})
        errors = out["webhookSubscriptionCreate"]["userErrors"]
        report.append(f"{topic}: " + ("; ".join(e["message"] for e in errors) if errors else "subscribed"))
    return report
