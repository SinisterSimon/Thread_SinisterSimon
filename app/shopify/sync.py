"""Import a Shopify store's abandoned checkouts into Thread's tables.

Real data lands in the same tables as simulated data, marked with
`Store.source = "shopify"`, so the same agents can work on it. What a real
store cannot tell us stays empty: there is no ground truth (`sim_*` fields keep
their defaults) and no product cost.

Running the sync again is safe: rows are matched on their Shopify IDs, new
checkouts are added and existing ones only get their status updated.
"""
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Checkout, CheckoutItem, CheckoutStatus, Customer, Product, Store
from app.shopify.client import ShopifyClient

MAX_PAGES = 5  # 250 checkouts is plenty for a dev store

SHOP_QUERY = "query { shop { name currencyCode } }"

CHECKOUTS_QUERY = """
query AbandonedCheckouts($cursor: String) {
  abandonedCheckouts(first: 50, after: $cursor) {
    nodes {
      id
      name
      createdAt
      updatedAt
      completedAt
      totalPriceSet { shopMoney { amount currencyCode } }
      customer {
        id
        firstName
        defaultEmailAddress { emailAddress }
        defaultPhoneNumber { phoneNumber }
      }
      shippingAddress { firstName phone }
      billingAddress { firstName phone }
      lineItems(first: 20) {
        nodes {
          title
          quantity
          originalUnitPriceSet { shopMoney { amount } }
          product { id title productType }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


@dataclass
class SyncResult:
    store_id: int
    fetched: int
    new_checkouts: int
    updated_checkouts: int


def _cents(amount: str | None) -> int:
    return int((Decimal(amount or "0") * 100).to_integral_value())


def _when(value: str | None) -> datetime | None:
    if not value:
        return None
    # Stored as naive UTC, like the rest of the database.
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).replace(tzinfo=None)


def fetch_abandoned_checkouts(client: ShopifyClient) -> list[dict]:
    nodes: list[dict] = []
    cursor = None
    for _ in range(MAX_PAGES):
        page = client.graphql(CHECKOUTS_QUERY, {"cursor": cursor})["abandonedCheckouts"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return nodes


def _store(session: Session, client: ShopifyClient) -> Store:
    shop = client.graphql(SHOP_QUERY)["shop"]
    store = session.scalars(
        select(Store).where(Store.source == "shopify", Store.shop_domain == client.shop_domain)
    ).first()
    if store is None:
        store = Store(source="shopify", shop_domain=client.shop_domain, created_at=datetime.now())
        session.add(store)
    store.name = shop["name"]
    store.currency = shop["currencyCode"]
    session.flush()
    return store


def _customer(session: Session, store: Store, node: dict) -> Customer:
    c = node.get("customer") or {}
    addr = node.get("shippingAddress") or node.get("billingAddress") or {}
    email = ((c.get("defaultEmailAddress") or {}).get("emailAddress")) or None
    phone = ((c.get("defaultPhoneNumber") or {}).get("phoneNumber")) or addr.get("phone") or None
    external = c.get("id") or (f"email:{email}" if email else f"checkout:{node['id']}")

    customer = session.scalars(
        select(Customer).where(Customer.store_id == store.id, Customer.external_id == external)
    ).first()
    if customer is None:
        customer = Customer(store_id=store.id, external_id=external, created_at=_when(node["createdAt"]))
        session.add(customer)
    customer.first_name = (c.get("firstName") or addr.get("firstName") or "there")[:60]
    customer.email = email
    customer.phone = phone[:20] if phone else None
    customer.preferred_channel = "sms" if phone else "email"
    # We don't know whether the shopper agreed to marketing texts, so a real
    # customer is never marked as consenting. Shadow mode sends nothing anyway.
    customer.sms_consent = False
    session.flush()
    return customer


def _product(session: Session, store: Store, line: dict) -> Product:
    p = line.get("product") or {}
    external = p.get("id") or f"title:{line.get('title')}"
    product = session.scalars(
        select(Product).where(Product.store_id == store.id, Product.external_id == external)
    ).first()
    if product is None:
        product = Product(
            store_id=store.id,
            external_id=external,
            title=(p.get("title") or line.get("title") or "Item")[:200],
            category=(p.get("productType") or "other")[:60] or "other",
            price_cents=_cents(line["originalUnitPriceSet"]["shopMoney"]["amount"]),
            unit_cost_cents=0,  # Shopify doesn't expose cost without extra scopes
        )
        session.add(product)
        session.flush()
    return product


def sync_store(session: Session, client: ShopifyClient) -> SyncResult:
    store = _store(session, client)
    nodes = fetch_abandoned_checkouts(client)
    new = updated = 0
    for node in nodes:
        status = CheckoutStatus.completed if node.get("completedAt") else CheckoutStatus.abandoned
        existing = session.scalars(
            select(Checkout).where(Checkout.store_id == store.id, Checkout.external_id == node["id"])
        ).first()
        if existing is not None:
            if existing.status != status:
                existing.status = status
                updated += 1
            continue

        customer = _customer(session, store, node)
        lines = node["lineItems"]["nodes"]
        session.add(
            Checkout(
                store_id=store.id,
                customer_id=customer.id,
                external_id=node["id"],
                external_name=(node.get("name") or "")[:40] or None,
                status=status,
                total_cents=_cents(node["totalPriceSet"]["shopMoney"]["amount"]),
                started_at=_when(node["createdAt"]),
                abandoned_at=_when(node.get("updatedAt")) if status == CheckoutStatus.abandoned else None,
                items=[
                    CheckoutItem(
                        product_id=_product(session, store, line).id,
                        quantity=line["quantity"],
                        unit_price_cents=_cents(line["originalUnitPriceSet"]["shopMoney"]["amount"]),
                    )
                    for line in lines
                ],
            )
        )
        new += 1
    session.commit()
    return SyncResult(store_id=store.id, fetched=len(nodes), new_checkouts=new, updated_checkouts=updated)
