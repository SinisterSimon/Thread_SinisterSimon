"""Synthetic store traffic.

Generates a store with products and customers, then simulates checkouts over a
number of days. Some checkouts complete, the rest are abandoned. Some abandoned
shoppers come back and buy on their own, with no message at all — exactly the
purchases a recovery tool can wrongly take credit for.

Every run is reproducible: the same seed gives the same data.
"""
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import (
    Checkout,
    CheckoutItem,
    CheckoutStatus,
    Customer,
    Order,
    Product,
    Store,
)

FIRST_NAMES = [
    "Daniel", "Maria", "Liis", "Karl", "Emma", "Noah", "Sofia", "Oliver", "Mia",
    "Lucas", "Ava", "Mark", "Grete", "Jaan", "Nora", "Leo", "Ella", "Hugo",
]

CATALOG = [
    # (title, category, price in dollars, cost share)
    ("Linen Dress", "apparel", 89, 0.35),
    ("Merino Sweater", "apparel", 120, 0.40),
    ("Everyday Tee", "apparel", 32, 0.30),
    ("AquaFlow Pet Fountain", "pets", 54, 0.45),
    ("Replacement Filters (3-pack)", "pets", 18, 0.25),
    ("Ceramic Pour-Over Set", "home", 64, 0.38),
    ("Organic Face Serum", "beauty", 48, 0.20),
    ("Daily Moisturiser", "beauty", 36, 0.22),
    ("Trail Running Socks", "outdoor", 22, 0.30),
    ("Insulated Bottle", "outdoor", 38, 0.35),
]


@dataclass
class SimConfig:
    seed: int = 42
    store_name: str = "Demo Store"
    days: int = 30
    start: datetime = field(default_factory=lambda: datetime(2026, 11, 1))
    customers: int = 400
    checkouts_per_day: int = 40
    # Share of started checkouts that are abandoned (industry checkout abandonment
    # is often quoted around 25-40%; we default to the high end).
    abandon_rate: float = 0.40
    # Share of customers who give a phone number AND consent to texts.
    sms_reachable_rate: float = 0.55
    # Share of abandoned shoppers who come back and buy with no prompting.
    natural_return_rate: float = 0.15
    natural_return_max_hours: int = 72


@dataclass
class SimSummary:
    store_id: int
    checkouts: int
    completed: int
    abandoned: int
    reachable_abandoned: int
    natural_returns: int


def _pick_items(rng: random.Random, products: list[Product]) -> list[tuple[Product, int]]:
    n_lines = rng.choices([1, 2, 3], weights=[60, 30, 10])[0]
    chosen = rng.sample(products, n_lines)
    return [(p, rng.choices([1, 2], weights=[85, 15])[0]) for p in chosen]


def run_simulation(session: Session, config: SimConfig | None = None) -> SimSummary:
    cfg = config or SimConfig()
    rng = random.Random(cfg.seed)

    store = Store(name=cfg.store_name, created_at=cfg.start)
    session.add(store)
    session.flush()

    products = [
        Product(
            store_id=store.id,
            title=title,
            category=category,
            price_cents=price * 100,
            unit_cost_cents=round(price * 100 * cost_share),
        )
        for title, category, price, cost_share in CATALOG
    ]
    session.add_all(products)

    customers = []
    for i in range(cfg.customers):
        reachable = rng.random() < cfg.sms_reachable_rate
        customers.append(
            Customer(
                store_id=store.id,
                first_name=rng.choice(FIRST_NAMES),
                phone=f"+1555{i:07d}" if reachable else None,
                sms_consent=reachable,
                preferred_channel=rng.choices(["imessage", "sms", "whatsapp"], [55, 35, 10])[0],
                price_sensitive=rng.random() < 0.35,
                created_at=cfg.start,
            )
        )
    session.add_all(customers)
    session.flush()

    end = cfg.start + timedelta(days=cfg.days)
    counts = dict(checkouts=0, completed=0, abandoned=0, reachable=0, returns=0)

    for day in range(cfg.days):
        day_start = cfg.start + timedelta(days=day)
        # Daily volume wobbles +/-25% so charts don't look flat.
        n = max(1, round(cfg.checkouts_per_day * rng.uniform(0.75, 1.25)))
        for _ in range(n):
            customer = rng.choice(customers)
            started = day_start + timedelta(minutes=rng.randint(0, 24 * 60 - 1))
            lines = _pick_items(rng, products)
            total = sum(p.price_cents * q for p, q in lines)
            cost = sum(p.unit_cost_cents * q for p, q in lines)

            checkout = Checkout(
                store_id=store.id,
                customer_id=customer.id,
                total_cents=total,
                started_at=started,
                status=CheckoutStatus.completed,
                sim_persuadability=rng.randint(0, 100),
                items=[
                    CheckoutItem(product_id=p.id, quantity=q, unit_price_cents=p.price_cents)
                    for p, q in lines
                ],
            )
            # No flush per checkout: orders attach through the relationship and
            # everything is inserted in batches by one flush at the end.
            session.add(checkout)
            counts["checkouts"] += 1

            if rng.random() >= cfg.abandon_rate:
                checkout.order = Order(
                    store_id=store.id,
                    customer_id=customer.id,
                    total_cents=total,
                    cost_cents=cost,
                    placed_at=started + timedelta(minutes=rng.randint(2, 15)),
                    source="same_session",
                )
                counts["completed"] += 1
                continue

            checkout.status = CheckoutStatus.abandoned
            checkout.abandoned_at = started + timedelta(minutes=rng.randint(3, 30))
            counts["abandoned"] += 1
            if customer.sms_consent:
                counts["reachable"] += 1

            if rng.random() < cfg.natural_return_rate:
                return_at = checkout.abandoned_at + timedelta(
                    hours=rng.uniform(1, cfg.natural_return_max_hours)
                )
                checkout.sim_returns_naturally = True
                checkout.sim_natural_return_at = return_at
                # Only record the order if it falls inside the simulated window.
                if return_at <= end:
                    checkout.order = Order(
                        store_id=store.id,
                        customer_id=customer.id,
                        total_cents=total,
                        cost_cents=cost,
                        placed_at=return_at,
                        source="returned",
                    )
                    counts["returns"] += 1

    session.commit()
    return SimSummary(
        store_id=store.id,
        checkouts=counts["checkouts"],
        completed=counts["completed"],
        abandoned=counts["abandoned"],
        reachable_abandoned=counts["reachable"],
        natural_returns=counts["returns"],
    )
