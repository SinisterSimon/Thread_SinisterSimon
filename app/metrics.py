"""Dashboard numbers, computed in SQL so they stay fast as data grows."""
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Checkout, CheckoutStatus, Customer, Order, Store


@dataclass
class StoreMetrics:
    store_name: str
    checkouts: int
    completed: int
    abandoned: int
    abandonment_rate: float
    reachable_abandoned: int
    abandoned_value_cents: int
    orders: int
    revenue_cents: int
    returned_orders: int
    returned_revenue_cents: int


def latest_store(session: Session) -> Store | None:
    return session.scalars(select(Store).order_by(Store.id.desc()).limit(1)).first()


def store_metrics(session: Session, store: Store) -> StoreMetrics:
    def count(*where) -> int:
        return session.scalar(select(func.count(Checkout.id)).where(*where)) or 0

    in_store = Checkout.store_id == store.id
    checkouts = count(in_store)
    abandoned = count(in_store, Checkout.status == CheckoutStatus.abandoned)
    completed = count(in_store, Checkout.status == CheckoutStatus.completed)
    reachable = (
        session.scalar(
            select(func.count(Checkout.id))
            .join(Customer, Customer.id == Checkout.customer_id)
            .where(in_store, Checkout.status == CheckoutStatus.abandoned, Customer.sms_consent)
        )
        or 0
    )
    abandoned_value = (
        session.scalar(
            select(func.coalesce(func.sum(Checkout.total_cents), 0)).where(
                in_store, Checkout.status == CheckoutStatus.abandoned
            )
        )
        or 0
    )

    def order_sum(*where) -> tuple[int, int]:
        row = session.execute(
            select(func.count(Order.id), func.coalesce(func.sum(Order.total_cents), 0)).where(
                Order.store_id == store.id, *where
            )
        ).one()
        return int(row[0]), int(row[1])

    orders, revenue = order_sum()
    returned, returned_rev = order_sum(Order.source == "returned")

    return StoreMetrics(
        store_name=store.name,
        checkouts=checkouts,
        completed=completed,
        abandoned=abandoned,
        abandonment_rate=(abandoned / checkouts) if checkouts else 0.0,
        reachable_abandoned=reachable,
        abandoned_value_cents=int(abandoned_value),
        orders=orders,
        revenue_cents=revenue,
        returned_orders=returned,
        returned_revenue_cents=returned_rev,
    )
