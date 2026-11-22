"""Database tables.

Money is stored as integer cents to avoid floating-point rounding errors.

The simulator also stores "ground truth" fields (prefixed sim_) that a real
store would never know, such as whether an abandoned shopper was going to come
back on their own. Later phases use them to check whether our holdout test
recovers the true lift — the measurement question at the heart of this project.
"""
import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class CheckoutStatus(str, enum.Enum):
    open = "open"            # shopper is still in checkout
    abandoned = "abandoned"  # left without paying
    completed = "completed"  # paid in the same session


class ExperimentGroup(str, enum.Enum):
    treatment = "treatment"  # gets recovery messages
    holdout = "holdout"      # deliberately gets nothing (Phase 4)


class Store(Base):
    __tablename__ = "stores"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime)

    products: Mapped[list["Product"]] = relationship(back_populates="store")
    customers: Mapped[list["Customer"]] = relationship(back_populates="store")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(60))
    price_cents: Mapped[int] = mapped_column(Integer)
    unit_cost_cents: Mapped[int] = mapped_column(Integer)  # for margin / contribution

    store: Mapped[Store] = relationship(back_populates="products")


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    first_name: Mapped[str] = mapped_column(String(60))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sms_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    preferred_channel: Mapped[str] = mapped_column(String(20), default="sms")
    price_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    store: Mapped[Store] = relationship(back_populates="customers")
    checkouts: Mapped[list["Checkout"]] = relationship(back_populates="customer")


class Checkout(Base):
    __tablename__ = "checkouts"

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[CheckoutStatus] = mapped_column(Enum(CheckoutStatus), index=True)
    total_cents: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    abandoned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Set in Phase 4 when a checkout is randomly assigned to an experiment arm.
    experiment_group: Mapped[ExperimentGroup | None] = mapped_column(
        Enum(ExperimentGroup), nullable=True
    )

    # Ground truth from the simulator: would this shopper return unprompted, and when?
    sim_returns_naturally: Mapped[bool] = mapped_column(Boolean, default=False)
    sim_natural_return_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # How persuadable they are by a good message (0-100). Used in Phase 2+.
    sim_persuadability: Mapped[int] = mapped_column(Integer, default=0)

    customer: Mapped[Customer] = relationship(back_populates="checkouts")
    items: Mapped[list["CheckoutItem"]] = relationship(
        back_populates="checkout", cascade="all, delete-orphan"
    )
    order: Mapped["Order | None"] = relationship(back_populates="checkout")


class CheckoutItem(Base):
    __tablename__ = "checkout_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    checkout_id: Mapped[int] = mapped_column(ForeignKey("checkouts.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price_cents: Mapped[int] = mapped_column(Integer)

    checkout: Mapped[Checkout] = relationship(back_populates="items")
    product: Mapped[Product] = relationship()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    checkout_id: Mapped[int | None] = mapped_column(
        ForeignKey("checkouts.id"), nullable=True, unique=True
    )
    total_cents: Mapped[int] = mapped_column(Integer)
    cost_cents: Mapped[int] = mapped_column(Integer)  # product cost, for contribution
    placed_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    # "same_session" or "returned" (came back after abandoning).
    source: Mapped[str] = mapped_column(String(20))

    checkout: Mapped[Checkout | None] = relationship(back_populates="order")
