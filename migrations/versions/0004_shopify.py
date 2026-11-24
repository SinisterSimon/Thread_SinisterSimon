"""v2: stores imported from Shopify, and shadow-mode experiments.

Revision ID: 0004_shopify
Revises: 0003_holdout
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_shopify"
down_revision: str | None = "0003_holdout"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("stores", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source", sa.String(length=20), server_default="simulated", nullable=False))
        batch_op.add_column(sa.Column("shop_domain", sa.String(length=120), nullable=True))
        batch_op.create_index(batch_op.f("ix_stores_source"), ["source"], unique=False)
    with op.batch_alter_table("products", schema=None) as batch_op:
        batch_op.add_column(sa.Column("external_id", sa.String(length=80), nullable=True))
        batch_op.create_index(batch_op.f("ix_products_external_id"), ["external_id"], unique=False)
    with op.batch_alter_table("customers", schema=None) as batch_op:
        batch_op.add_column(sa.Column("email", sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column("external_id", sa.String(length=80), nullable=True))
        batch_op.create_index(batch_op.f("ix_customers_external_id"), ["external_id"], unique=False)
    with op.batch_alter_table("checkouts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("external_id", sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column("external_name", sa.String(length=40), nullable=True))
        batch_op.create_index(batch_op.f("ix_checkouts_external_id"), ["external_id"], unique=False)
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("shadow", sa.Boolean(), server_default=sa.false(), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.drop_column("shadow")
    with op.batch_alter_table("checkouts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_checkouts_external_id"))
        batch_op.drop_column("external_name")
        batch_op.drop_column("external_id")
    with op.batch_alter_table("customers", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_customers_external_id"))
        batch_op.drop_column("external_id")
        batch_op.drop_column("email")
    with op.batch_alter_table("products", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_products_external_id"))
        batch_op.drop_column("external_id")
    with op.batch_alter_table("stores", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_stores_source"))
        batch_op.drop_column("shop_domain")
        batch_op.drop_column("source")
