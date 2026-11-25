"""Shopify's checkout recovery link, appended to drafted messages.

Revision ID: 0006_recovery_url
Revises: 0005_agent_cost
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_recovery_url"
down_revision: str | None = "0005_agent_cost"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("checkouts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("recovery_url", sa.String(length=500), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("checkouts", schema=None) as batch_op:
        batch_op.drop_column("recovery_url")
