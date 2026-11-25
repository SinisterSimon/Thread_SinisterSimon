"""Per-conversation agent model cost, for comparing models on lift per dollar.

Revision ID: 0005_agent_cost
Revises: 0004_shopify
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_agent_cost"
down_revision: str | None = "0004_shopify"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("agent_cost_micros", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_column("agent_cost_micros")
