"""Phase 4: holdout groups and true-lift ground truth.

Revision ID: 0003_holdout
Revises: 0002_scoring
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_holdout"
down_revision: str | None = "0002_scoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("holdout_pct", sa.Integer(), server_default="0", nullable=False))
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("arm", sa.String(length=10), server_default="treatment", nullable=False))
        batch_op.add_column(sa.Column("baseline_revenue_cents", sa.Integer(), server_default="0", nullable=False))
        batch_op.create_index(batch_op.f("ix_conversations_arm"), ["arm"], unique=False)

    # Backfill existing chats: a shopper who was coming back anyway would have
    # paid the full cart price with no message.
    op.execute(
        """
        UPDATE conversations
        SET baseline_revenue_cents = (
            SELECT checkouts.total_cents FROM checkouts WHERE checkouts.id = conversations.checkout_id
        )
        WHERE would_buy_anyway
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_conversations_arm"))
        batch_op.drop_column("baseline_revenue_cents")
        batch_op.drop_column("arm")
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.drop_column("holdout_pct")
