"""Safe sending: a log of every send attempt and the numbers that opted out.

Revision ID: 0007_sending
Revises: 0006_recovery_url
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_sending"
down_revision: str | None = "0006_recovery_url"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sent_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("to_number", sa.String(length=20), nullable=False),
        sa.Column("body", sa.String(length=1600), nullable=False),
        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("detail", sa.String(length=300), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("sent_messages", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_sent_messages_conversation_id"), ["conversation_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_sent_messages_to_number"), ["to_number"], unique=False)
        batch_op.create_index(batch_op.f("ix_sent_messages_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_sent_messages_created_at"), ["created_at"], unique=False)
    op.create_table(
        "opt_outs",
        sa.Column("number", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("number"),
    )


def downgrade() -> None:
    op.drop_table("opt_outs")
    with op.batch_alter_table("sent_messages", schema=None) as batch_op:
        for name in ("created_at", "status", "to_number", "conversation_id"):
            batch_op.drop_index(batch_op.f(f"ix_sent_messages_{name}"))
    op.drop_table("sent_messages")
