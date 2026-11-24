"""Phase 3: conversation scoring and guardrails.

Revision ID: 0002_scoring
Revises: 0001_baseline
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_scoring"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("guardrails", sa.Boolean(), server_default=sa.false(), nullable=False))
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("guardrail_fixes", sa.Integer(), server_default="0", nullable=False))

    op.create_table('conversation_scores',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=False),
    sa.Column('addressed_objection', sa.Boolean(), nullable=False),
    sa.Column('correct_handoff', sa.Boolean(), nullable=False),
    sa.Column('discount_within_policy', sa.Boolean(), nullable=False),
    sa.Column('no_pressure', sa.Boolean(), nullable=False),
    sa.Column('respected_browser', sa.Boolean(), nullable=False),
    sa.Column('judge_model', sa.String(length=60), nullable=False),
    sa.Column('helpfulness', sa.Integer(), nullable=False),
    sa.Column('accuracy', sa.Integer(), nullable=False),
    sa.Column('tone', sa.Integer(), nullable=False),
    sa.Column('judge_notes', sa.String(length=1000), nullable=False),
    sa.Column('quality', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('conversation_scores', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_conversation_scores_conversation_id'), ['conversation_id'], unique=True)



def downgrade() -> None:
    with op.batch_alter_table("conversation_scores", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_conversation_scores_conversation_id"))
    op.drop_table("conversation_scores")
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_column("guardrail_fixes")
    with op.batch_alter_table("experiments", schema=None) as batch_op:
        batch_op.drop_column("guardrails")
