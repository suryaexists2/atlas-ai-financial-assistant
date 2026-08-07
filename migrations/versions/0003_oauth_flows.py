"""OAuth flows table for server-side one-time OAuth state.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-07
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_flows",
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("code_verifier", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("state", name=op.f("pk_oauth_flows")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_oauth_flows_user_id_users"), ondelete="CASCADE"
        ),
    )
    op.create_index(op.f("ix_oauth_flows_user_id"), "oauth_flows", ["user_id"], unique=False)
    op.create_index(op.f("ix_oauth_flows_chat_id"), "oauth_flows", ["chat_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_oauth_flows_chat_id"), table_name="oauth_flows")
    op.drop_index(op.f("ix_oauth_flows_user_id"), table_name="oauth_flows")
    op.drop_table("oauth_flows")
