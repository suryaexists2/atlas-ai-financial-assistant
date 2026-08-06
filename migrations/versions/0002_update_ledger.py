"""Update dedup ledger + correlation_id on messages.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-06
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("correlation_id", sa.String(length=36), nullable=True),
    )

    op.create_table(
        "ingested_updates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("update_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("correlation_id", sa.String(length=36), nullable=False),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingested_updates")),
        sa.UniqueConstraint("update_id", name=op.f("uq_ingested_updates_update_id")),
        sa.UniqueConstraint(
            "chat_id", "message_id", name="uq_ingested_updates_chat_id_message_id"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_ingested_updates_user_id_users"),
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        op.f("ix_ingested_updates_update_id"), "ingested_updates", ["update_id"], unique=False
    )
    op.create_index(
        op.f("ix_ingested_updates_chat_id"), "ingested_updates", ["chat_id"], unique=False
    )
    op.create_index(
        op.f("ix_ingested_updates_correlation_id"),
        "ingested_updates",
        ["correlation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ingested_updates_user_id"), "ingested_updates", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_table("ingested_updates")
    op.drop_column("messages", "correlation_id")