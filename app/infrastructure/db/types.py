"""Shared column type variants: JSONB on Postgres, JSON elsewhere."""

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

# Use this everywhere so Postgres gets native JSONB while SQLite stays usable.
JSONType = JSON().with_variant(JSONB(), "postgresql")
