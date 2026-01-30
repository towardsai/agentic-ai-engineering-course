"""Create tool_call_usage table for rate limiting.

Revision ID: 008
Revises: 007
Create Date: 2024-12-16 00:00:00.000000

This migration creates the 'tool_call_usage' table that tracks:
- MCP tool calls per user for rate limiting
- Year-month grouping for efficient monthly quota checks
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the tool_call_usage table."""
    op.create_table(
        "tool_call_usage",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique tool call usage identifier",
        ),
        # User association
        sa.Column(
            "user_id",
            sa.Text(),
            nullable=False,
            comment="User ID from Descope authentication",
        ),
        # Tool name
        sa.Column(
            "tool_name",
            sa.Text(),
            nullable=False,
            comment="Name of the MCP tool that was called",
        ),
        # Year-month for efficient counting
        sa.Column(
            "year_month",
            sa.Text(),
            nullable=False,
            comment="Year and month of the tool call (YYYY-MM format)",
        ),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the tool was called",
        ),
    )

    # Composite index for efficient user+month queries (used for counting)
    op.create_index(
        "ix_tool_call_usage_user_month",
        "tool_call_usage",
        ["user_id", "year_month"],
    )

    # Additional index for per-user queries
    op.create_index(
        "ix_tool_call_usage_user_id",
        "tool_call_usage",
        ["user_id"],
    )


def downgrade() -> None:
    """Drop the tool_call_usage table."""
    # Drop indexes first
    op.drop_index("ix_tool_call_usage_user_id", table_name="tool_call_usage")
    op.drop_index("ix_tool_call_usage_user_month", table_name="tool_call_usage")

    # Drop table
    op.drop_table("tool_call_usage")
