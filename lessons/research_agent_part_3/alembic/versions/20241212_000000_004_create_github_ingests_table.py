"""Create github_ingests table.

Revision ID: 004
Revises: 003
Create Date: 2024-12-12 00:00:00.000000

This migration creates the 'github_ingests' table that stores:
- GitHub repository analysis results from gitingest
- Each entry has: user_id, article_guideline_id (FK), github_url, gitingest_result
- Many-to-one relationship with articles table
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the github_ingests table."""
    # Create the github_ingests table
    op.create_table(
        "github_ingests",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique GitHub ingest identifier",
        ),
        # User association
        sa.Column(
            "user_id",
            sa.Text(),
            nullable=False,
            index=True,
            comment="User ID from Descope authentication",
        ),
        # Article association (foreign key)
        sa.Column(
            "article_guideline_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            index=True,
            comment="Foreign key to articles table",
        ),
        # GitHub URL
        sa.Column(
            "github_url",
            sa.Text(),
            nullable=False,
            comment="The GitHub URL that was processed",
        ),
        # GitIngest result
        sa.Column(
            "gitingest_result",
            sa.Text(),
            nullable=False,
            comment="Markdown output from GitIngest processing",
        ),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the GitHub URL was processed",
        ),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["article_guideline_id"],
            ["articles.id"],
            ondelete="CASCADE",
        ),
    )

    # Create composite index for efficient lookups by article and user
    op.create_index(
        "ix_github_ingests_article_user",
        "github_ingests",
        ["article_guideline_id", "user_id"],
    )


def downgrade() -> None:
    """Drop the github_ingests table."""
    # Drop index first
    op.drop_index("ix_github_ingests_article_user", table_name="github_ingests")

    # Drop table
    op.drop_table("github_ingests")
