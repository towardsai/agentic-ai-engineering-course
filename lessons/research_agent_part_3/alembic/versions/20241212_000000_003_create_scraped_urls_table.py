"""Create scraped_urls table.

Revision ID: 003
Revises: 002
Create Date: 2024-12-12 00:00:00.000000

This migration creates the 'scraped_urls' table that stores:
- Scraped and cleaned web content from URLs referenced in article guidelines
- Each entry has: user_id, article_guideline_id (FK), url, content
- Many-to-one relationship with articles table
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the scraped_urls table."""
    # Create the scraped_urls table
    op.create_table(
        "scraped_urls",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique scraped URL identifier",
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
        # URL
        sa.Column(
            "url",
            sa.Text(),
            nullable=False,
            comment="The URL that was scraped",
        ),
        # Scraped content
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="Cleaned markdown content from the URL",
        ),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the URL was scraped",
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
        "ix_scraped_urls_article_user",
        "scraped_urls",
        ["article_guideline_id", "user_id"],
    )


def downgrade() -> None:
    """Drop the scraped_urls table."""
    # Drop index first
    op.drop_index("ix_scraped_urls_article_user", table_name="scraped_urls")

    # Drop table
    op.drop_table("scraped_urls")
