"""Create articles table.

Revision ID: 001
Revises:
Create Date: 2024-12-11 00:00:00.000000

This migration creates the initial 'articles' table that stores:
- Article metadata (id, user_id, status)
- The article guideline content
- All intermediate workflow outputs that were previously stored as files:
  - guidelines_filenames.json → extracted_urls (JSON)
  - perplexity_results.md → perplexity_results (TEXT)
  - perplexity_results_selected.md → perplexity_results_selected (TEXT)
  - perplexity_sources_selected.md → perplexity_sources_selected (TEXT)
  - urls_to_scrape_from_research.md → urls_to_scrape_from_research (TEXT)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the articles table."""
    # Create the article_status enum type if it doesn't exist
    # Using raw SQL with IF NOT EXISTS for idempotency
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE article_status AS ENUM (
                'created',
                'extracting_urls',
                'processing_sources',
                'researching',
                'selecting_sources',
                'scraping_research',
                'finalizing',
                'completed',
                'failed'
            );
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create the articles table
    op.create_table(
        "articles",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique article identifier",
        ),
        # User association
        sa.Column(
            "user_id",
            sa.Text(),
            nullable=False,
            index=True,
            comment="User ID from Descope authentication",
        ),
        # Article guideline content
        sa.Column(
            "guideline_text",
            sa.Text(),
            nullable=False,
            comment="Full content of the article guideline markdown file",
        ),
        # Workflow status
        sa.Column(
            "status",
            postgresql.ENUM(
                "created",
                "extracting_urls",
                "processing_sources",
                "researching",
                "selecting_sources",
                "scraping_research",
                "finalizing",
                "completed",
                "failed",
                name="article_status",
                create_type=False,  # Already created above
            ),
            nullable=False,
            server_default="created",
            comment="Current status of the research workflow",
        ),
        # =====================================================================
        # Intermediate workflow outputs (replacing file-based storage)
        # =====================================================================
        # Replaces: .nova/guidelines_filenames.json
        sa.Column(
            "extracted_urls",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Extracted URLs from guidelines (JSON): github_urls, youtube_videos_urls, other_urls, local_file_paths",
        ),
        # Replaces: .nova/perplexity_results.md
        sa.Column(
            "perplexity_results",
            sa.Text(),
            nullable=True,
            comment="Raw Perplexity research results (markdown format)",
        ),
        # Replaces: .nova/perplexity_results_selected.md
        sa.Column(
            "perplexity_results_selected",
            sa.Text(),
            nullable=True,
            comment="Filtered Perplexity results containing only selected high-quality sources (markdown format)",
        ),
        # Replaces: .nova/perplexity_sources_selected.md
        sa.Column(
            "perplexity_sources_selected",
            sa.Text(),
            nullable=True,
            comment="Comma-separated IDs of selected Perplexity sources",
        ),
        # Replaces: .nova/urls_to_scrape_from_research.md
        sa.Column(
            "urls_to_scrape_from_research",
            sa.Text(),
            nullable=True,
            comment="URLs selected for full content scraping (one per line)",
        ),
        # =====================================================================
        # Timestamps
        # =====================================================================
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the article was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the article was last updated",
        ),
    )
    # Note: Indexes are automatically created via index=True in column definitions above


def downgrade() -> None:
    """Drop the articles table."""
    # Drop table (indexes are automatically dropped with the table)
    op.drop_table("articles")

    # Drop enum type if it exists
    op.execute("DROP TYPE IF EXISTS article_status")
