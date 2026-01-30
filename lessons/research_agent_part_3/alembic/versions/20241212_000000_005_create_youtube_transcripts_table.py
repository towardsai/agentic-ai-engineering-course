"""Create youtube_transcripts table.

Revision ID: 005
Revises: 004
Create Date: 2024-12-12 00:00:00.000000

This migration creates the 'youtube_transcripts' table that stores:
- YouTube video transcriptions from article guidelines
- Each entry has: user_id, article_guideline_id (FK), youtube_url, transcription
- Many-to-one relationship with articles table
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the youtube_transcripts table."""
    # Create the youtube_transcripts table
    op.create_table(
        "youtube_transcripts",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique YouTube transcript identifier",
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
        # YouTube URL
        sa.Column(
            "youtube_url",
            sa.Text(),
            nullable=False,
            comment="The YouTube URL that was transcribed",
        ),
        # Transcription
        sa.Column(
            "transcription",
            sa.Text(),
            nullable=False,
            comment="Markdown transcription content from the video",
        ),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the video was transcribed",
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
        "ix_youtube_transcripts_article_user",
        "youtube_transcripts",
        ["article_guideline_id", "user_id"],
    )


def downgrade() -> None:
    """Drop the youtube_transcripts table."""
    # Drop index first
    op.drop_index("ix_youtube_transcripts_article_user", table_name="youtube_transcripts")

    # Drop table
    op.drop_table("youtube_transcripts")
