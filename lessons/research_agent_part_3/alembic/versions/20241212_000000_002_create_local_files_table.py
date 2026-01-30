"""Create local_files table.

Revision ID: 002
Revises: 001
Create Date: 2024-12-12 00:00:00.000000

This migration creates the 'local_files' table that stores:
- Uploaded local files referenced in article guidelines
- Each file has: user_id, article_guideline_id (FK), filename, content
- Many-to-one relationship with articles table
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the local_files table."""
    # Create the local_files table
    op.create_table(
        "local_files",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique local file identifier",
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
        # File metadata
        sa.Column(
            "filename",
            sa.Text(),
            nullable=False,
            comment="Name of the uploaded file",
        ),
        # File content
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="Content of the uploaded file",
        ),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when the file was uploaded",
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
        "ix_local_files_article_user",
        "local_files",
        ["article_guideline_id", "user_id"],
    )


def downgrade() -> None:
    """Drop the local_files table."""
    # Drop index first
    op.drop_index("ix_local_files_article_user", table_name="local_files")

    # Drop table
    op.drop_table("local_files")
