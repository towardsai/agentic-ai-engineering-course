"""Add research field to articles table.

Revision ID: 007
Revises: 006
Create Date: 2024-12-15 00:00:01.000000

This migration adds the 'research' field to the 'articles' table:
- Stores the final comprehensive research markdown content
- Replaces the need for research.md file in filesystem
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add research field to articles table."""
    op.add_column(
        "articles",
        sa.Column(
            "research",
            sa.Text(),
            nullable=True,
            comment="Final comprehensive research markdown content",
        ),
    )


def downgrade() -> None:
    """Remove research field from articles table."""
    op.drop_column("articles", "research")
