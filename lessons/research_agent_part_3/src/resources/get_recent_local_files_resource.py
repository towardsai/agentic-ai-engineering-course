"""Recent local files resource implementation."""

import logging
from typing import Any, Dict

from sqlalchemy import select

from ..db.models import LocalFile
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def get_recent_local_files_resource() -> Dict[str, Any]:
    """
    Get the last 5 uploaded local files across all users.

    Returns files ordered by creation time (most recent first), showing:
    - Local file ID
    - Creation timestamp
    - Filename
    - Article guideline ID
    - User ID
    - First 5 lines of file content

    Returns:
        Dict with list of recent local files
    """
    logger.debug("Fetching recent local files (all users)")

    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        # Query for last 5 local files across all users, ordered by created_at DESC
        stmt = select(LocalFile).order_by(LocalFile.created_at.desc()).limit(5)

        result = await session.execute(stmt)
        files = result.scalars().all()

        # Format the results
        file_list = []
        for file in files:
            # Get first 5 lines of content
            lines = file.content.splitlines()
            first_5_lines = "\n".join(lines[:5])
            if len(lines) > 5:
                first_5_lines += "\n..."

            file_list.append(
                {
                    "local_file_id": str(file.id),
                    "created_at": file.created_at.isoformat(),
                    "filename": file.filename,
                    "article_guideline_id": str(file.article_guideline_id),
                    "user_id": file.user_id,
                    "first_5_lines": first_5_lines,
                    "total_lines": len(lines),
                }
            )

        return {
            "count": len(file_list),
            "files": file_list,
        }
