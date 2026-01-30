"""Recent article guidelines resource implementation."""

import logging
from typing import Any, Dict

from fastmcp.server.dependencies import get_access_token
from sqlalchemy import select

from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def get_recent_article_guidelines_resource() -> Dict[str, Any]:
    """
    Get the last 5 created article guidelines for the current user.

    Returns articles ordered by creation time (most recent first), showing:
    - Article guideline ID
    - Creation timestamp

    Returns:
        Dict with list of recent article guidelines
    """
    # Get user ID from access token
    token = get_access_token()
    user_id = token.claims["sub"]

    logger.debug(f"Fetching recent article guidelines for user: {user_id}")

    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        # Query for last 5 articles by this user, ordered by created_at DESC
        stmt = select(Article).where(Article.user_id == user_id).order_by(Article.created_at.desc()).limit(5)

        result = await session.execute(stmt)
        articles = result.scalars().all()

        # Format the results
        article_list = [
            {
                "article_guideline_id": str(article.id),
                "created_at": article.created_at.isoformat(),
                "status": article.status.value,
            }
            for article in articles
        ]

        return {
            "user_id": user_id,
            "count": len(article_list),
            "articles": article_list,
        }
