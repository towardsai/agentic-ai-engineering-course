"""GitHub URLs processing tool implementation."""

import logging
import uuid
from typing import Any, Dict

from ..app.github_handler import process_github_url
from ..config.settings import settings
from ..db.models import Article, GitHubIngest
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def process_github_urls_tool(article_guideline_id: str) -> Dict[str, Any]:
    """
    Process GitHub URLs from article guidelines in the database.

    Reads the list of GitHub URLs from the database (extracted in step 1.3)
    and processes each URL with GitIngest. The processed markdown content is
    saved to the database.

    Args:
        article_guideline_id: UUID of the article guideline in the database

    Returns:
        Dict with status, processing results, and counts
    """
    logger.debug(f"Processing GitHub URLs for article guideline ID: {article_guideline_id}")

    # Convert string to UUID
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as e:
        msg = f"Invalid article guideline ID format: {article_guideline_id}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Query database for article
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)

        if not article:
            msg = f"Article with ID '{article_guideline_id}' not found in database"
            logger.error(msg)
            raise ValueError(msg)

        # Get URLs from extracted_urls
        github_urls = []
        if article.extracted_urls and "github_urls" in article.extracted_urls:
            github_urls = article.extracted_urls["github_urls"]

        if not github_urls:
            return {
                "status": "success",
                "urls_processed": 0,
                "urls_total": 0,
                "message": f"No GitHub URLs found in article guideline '{article_guideline_id}'",
            }

        # Get GitHub token
        github_token = settings.github_token.get_secret_value() if settings.github_token else None

        logger.debug(f"Processing {len(github_urls)} GitHub URLs...")

        # Process GitHub URLs sequentially
        successful_ingests = 0
        failed_ingests = 0

        for url in github_urls:
            try:
                result = await process_github_url(url, github_token)
                if result.get("success", False):
                    github_ingest = GitHubIngest(
                        user_id=article.user_id,
                        article_guideline_id=article_uuid,
                        github_url=result.get("url", ""),
                        gitingest_result=result.get("markdown", ""),
                    )
                    session.add(github_ingest)
                    successful_ingests += 1
                else:
                    failed_ingests += 1
            except Exception as e:
                logger.error(f"Error processing GitHub URL {url}: {e}")
                failed_ingests += 1
                continue

        try:
            await session.commit()
            logger.info(f"Successfully saved {successful_ingests} GitHub ingests to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving GitHub ingests to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    total_attempted = len(github_urls)
    return {
        "status": "success" if successful_ingests > 0 else "warning",
        "urls_processed": successful_ingests,
        "urls_failed": failed_ingests,
        "urls_total": total_attempted,
        "message": (
            f"Processed {successful_ingests}/{total_attempted} GitHub URLs from article guideline "
            f"'{article_guideline_id}'. Results saved to database."
        ),
    }
