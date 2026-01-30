"""Scrape and clean other URLs tool implementation."""

import logging
import uuid
from typing import Any, Dict

from ..app.scraping_handler import scrape_urls_concurrently
from ..db.models import Article, ScrapedUrl
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def scrape_and_clean_other_urls_tool(article_guideline_id: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape and clean other URLs from article guidelines in the database.

    Reads the list of other URLs from the database (extracted in step 1.3)
    and scrapes/cleans each URL. The cleaned markdown content is saved to
    the database.

    Args:
        article_guideline_id: UUID of the article guideline in the database
        concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

    Returns:
        Dict with status, processing results, and counts
    """
    logger.debug(f"Scraping and cleaning other URLs for article guideline ID: {article_guideline_id}")

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
        other_urls = []
        if article.extracted_urls and "other_urls" in article.extracted_urls:
            other_urls = article.extracted_urls["other_urls"]

        if not other_urls:
            return {
                "status": "success",
                "urls_processed": 0,
                "urls_total": 0,
                "message": f"No other URLs found in article guideline '{article_guideline_id}'",
            }

        # Get article guidelines text for context
        article_guidelines = article.guideline_text

        # Scrape URLs concurrently
        completed_results = await scrape_urls_concurrently(other_urls, article_guidelines, concurrency_limit)

        # Save to database
        successful_scrapes = 0
        failed_scrapes = 0

        for res in completed_results:
            if res.get("success", False):
                scraped_url = ScrapedUrl(
                    user_id=article.user_id,
                    article_guideline_id=article_uuid,
                    url=res.get("url", ""),
                    content=res.get("markdown", ""),
                )
                session.add(scraped_url)
                successful_scrapes += 1
            else:
                failed_scrapes += 1

        try:
            await session.commit()
            logger.info(f"Successfully saved {successful_scrapes} scraped URLs to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving scraped URLs to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    total_attempted = len(other_urls)
    return {
        "status": "success" if successful_scrapes > 0 else "warning",
        "urls_processed": successful_scrapes,
        "urls_failed": failed_scrapes,
        "urls_total": total_attempted,
        "message": (
            f"Scraped and cleaned {successful_scrapes}/{total_attempted} other URLs from article guideline "
            f"'{article_guideline_id}'. Results saved to database."
        ),
    }
