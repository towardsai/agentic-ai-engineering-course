"""Scrape research URLs tool implementation."""

import logging
import uuid
from typing import Any, Dict

from ..app.scraping_handler import scrape_urls_concurrently
from ..db.models import Article, ScrapedResearchUrl
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def scrape_research_urls_tool(article_guideline_id: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape the selected research URLs for full content.

    Reads the URLs from the database (saved in step 5.1) and scrapes/cleans each URL's
    full content. YouTube URLs are ignored. The cleaned markdown content is saved to
    the database.

    Args:
        article_guideline_id: UUID of the article guideline in the database
        concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

    Returns:
        Dict with status, processing results, and counts
    """
    logger.debug(f"Scraping research URLs for article guideline ID: {article_guideline_id}")

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

        # Get URLs from urls_to_scrape_from_research field
        urls_text = article.urls_to_scrape_from_research

        if not urls_text:
            return {
                "status": "success",
                "urls_processed": 0,
                "urls_failed": 0,
                "urls_total": 0,
                "message": f"No research URLs to scrape for article guideline '{article_guideline_id}'",
            }

        # Parse URLs (one per line)
        all_urls = [url.strip() for url in urls_text.split("\n") if url.strip()]

        if not all_urls:
            return {
                "status": "success",
                "urls_processed": 0,
                "urls_failed": 0,
                "urls_total": 0,
                "message": f"No valid research URLs found for article guideline '{article_guideline_id}'",
            }

        # Filter out YouTube URLs (we ignore them in step 5.2)
        youtube_urls = [url for url in all_urls if "youtube.com" in url or "youtu.be" in url]
        urls_to_scrape = [url for url in all_urls if url not in youtube_urls]

        if youtube_urls:
            logger.info(f"Ignoring {len(youtube_urls)} YouTube URL(s) in research URLs: {youtube_urls}")

        if not urls_to_scrape:
            return {
                "status": "success",
                "urls_processed": 0,
                "urls_failed": 0,
                "urls_total": len(all_urls),
                "youtube_urls_ignored": len(youtube_urls),
                "message": (
                    f"All {len(all_urls)} URL(s) were YouTube URLs and were ignored. "
                    f"No research URLs to scrape for article guideline '{article_guideline_id}'."
                ),
            }

        # Get article guidelines text for context
        article_guidelines = article.guideline_text

        # Scrape URLs concurrently
        completed_results = await scrape_urls_concurrently(urls_to_scrape, article_guidelines, concurrency_limit)

        # Save to database
        successful_scrapes = 0
        failed_scrapes = 0

        for res in completed_results:
            if res.get("success", False):
                scraped_research_url = ScrapedResearchUrl(
                    user_id=article.user_id,
                    article_guideline_id=article_uuid,
                    url=res.get("url", ""),
                    content=res.get("markdown", ""),
                )
                session.add(scraped_research_url)
                successful_scrapes += 1
            else:
                failed_scrapes += 1

        try:
            await session.commit()
            logger.info(f"Successfully saved {successful_scrapes} scraped research URLs to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving scraped research URLs to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    total_attempted = len(urls_to_scrape)
    message_parts = [
        f"Scraped and cleaned {successful_scrapes}/{total_attempted} research URLs",
        f"from article guideline '{article_guideline_id}'.",
    ]

    if youtube_urls:
        message_parts.append(f"Ignored {len(youtube_urls)} YouTube URL(s).")

    message_parts.append("Results saved to database.")

    return {
        "status": "success" if successful_scrapes > 0 else "warning",
        "urls_processed": successful_scrapes,
        "urls_failed": failed_scrapes,
        "urls_total": total_attempted,
        "youtube_urls_ignored": len(youtube_urls),
        "message": " ".join(message_parts),
    }
