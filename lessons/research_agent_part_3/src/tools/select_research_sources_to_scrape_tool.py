"""Research sources to scrape selection tool implementation."""

import logging
import uuid
from typing import Any, Dict

from ..app.source_selection_handler import load_scraped_guideline_context, select_top_sources
from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def select_research_sources_to_scrape_tool(article_guideline_id: str, max_sources: int = 5) -> Dict[str, Any]:
    """
    Select up to max_sources priority research sources to scrape in full.

    Analyzes the filtered Perplexity results from the database together with
    the article guidelines and material already scraped, then chooses diverse,
    authoritative sources. The chosen URLs are saved to the database.

    Args:
        article_guideline_id: UUID of the article guideline in the database
        max_sources: Maximum number of sources to select (default: 5)

    Returns:
        Dict with status, selection results, and reasoning
    """
    logger.debug(f"Selecting research sources to scrape for article guideline ID: {article_guideline_id}")

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

        # Get article guidelines and selected results
        article_guidelines = article.guideline_text or ""
        md_results_selected = article.perplexity_results_selected

        if not md_results_selected:
            msg = f"No selected Perplexity results found for article '{article_guideline_id}'. Run source selection first."
            logger.error(msg)
            raise ValueError(msg)

        # Load scraped guideline context from database
        guideline_ctx = await load_scraped_guideline_context(article_guideline_id)

        # Select top sources
        selection_result = await select_top_sources(article_guidelines, guideline_ctx, md_results_selected, max_sources)
        top_urls = selection_result["selected_urls"]
        reasoning = selection_result["reasoning"]

        # Write URLs to database (one per line)
        urls_text = "\n".join(top_urls)
        article.urls_to_scrape_from_research = urls_text

        try:
            await session.commit()
            logger.info(f"Successfully saved {len(top_urls)} URLs to scrape to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving URLs to scrape to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    return {
        "status": "success",
        "sources_selected": top_urls,
        "sources_selected_count": len(top_urls),
        "reasoning": reasoning,
        "message": (
            f"✅ Selected {len(top_urls)} URL(s) to scrape and saved to database for article '{article_guideline_id}'.\n"
            f"Reasoning: {reasoning}"
        ),
    }
