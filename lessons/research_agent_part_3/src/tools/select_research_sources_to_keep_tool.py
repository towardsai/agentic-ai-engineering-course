"""Research sources selection tool implementation."""

import logging
import re
import uuid
from typing import Any, Dict

from ..app.source_selection_handler import select_sources
from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


def extract_selected_blocks_content(selected_ids: list[int], md_results: str) -> str:
    """
    Extract selected source blocks from markdown results and return as content.

    Parses the markdown results to find source blocks by their IDs, filters
    to include only the selected sources, and returns the filtered content
    as a string.

    Args:
        selected_ids: List of source IDs to include in the output
        md_results: Raw markdown content containing all source blocks

    Returns:
        str: Filtered markdown content containing only selected source blocks
    """
    if selected_ids:
        # Split original results into blocks by source
        block_pattern = re.compile(r"(### Source \[(\d+)]:[\s\S]*?)(?=### Source \[|\Z)")
        selected_blocks: list[str] = []
        for block, sid_str in block_pattern.findall(md_results):
            if int(sid_str) in selected_ids:
                selected_blocks.append(block.rstrip())
        content_out = "\n\n".join(selected_blocks) + "\n" if selected_blocks else ""
    else:
        content_out = ""

    return content_out


async def select_research_sources_to_keep_tool(article_guideline_id: str) -> Dict[str, Any]:
    """
    Automatically select high-quality sources from Perplexity results.

    Uses LLM to evaluate each source in the database for trustworthiness,
    authority, and relevance based on the article guidelines. Saves the
    comma-separated IDs and filtered results to the database.

    Args:
        article_guideline_id: UUID of the article guideline in the database

    Returns:
        Dict with status, selection results, and counts
    """
    logger.debug(f"Selecting research sources to keep for article guideline ID: {article_guideline_id}")

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

        # Get article guidelines and perplexity results
        article_guidelines = article.guideline_text or ""
        md_results = article.perplexity_results

        if not md_results:
            msg = f"No Perplexity results found for article '{article_guideline_id}'. Run the research round first."
            logger.error(msg)
            raise ValueError(msg)

        # Select sources using LLM
        selected_ids = await select_sources(article_guidelines, md_results)

        # Extract selected blocks content
        content_out = extract_selected_blocks_content(selected_ids, md_results)

        # Write to database fields
        article.perplexity_sources_selected = ",".join(map(str, selected_ids))
        article.perplexity_results_selected = content_out

        try:
            await session.commit()
            logger.info(f"Successfully saved {len(selected_ids)} selected sources to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving selected sources to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    return {
        "status": "success",
        "sources_selected_count": len(selected_ids),
        "selected_source_ids": selected_ids,
        "message": f"✅ Selected {len(selected_ids)} source(s) and saved to database for article '{article_guideline_id}'.",
    }
