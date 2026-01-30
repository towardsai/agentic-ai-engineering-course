"""Perplexity research tool implementation."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List

from ..app.perplexity_handler import (
    compute_next_source_id_from_text,
    format_perplexity_results,
    run_perplexity_search,
)
from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def run_perplexity_research_tool(article_guideline_id: str, queries: List[str]) -> Dict[str, Any]:
    """
    Run Perplexity research queries and store results in the database.

    Executes the provided queries using Perplexity and appends
    the results to the perplexity_results field in the articles table.

    Args:
        article_guideline_id: UUID of the article guideline in the database
        queries: List of web-search queries to execute

    Returns:
        Dict with status, processing results, and counts
    """
    logger.debug(f"Running Perplexity research for article guideline ID: {article_guideline_id}")

    # Convert string to UUID
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as e:
        msg = f"Invalid article guideline ID format: {article_guideline_id}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    if not queries:
        return {
            "status": "success",
            "message": f"No queries provided for article guideline '{article_guideline_id}' – nothing to do.",
            "queries_processed": 0,
            "sources_added": 0,
        }

    # Query database for article
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)

        if not article:
            msg = f"Article with ID '{article_guideline_id}' not found in database"
            logger.error(msg)
            raise ValueError(msg)

        # Get existing results
        existing_results = article.perplexity_results

        # Compute starting source ID
        starting_id = compute_next_source_id_from_text(existing_results)

        logger.debug(f"Executing {len(queries)} Perplexity queries...")
        tasks = [run_perplexity_search(query) for query in queries]
        search_results = await asyncio.gather(*tasks)
        logger.debug("All Perplexity queries finished. Formatting results.")

        # Format and collect all results
        all_results_parts = []
        current_id = starting_id
        total_sources = 0

        for query, (_, answer_by_source, citations) in zip(queries, search_results):
            if citations:
                formatted_result, next_id = format_perplexity_results(
                    query,
                    answer_by_source,
                    citations,
                    current_id,
                )
                if formatted_result:
                    all_results_parts.append(formatted_result)
                    total_sources += len(citations)
                    current_id = next_id
                    logger.debug(f"Formatted results for query: '{query}' (added {len(citations)} source section(s)).")

        # Concatenate all new results
        new_results_text = "\n".join(all_results_parts)

        # Append to existing perplexity_results
        if existing_results:
            article.perplexity_results = existing_results + "\n" + new_results_text
        else:
            article.perplexity_results = new_results_text

        try:
            await session.commit()
            logger.info(f"Successfully saved Perplexity results to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving Perplexity results to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    processed_queries_count = len(queries)
    return {
        "status": "success",
        "queries_processed": processed_queries_count,
        "sources_added": total_sources,
        "message": (
            f"Successfully completed Perplexity research round for article guideline '{article_guideline_id}'. "
            f"Processed {processed_queries_count} queries and added {total_sources} source sections to database."
        ),
    }
