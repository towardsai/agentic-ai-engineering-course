"""Provider-neutral database-backed web research tool."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Tuple

from ..app.perplexity_handler import compute_next_source_id_from_text, format_perplexity_results
from ..app.web_search_handler import resolve_web_search_provider, run_web_search
from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def run_web_research_tool(
    article_guideline_id: str,
    queries: List[str],
    provider: str | None = None,
) -> Dict[str, Any]:
    """Run web research and append normalized results to the legacy database field."""
    selected_provider = resolve_web_search_provider(provider)
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as exc:
        raise ValueError(f"Invalid article guideline ID format: {article_guideline_id}") from exc

    if not queries:
        return {
            "status": "success",
            "provider": selected_provider,
            "message": f"No queries provided for article guideline '{article_guideline_id}' – nothing to do.",
            "queries_processed": 0,
            "queries_succeeded": [],
            "queries_failed": [],
            "sources_added": 0,
        }

    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)
        if not article:
            raise ValueError(f"Article with ID '{article_guideline_id}' not found in database")

        raw_results = await asyncio.gather(
            *(run_web_search(query, provider=selected_provider) for query in queries),
            return_exceptions=True,
        )
        successful: List[Tuple[str, Tuple]] = []
        failed_queries: List[Dict[str, str]] = []
        for query, result in zip(queries, raw_results):
            if isinstance(result, BaseException):
                logger.error(f"Web search failed for '{query}': {result}")
                failed_queries.append({"query": query, "error": str(result)})
            else:
                successful.append((query, result))

        existing_results = article.perplexity_results
        current_id = compute_next_source_id_from_text(existing_results)
        all_results_parts: List[str] = []
        total_sources = 0
        for query, (_, answer_by_source, citations) in successful:
            formatted_result, next_id = format_perplexity_results(
                query,
                answer_by_source,
                citations,
                current_id,
            )
            if formatted_result:
                all_results_parts.append(formatted_result)
                total_sources += next_id - current_id
                current_id = next_id

        if all_results_parts:
            new_results_text = "\n".join(all_results_parts)
            article.perplexity_results = f"{existing_results}\n{new_results_text}" if existing_results else new_results_text
            try:
                await session.commit()
            except Exception as exc:
                await session.rollback()
                raise ValueError(f"Error saving web research results to database: {exc}") from exc

    successful_queries = [query for query, _ in successful]
    if failed_queries and successful_queries:
        status = "partial_success"
    elif failed_queries:
        status = "error"
    else:
        status = "success"

    return {
        "status": status,
        "provider": selected_provider,
        "queries_processed": len(queries),
        "queries_succeeded": successful_queries,
        "queries_failed": failed_queries,
        "sources_added": total_sources,
        "message": (
            f"Completed web research with {selected_provider} for article guideline '{article_guideline_id}'. "
            f"{len(successful_queries)}/{len(queries)} queries succeeded and added {total_sources} source sections."
        ),
    }
