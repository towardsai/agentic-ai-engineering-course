"""Provider-neutral web research tool implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..app.perplexity_handler import append_perplexity_results, compute_next_source_id
from ..app.web_search_handler import resolve_web_search_provider, run_web_search
from ..config.constants import NOVA_FOLDER, PERPLEXITY_RESULTS_FILE
from ..utils.file_utils import validate_research_folder

logger = logging.getLogger(__name__)


def append_search_results_to_file(results_path: Path, queries: List[str], search_results: List[Tuple]) -> int:
    """Append normalized web-search results and return the number of sources added."""
    next_global_id = compute_next_source_id(results_path)
    total_sources = 0

    for query, (_, answer_by_source, citations) in zip(queries, search_results):
        if not citations:
            continue
        previous_id = next_global_id
        next_global_id = append_perplexity_results(
            results_path,
            query,
            answer_by_source,
            citations,
            next_global_id,
        )
        sources_added = next_global_id - previous_id
        total_sources += sources_added
        logger.debug(f"Appended results for query: '{query}' (added {sources_added} source section(s)).")

    return total_sources


async def run_web_research_tool(
    research_directory: str,
    queries: List[str],
    provider: str | None = None,
) -> Dict[str, Any]:
    """Run web research with Perplexity by default or an explicitly selected provider."""
    selected_provider = resolve_web_search_provider(provider)
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER
    validate_research_folder(research_path)

    if not queries:
        return {
            "status": "success",
            "provider": selected_provider,
            "message": f"No queries provided for research folder '{research_directory}' – nothing to do.",
            "queries_processed": 0,
            "queries_succeeded": [],
            "queries_failed": [],
            "sources_added": 0,
        }

    results_path = nova_path / PERPLEXITY_RESULTS_FILE
    results_path.touch(exist_ok=True)

    logger.debug(f"Executing {len(queries)} web-search queries with {selected_provider}...")
    raw_results = await asyncio.gather(
        *(run_web_search(query, provider=selected_provider) for query in queries),
        return_exceptions=True,
    )

    successful_queries: List[str] = []
    successful_results: List[Tuple] = []
    failed_queries: List[Dict[str, str]] = []
    for query, result in zip(queries, raw_results):
        if isinstance(result, BaseException):
            logger.error(f"Web search failed for '{query}': {result}")
            failed_queries.append({"query": query, "error": str(result)})
        else:
            successful_queries.append(query)
            successful_results.append(result)

    total_sources = append_search_results_to_file(results_path, successful_queries, successful_results)
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
        "output_path": str(results_path.resolve()),
        "message": (
            f"Completed web research with {selected_provider} for '{research_directory}'. "
            f"{len(successful_queries)}/{len(queries)} queries succeeded and added {total_sources} "
            f"source sections to {PERPLEXITY_RESULTS_FILE}."
        ),
    }
