"""Perplexity research tool implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..app.perplexity_handler import (
    append_perplexity_results,
    compute_next_source_id,
    run_perplexity_search,
)
from ..config.constants import (
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_FILE,
)
from ..utils.file_utils import validate_research_folder

logger = logging.getLogger(__name__)


def append_search_results_to_file(results_path: Path, queries: List[str], search_results: List[Tuple]) -> int:
    """
    Process search results and append them to the results file.

    Args:
        results_path: Path to the results file
        queries: List of search queries
        search_results: List of search results from run_perplexity_search

    Returns:
        Total number of sources added
    """
    next_global_id = compute_next_source_id(results_path)
    total_sources = 0

    for query, (_, answer_by_source, citations) in zip(queries, search_results):
        if citations:
            next_global_id = append_perplexity_results(
                results_path,
                query,
                answer_by_source,
                citations,
                next_global_id,
            )
            total_sources += len(citations)
            logger.debug(f"Appended results for query: '{query}' (added {len(citations)} source section(s)).")

    return total_sources


async def run_perplexity_research_tool(research_directory: str, queries: List[str]) -> Dict[str, Any]:
    """
    Run Perplexity research queries for the research folder.

    Executes the provided queries using Perplexity and appends
    the results to perplexity_results.md in the research directory. Each query
    result includes the answer and source citations.

    Args:
        research_directory: Path to the research directory where results will be saved
        queries: List of web-search queries to execute

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Running Perplexity research for directory: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    if not queries:
        return {
            "status": "success",
            "message": f"No queries provided for research folder '{research_directory}' – nothing to do.",
            "queries_processed": 0,
            "sources_added": 0,
            "queries": queries,
        }

    results_path = nova_path / PERPLEXITY_RESULTS_FILE

    # Ensure output file exists
    results_path.touch(exist_ok=True)

    logger.debug(f"Executing {len(queries)} Perplexity queries...")
    tasks = [run_perplexity_search(query) for query in queries]
    search_results = await asyncio.gather(*tasks)
    logger.debug("All Perplexity queries finished. Appending results.")

    # Process and append search results to file
    total_sources = append_search_results_to_file(results_path, queries, search_results)

    processed_queries_count = len(queries)
    return {
        "status": "success",
        "queries_processed": processed_queries_count,
        "sources_added": total_sources,
        "output_path": str(results_path.resolve()),
        "message": (
            f"Successfully completed Perplexity research round for research folder '{research_directory}'. "
            f"Processed {processed_queries_count} queries and added {total_sources} "
            f"source sections to {PERPLEXITY_RESULTS_FILE}"
        ),
    }
