"""Tavily web-search operations and response normalization."""

import logging
from typing import Any, Dict, Tuple

from ..config.settings import settings

logger = logging.getLogger(__name__)


def parse_tavily_response(response: Dict[str, Any]) -> Tuple[str, Dict[int, str], Dict[int, str]]:
    """Normalize a Tavily response to Nova's shared web-search result format."""
    answer_by_source: Dict[int, str] = {}
    citations: Dict[int, str] = {}

    for result in response.get("results", []):
        url = str(result.get("url", "")).strip()
        content = str(result.get("content", "")).strip()
        if not url or not content:
            continue

        title = str(result.get("title", "")).strip()
        source_id = len(citations) + 1
        answer_by_source[source_id] = f"{title}\n\n{content}" if title else content
        citations[source_id] = url

    full_answer_lines = []
    for source_id, url in citations.items():
        full_answer_lines.append(f"### [{source_id}]: {url}")
        full_answer_lines.append(answer_by_source[source_id])
        full_answer_lines.append("")

    return "\n".join(full_answer_lines), answer_by_source, citations


async def run_tavily_search(query: str) -> Tuple[str, Dict[int, str], Dict[int, str]]:
    """Run a Tavily search and return full answer, per-source content, and URLs."""
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY environment variable not set.")

    try:
        from tavily import AsyncTavilyClient
    except ImportError as exc:
        raise RuntimeError(
            "The Tavily provider requires the optional 'tavily-python' package. Install it in this environment first."
        ) from exc

    client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())
    search_kwargs: Dict[str, Any] = {
        "query": query,
        "search_depth": settings.tavily_search_depth,
        "max_results": settings.tavily_max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if settings.tavily_search_depth == "advanced":
        search_kwargs["chunks_per_source"] = settings.tavily_chunks_per_source

    logger.debug(f"Searching the web with Tavily for: {query} …")
    response = await client.search(**search_kwargs)
    return parse_tavily_response(response)
