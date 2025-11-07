"""Perplexity API operations and utilities."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from ..config.constants import PERPLEXITY_RESULTS_FILE
from ..config.prompts import PROMPT_WEB_SEARCH
from ..utils.llm_utils import get_chat_model

logger = logging.getLogger(__name__)


class SourceAnswer(BaseModel):
    """A single source answer with URL and content."""

    url: str = Field(description="The URL of the source")
    answer: str = Field(description="The detailed answer extracted from that source")


class PerplexityResponse(BaseModel):
    """Structured response from Perplexity search containing multiple sources."""

    sources: List[SourceAnswer] = Field(description="List of sources with their answers")


async def run_perplexity_search(query: str) -> Tuple[str, Dict[int, str], Dict[int, str]]:
    """Run a Perplexity Sonar-Pro search and return full answer + parsed sections."""

    # run perplexity search with structured output
    llm = get_chat_model("perplexity", PerplexityResponse)

    prompt = PROMPT_WEB_SEARCH.format(query=query)
    logger.debug(f"Searching web for: {query} …")

    response = await llm.ainvoke(prompt)

    # Convert structured response to the expected format
    answer_by_source = {}
    citations = {}
    for i, source in enumerate(response.sources, 1):
        answer_by_source[i] = source.answer
        citations[i] = source.url

    # Create a readable full answer string for compatibility
    full_answer_lines = []
    for i, source in enumerate(response.sources, 1):
        full_answer_lines.append(f"### [{i}]: {source.url}")
        full_answer_lines.append(source.answer)
        full_answer_lines.append("")
    full_answer = "\n".join(full_answer_lines)

    return full_answer, answer_by_source, citations


def compute_next_source_id(results_path: Path) -> int:
    """Compute the next available source ID by finding the highest existing ID."""
    if not results_path.exists():
        return 1
    pattern = re.compile(r"### Source \[(\d+)\]:")
    max_id = 0
    for line in results_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def append_perplexity_results(
    results_path: Path,
    query: str,
    answer_by_source: Dict[int, str],
    citations: Dict[int, str],
    starting_id: int,
) -> int:
    """Append results for a single query. Returns next available global id."""
    with results_path.open("a", encoding="utf-8") as f:
        last_id = starting_id
        for local_id, url in citations.items():
            content = answer_by_source.get(local_id, "").strip()
            if not content:
                continue
            f.write(f"### Source [{last_id}]: {url}\n\n")
            f.write(f"Query: {query}\n\n")
            f.write(f"Answer: {content}\n\n")
            f.write("-----\n\n")
            last_id += 1
    return last_id


async def run_queries(article_id: str, queries: List[str]) -> None:
    if not queries:
        logger.warning("⚠️  No queries supplied – nothing to do.")
        return

    base_dir = Path("articles") / article_id
    results_path = base_dir / PERPLEXITY_RESULTS_FILE

    # Ensure output directory and file exist
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.touch(exist_ok=True)

    next_global_id = compute_next_source_id(results_path)

    logger.debug(f"Executing {len(queries)} Perplexity queries concurrently...")
    tasks = [run_perplexity_search(query) for query in queries]
    search_results = await asyncio.gather(*tasks)
    logger.debug("...all Perplexity queries finished. Appending results.")

    for query, (_, answer_by_source, citations) in zip(queries, search_results):
        if citations:
            next_global_id = append_perplexity_results(
                results_path,
                query,
                answer_by_source,
                citations,
                next_global_id,
            )
            logger.debug(f"Appended results for query: '{query}' (added {len(citations)} source section(s)).")

    logger.debug(f"\n✅ Completed Perplexity research round. Results saved to {results_path}.")


def extract_perplexity_chunks(markdown: str) -> Dict[int, str]:
    """Return a map {source_id: markdown_block} for each source block in the file.

    Each block starts with a heading like:
    `### Source [<id>]: <url>`
    and ends before the next "### Source [" or EOF.
    """
    pattern = re.compile(r"^### Source \[(\d+)]:(?:.*)$", re.MULTILINE)
    chunks: Dict[int, str] = {}

    matches = list(pattern.finditer(markdown))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        source_id = int(match.group(1))
        chunks[source_id] = markdown[start:end].strip()
    return chunks


def group_perplexity_by_query(chunks: Dict[int, str], selected_ids: List[int]) -> Dict[str, List[str]]:
    """
    Group source blocks by their query string. Returns {query: [block, ...]}.
    Only includes blocks whose source_id is in selected_ids.
    """
    query_to_blocks = {}
    for src_id in selected_ids:
        block = chunks.get(src_id)
        if not block:
            continue
        query_match = re.search(r"^Query:\s*(.+)$", block, re.MULTILINE)
        query = query_match.group(1).strip() if query_match else f"Query for Source [{src_id}]"
        query_to_blocks.setdefault(query, []).append(block)
    return query_to_blocks
