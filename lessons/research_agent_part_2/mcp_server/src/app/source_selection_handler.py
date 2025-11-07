"""Source selection operations and utilities."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..config.constants import (
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_FILE,
    URLS_FROM_GUIDELINES_FOLDER,
)
from ..config.prompts import (
    PROMPT_AUTO_SOURCE_SELECTION,
    PROMPT_SELECT_TOP_SOURCES,
)
from ..config.settings import settings
from ..models.query_models import SourceSelection, TopSourceSelection
from ..utils.llm_utils import get_chat_model

logger = logging.getLogger(__name__)


def parse_perplexity_results(md_text: str) -> Dict[int, Dict[str, str]]:
    """Return mapping {id: {url, query, answer}} extracted from the markdown."""
    _source_block_re = re.compile(
        r"^### Source \[(\d+)]\: (.*?)\n\nQuery: (.*?)\n\nAnswer: (.*?)\n(?:-----|\Z)",
        re.S | re.M,
    )

    results: Dict[int, Dict[str, str]] = {}
    for match in _source_block_re.finditer(md_text):
        src_id = int(match.group(1))
        url = match.group(2).strip()
        query = match.group(3).strip()
        answer = match.group(4).strip()
        results[src_id] = {"url": url, "query": query, "answer": answer}
    return results


def build_sources_data_text(parsed: Dict[int, Dict[str, str]]) -> str:
    """Format sources into the string expected by the prompt."""
    lines: List[str] = []
    for src_id in sorted(parsed):
        entry = parsed[src_id]
        lines.append(f"### Source ID {src_id}: {entry['url']}")
        lines.append("")
        lines.append(f"**Query:** {entry['query']}")
        lines.append(f"**Answer:** {entry['answer']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


async def select_sources(article_guidelines: str, md_results: str) -> List[int]:
    """Use an LLM to select the best subset of sources."""
    parsed_results = parse_perplexity_results(md_results)
    if not parsed_results:
        logger.warning(f"⚠️  No sources found in {PERPLEXITY_RESULTS_FILE} – accepting none.")
        return []

    sources_data_text = build_sources_data_text(parsed_results)

    prompt_text = PROMPT_AUTO_SOURCE_SELECTION.format(
        article_guidelines=article_guidelines or "<none>",
        sources_data=sources_data_text,
    )

    chat_llm = get_chat_model(settings.source_selection_model, SourceSelection)
    logger.debug("Selecting sources to keep")

    try:
        response = await chat_llm.ainvoke(prompt_text)
    except Exception as exc:
        logger.error(f"⚠️ LLM call failed ({exc}). Falling back to accepting all sources.", exc_info=True)
        return sorted(parsed_results.keys())

    if not isinstance(response, SourceSelection):
        logger.error(f"⚠️ LLM call returned unexpected type: {type(response)}")
        return sorted(parsed_results.keys())

    if response.selection_type == "none":
        logger.debug("No sources accepted.")
        return []
    if response.selection_type == "all":
        logger.info("👍 All sources accepted.")
        return sorted(parsed_results.keys())
    # 'specific'
    logger.info(f"👍 {len(response.source_ids)} sources accepted.")
    return [sid for sid in response.source_ids if sid in parsed_results]


def parse_results_selected(md_text: str) -> List[Dict[str, str]]:
    """Return list of dicts with keys url, query, answer from selected results file."""
    _source_block_re = re.compile(
        r"^### Source \[(\d+)]\: (.*?)\n\nQuery: (.*?)\n\nAnswer: (.*?)\n(?:-----|\Z)",
        re.S | re.M,
    )

    sources: List[Dict[str, str]] = []
    for match in _source_block_re.finditer(md_text):
        url = match.group(2).strip()
        query = match.group(3).strip()
        answer = match.group(4).strip()
        sources.append({"url": url, "query": query, "answer": answer})
    return sources


def load_scraped_guideline_context(research_directory: str) -> str:
    """Concatenate all markdown files from URLS_FROM_GUIDELINES_FOLDER as context."""
    ctx_parts: List[str] = []
    dir_path = Path(research_directory) / NOVA_FOLDER / URLS_FROM_GUIDELINES_FOLDER
    if dir_path.exists():
        for md_file in sorted(dir_path.glob("*.md")):
            ctx_parts.append(md_file.read_text(encoding="utf-8"))
    return "\n\n".join(ctx_parts)


async def select_top_sources(
    article_guidelines: str, guideline_ctx: str, md_results_selected: str, max_sources: int = 5
) -> Dict[str, Any]:
    """Select up to max_sources top sources to scrape fully.

    Returns:
        dict: Contains 'selected_urls' (List[str]) and 'reasoning' (str)
    """
    sources = parse_results_selected(md_results_selected)
    if not sources:
        msg = "⚠️  No sources found in perplexity_results_selected.md. Nothing to select."
        logger.warning(msg)
        return {"selected_urls": [], "reasoning": "No sources available for selection."}

    # Build sources data text for prompt
    lines: List[str] = []
    for idx, src in enumerate(sources, 1):
        lines.append(f"### Source {idx}: {src['url']}")
        lines.append(f"Query: {src['query']}")
        lines.append(f"Snippet: {src['answer']}")
        lines.append("---")
    sources_text = "\n".join(lines)

    prompt = PROMPT_SELECT_TOP_SOURCES.format(
        article_guidelines=article_guidelines or "<none>",
        scraped_guideline_ctx=guideline_ctx or "<none>",
        accepted_sources_data=sources_text,
        top_n=max_sources,
    )

    chat_llm = get_chat_model(settings.source_selection_model, TopSourceSelection)
    logger.debug("Selecting top sources to scrape")
    try:
        response = await chat_llm.ainvoke(prompt)
    except Exception as exc:
        msg = f"⚠️  LLM selection failed ({exc}). Returning first {max_sources} sources by default."
        logger.warning(msg)
        return {
            "selected_urls": [s["url"] for s in sources[:max_sources]],
            "reasoning": f"LLM selection failed, falling back to first {max_sources} sources.",
        }

    if not isinstance(response, TopSourceSelection):
        logger.error(f"⚠️ LLM call returned unexpected type: {type(response)}")
        return {
            "selected_urls": [s["url"] for s in sources[:max_sources]],
            "reasoning": f"LLM returned unexpected response type, falling back to first {max_sources} sources.",
        }

    # Ensure max sources limit
    logger.info(f"👍 {len(response.selected_urls)} sources selected to scrape.")
    return {"selected_urls": response.selected_urls[:max_sources], "reasoning": response.reasoning}
