"""Source selection operations and utilities."""

import logging
import re
import uuid
from typing import Any, Dict, List

from sqlalchemy import select

from ..config.constants import (
    PERPLEXITY_RESULTS_FILE,
)
from ..config.prompts import (
    PROMPT_AUTO_SOURCE_SELECTION,
    PROMPT_SELECT_TOP_SOURCES,
)
from ..config.settings import settings
from ..db.models import GitHubIngest, LocalFile, ScrapedUrl, YouTubeTranscript
from ..db.session import get_async_session_factory
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


async def load_scraped_guideline_context(article_guideline_id: str) -> str:
    """
    Load scraped guideline context from database.

    Concatenates content from ScrapedUrl, GitHubIngest, YouTubeTranscript, and LocalFile tables.

    Args:
        article_guideline_id: UUID string of the article guideline

    Returns:
        Concatenated context string from all scraped sources
    """
    article_uuid = uuid.UUID(article_guideline_id)
    ctx_parts: List[str] = []

    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        # Get scraped URLs
        scraped_urls_result = await session.execute(select(ScrapedUrl).where(ScrapedUrl.article_guideline_id == article_uuid))
        for scraped_url in scraped_urls_result.scalars():
            if scraped_url.content:
                ctx_parts.append(scraped_url.content)

        # Get GitHub ingests
        github_ingests_result = await session.execute(select(GitHubIngest).where(GitHubIngest.article_guideline_id == article_uuid))
        for ingest in github_ingests_result.scalars():
            if ingest.gitingest_result:
                ctx_parts.append(ingest.gitingest_result)

        # Get YouTube transcripts
        youtube_transcripts_result = await session.execute(
            select(YouTubeTranscript).where(YouTubeTranscript.article_guideline_id == article_uuid)
        )
        for transcript in youtube_transcripts_result.scalars():
            if transcript.transcription:
                ctx_parts.append(transcript.transcription)

        # Get local files
        local_files_result = await session.execute(select(LocalFile).where(LocalFile.article_guideline_id == article_uuid))
        for local_file in local_files_result.scalars():
            if local_file.content:
                ctx_parts.append(local_file.content)

    return "\n\n".join(ctx_parts)


async def select_top_sources(article_guidelines: str, guideline_ctx: str, md_results_selected: str, max_sources: int = 5) -> Dict[str, Any]:
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
