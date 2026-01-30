"""Query generation tool implementation."""

import logging
import uuid
from typing import Any, Dict, List, Tuple

from sqlalchemy import select

from ..app.generate_queries_handler import generate_queries_with_reasons
from ..db.models import Article, GitHubIngest, LocalFile, ScrapedUrl, YouTubeTranscript
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


def format_queries_for_display(queries_and_reasons: List[Tuple[str, str]]) -> str:
    """
    Format the queries and reasons for display in the response message.

    Args:
        queries_and_reasons: List of tuples containing (query, reason) pairs

    Returns:
        Formatted string with all queries and reasons
    """
    formatted_queries = []
    for idx, (query, reason) in enumerate(queries_and_reasons, 1):
        formatted_queries.append(f"{idx}. {query}\nReason: {reason}")

    return "\n\n".join(formatted_queries)


async def generate_next_queries_tool(article_guideline_id: str, n_queries: int = 5) -> Dict[str, Any]:
    """
    Generate candidate web-search queries for the next research round.

    Analyzes the article guidelines, already-scraped content from the database,
    and existing Perplexity results to identify knowledge gaps and propose new
    web-search questions. Each query includes a rationale.

    Args:
        article_guideline_id: UUID of the article guideline in the database
        n_queries: Number of queries to generate (default: 5)

    Returns:
        Dict with status, generated queries, and counts
    """
    logger.debug(f"Generating candidate web-search queries for article guideline ID: {article_guideline_id}")

    # Convert string to UUID
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as e:
        msg = f"Invalid article guideline ID format: {article_guideline_id}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Query database for article and scraped content
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)

        if not article:
            msg = f"Article with ID '{article_guideline_id}' not found in database"
            logger.error(msg)
            raise ValueError(msg)

        # Get article guidelines
        article_guidelines = article.guideline_text or ""

        # Get past research (perplexity results)
        past_research = article.perplexity_results or ""

        # Gather scraped content from all database tables
        scraped_parts: List[str] = []

        # Get scraped URLs
        scraped_urls_result = await session.execute(select(ScrapedUrl).where(ScrapedUrl.article_guideline_id == article_uuid))
        for scraped_url in scraped_urls_result.scalars():
            if scraped_url.content:
                scraped_parts.append(scraped_url.content)

        # Get GitHub ingests
        github_ingests_result = await session.execute(select(GitHubIngest).where(GitHubIngest.article_guideline_id == article_uuid))
        for ingest in github_ingests_result.scalars():
            if ingest.gitingest_result:
                scraped_parts.append(ingest.gitingest_result)

        # Get YouTube transcripts
        youtube_transcripts_result = await session.execute(
            select(YouTubeTranscript).where(YouTubeTranscript.article_guideline_id == article_uuid)
        )
        for transcript in youtube_transcripts_result.scalars():
            if transcript.transcription:
                scraped_parts.append(transcript.transcription)

        # Get local files
        local_files_result = await session.execute(select(LocalFile).where(LocalFile.article_guideline_id == article_uuid))
        for local_file in local_files_result.scalars():
            if local_file.content:
                scraped_parts.append(local_file.content)

        scraped_ctx_str = "\n\n".join(scraped_parts)

        if not article_guidelines:
            logger.warning(f"⚠️  Article guidelines empty for article {article_guideline_id}. Proceeding anyway.")

        # Generate queries with reasons
        queries_and_reasons = await generate_queries_with_reasons(article_guidelines, past_research, scraped_ctx_str, n_queries=n_queries)

        # Create the formatted queries string for display
        queries_string = format_queries_for_display(queries_and_reasons)

    return {
        "status": "success",
        "queries_count": len(queries_and_reasons),
        "queries": queries_and_reasons,
        "message": (
            f"Successfully generated {len(queries_and_reasons)} candidate queries for article guideline "
            f"'{article_guideline_id}'.\n\nGenerated Queries:\n\n{queries_string}"
        ),
    }
