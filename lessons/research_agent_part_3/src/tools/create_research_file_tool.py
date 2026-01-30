"""Research file creation tool implementation."""

import logging
import uuid
from typing import Any, Dict

from sqlalchemy import select

from ..app.perplexity_handler import extract_perplexity_chunks, group_perplexity_by_query
from ..config.settings import settings
from ..db.models import Article, GitHubIngest, ScrapedResearchUrl, ScrapedUrl, YouTubeTranscript
from ..db.session import get_async_session_factory
from ..utils.markdown_utils import build_research_results_section, build_sources_section, combine_research_sections

logger = logging.getLogger(__name__)


def get_download_url(article_guideline_id: str) -> str:
    """
    Get the download URL for the research file based on server configuration.

    Args:
        article_guideline_id: The article guideline ID

    Returns:
        The full URL to download the research.md file
    """
    if settings.server_base_url:
        base_url = settings.server_base_url.rstrip("/")
    else:
        base_url = f"http://{settings.server_host}:{settings.server_port}"
    return f"{base_url}/download_research?article_guideline_id={article_guideline_id}"


async def create_research_file_tool(article_guideline_id: str) -> Dict[str, Any]:
    """
    Generate comprehensive research markdown from database data.

    Combines all research data from the database including filtered Perplexity results,
    scraped guideline sources, GitHub ingests, YouTube transcripts, and scraped research
    sources. The final markdown is saved to the database and a download link is provided.

    Args:
        article_guideline_id: UUID of the article guideline in the database

    Returns:
        Dict with status, download URL, and summary information
    """
    logger.debug(f"Creating research file for article guideline ID: {article_guideline_id}")

    # Convert string to UUID
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as e:
        msg = f"Invalid article guideline ID format: {article_guideline_id}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Query database for article and related data
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)

        if not article:
            msg = f"Article with ID '{article_guideline_id}' not found in database"
            logger.error(msg)
            raise ValueError(msg)

        # Extract and parse perplexity results
        perplexity_results_selected = article.perplexity_results_selected or ""
        if perplexity_results_selected:
            chunks = extract_perplexity_chunks(perplexity_results_selected)
            selected_ids = list(chunks.keys())
        else:
            # Fallback to unfiltered results if selected results don't exist
            perplexity_results = article.perplexity_results or ""
            chunks = extract_perplexity_chunks(perplexity_results)
            selected_ids = list(chunks.keys())
            if not chunks:
                logger.warning(f"No Perplexity results found for article '{article_guideline_id}'")

        # Build Research Results section
        grouped = group_perplexity_by_query(chunks, selected_ids)
        research_results_section = build_research_results_section(grouped)

        # Query ScrapedResearchUrl table for scraped research sources
        scraped_research_result = await session.execute(
            select(ScrapedResearchUrl).where(ScrapedResearchUrl.article_guideline_id == article_uuid)
        )
        scraped_research_urls = scraped_research_result.scalars().all()

        # Build Sources Scraped From Research Results section
        scraped_sources = [(entry.url, entry.content) for entry in scraped_research_urls]
        sources_scraped_section = build_sources_section(
            "## Sources Scraped From Research Results", scraped_sources, "No scraped sources found for research results."
        )

        # Query GitHubIngest table for code sources
        github_ingests_result = await session.execute(select(GitHubIngest).where(GitHubIngest.article_guideline_id == article_uuid))
        github_ingests = github_ingests_result.scalars().all()

        # Build Code Sources section
        code_sources = [(entry.github_url, entry.gitingest_result) for entry in github_ingests]
        code_sources_section = build_sources_section("## Code Sources", code_sources, "No code sources found.")

        # Query YouTubeTranscript table for YouTube transcripts
        youtube_transcripts_result = await session.execute(
            select(YouTubeTranscript).where(YouTubeTranscript.article_guideline_id == article_uuid)
        )
        youtube_transcripts = youtube_transcripts_result.scalars().all()

        # Build YouTube Video Transcripts section
        youtube_sources = [(entry.youtube_url, entry.transcription) for entry in youtube_transcripts]
        youtube_transcripts_section = build_sources_section(
            "## YouTube Video Transcripts", youtube_sources, "No YouTube video transcripts found."
        )

        # Query ScrapedUrl table for additional scraped sources
        scraped_urls_result = await session.execute(select(ScrapedUrl).where(ScrapedUrl.article_guideline_id == article_uuid))
        scraped_urls = scraped_urls_result.scalars().all()

        # Build Additional Sources Scraped section
        additional_sources = [(entry.url, entry.content) for entry in scraped_urls]
        additional_sources_section = build_sources_section(
            "## Additional Sources Scraped", additional_sources, "No additional sources scraped."
        )

        # Combine all sections
        final_md = combine_research_sections(
            research_results_section,
            sources_scraped_section,
            code_sources_section,
            youtube_transcripts_section,
            additional_sources_section,
        )

        # Save to database
        article.research = final_md

        try:
            await session.commit()
            logger.info(f"Successfully saved research file to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving research file to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    # Build download URL
    download_url = get_download_url(article_guideline_id)

    return {
        "status": "success",
        "download_url": download_url,
        "research_results_count": len(grouped),
        "scraped_sources_count": len(scraped_sources),
        "code_sources_count": len(code_sources),
        "youtube_transcripts_count": len(youtube_sources),
        "additional_sources_count": len(additional_sources),
        "message": (
            f"✅ Generated research markdown and saved to database for article '{article_guideline_id}'.\n"
            f"Download URL: {download_url}\n"
            f"Open this URL in your browser to download the research.md file."
        ),
    }
