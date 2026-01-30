"""Guidelines URL extraction tool implementation."""

import logging
import uuid
from typing import Any, Dict

from ..app.guideline_extractions_handler import extract_local_paths, extract_urls
from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def extract_guidelines_urls_tool(article_guideline_id: str) -> Dict[str, Any]:
    """
    Extract URLs and local file references from the article guidelines in the database.

    Reads the article guideline from the database and extracts:
    - GitHub URLs
    - YouTube video URLs
    - Other HTTP/HTTPS URLs
    - Local file references

    Results are saved to the database in the extracted_urls column.

    Args:
        article_guideline_id: UUID of the article guideline in the database

    Returns:
        Dict with status, extraction results, and success message
    """
    logger.debug(f"Extracting URLs from article guideline ID: {article_guideline_id}")

    # Convert string to UUID
    try:
        article_uuid = uuid.UUID(article_guideline_id)
    except ValueError as e:
        msg = f"Invalid article guideline ID format: {article_guideline_id}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Query database for article
    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        article = await session.get(Article, article_uuid)

        if not article:
            msg = f"Article with ID '{article_guideline_id}' not found in database"
            logger.error(msg)
            raise ValueError(msg)

        # Read the guidelines text from the database
        text = article.guideline_text

        # Extract URLs and categorize them
        all_urls = extract_urls(text)
        github_source_urls = [u for u in all_urls if "github.com" in u]
        youtube_source_urls = [u for u in all_urls if "youtube.com" in u]
        web_source_urls = [u for u in all_urls if "github.com" not in u and "youtube.com" not in u]

        # Extract local file references
        local_file_paths = extract_local_paths(text)

        # Prepare the data structure - use keys that match what processing tools expect
        data = {
            "github_urls": github_source_urls,
            "youtube_videos_urls": youtube_source_urls,
            "other_urls": web_source_urls,
            "local_file_paths": local_file_paths,
        }

        # Save to database
        article.extracted_urls = data

        try:
            await session.commit()
            logger.info(f"Successfully saved extracted URLs to database for article {article_guideline_id}")
        except Exception as e:
            msg = f"Error saving extracted URLs to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    return {
        "status": "success",
        "github_sources_count": len(github_source_urls),
        "youtube_sources_count": len(youtube_source_urls),
        "web_sources_count": len(web_source_urls),
        "local_files_count": len(local_file_paths),
        "message": (
            f"Successfully extracted URLs from article guideline '{article_guideline_id}'. "
            f"Found {len(github_source_urls)} GitHub URLs, {len(youtube_source_urls)} YouTube video URLs, "
            f"{len(web_source_urls)} other URLs, and {len(local_file_paths)} local file references. "
            f"Results saved to database."
        ),
    }
