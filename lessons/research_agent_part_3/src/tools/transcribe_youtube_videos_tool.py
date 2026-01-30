"""YouTube video transcription tool implementation."""

import asyncio
import logging
import uuid
from typing import Any, Dict

from ..app.youtube_handler import process_youtube_url
from ..config.constants import YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS
from ..db.models import Article, YouTubeTranscript
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


async def transcribe_youtube_videos_tool(article_guideline_id: str) -> Dict[str, Any]:
    """
    Transcribe YouTube video URLs from article guidelines in the database.

    Reads the list of YouTube URLs from the database (extracted in step 1.3)
    and transcribes each video. The transcription results are saved to the
    database.

    Args:
        article_guideline_id: UUID of the article guideline in the database

    Returns:
        Dict with status, processing results, and counts
    """
    logger.debug(f"Transcribing YouTube videos for article guideline ID: {article_guideline_id}")

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

        # Get URLs from extracted_urls
        youtube_urls = []
        if article.extracted_urls and "youtube_videos_urls" in article.extracted_urls:
            youtube_urls = article.extracted_urls["youtube_videos_urls"]

        if not youtube_urls:
            return {
                "status": "success",
                "videos_processed": 0,
                "videos_total": 0,
                "message": f"No YouTube URLs found in article guideline '{article_guideline_id}'",
            }

        logger.debug(f"Processing {len(youtube_urls)} YouTube URL(s)...")

        # Process YouTube URLs concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS)
        tasks = [process_youtube_url(url, semaphore) for url in youtube_urls]
        results = await asyncio.gather(*tasks)

        # Save results to database
        successful_transcriptions = 0
        failed_transcriptions = 0

        for result in results:
            if result.get("success", False):
                youtube_transcript = YouTubeTranscript(
                    user_id=article.user_id,
                    article_guideline_id=article_uuid,
                    youtube_url=result.get("url", ""),
                    transcription=result.get("transcription", ""),
                )
                session.add(youtube_transcript)
                successful_transcriptions += 1
            else:
                failed_transcriptions += 1

        try:
            await session.commit()
            logger.info(
                f"Successfully saved {successful_transcriptions} YouTube transcripts to database for article {article_guideline_id}"
            )
        except Exception as e:
            msg = f"Error saving YouTube transcripts to database: {e}"
            logger.error(msg, exc_info=True)
            await session.rollback()
            raise ValueError(msg) from e

    total_attempted = len(youtube_urls)
    return {
        "status": "success" if successful_transcriptions > 0 else "warning",
        "videos_processed": successful_transcriptions,
        "videos_failed": failed_transcriptions,
        "videos_total": total_attempted,
        "message": (
            f"Transcribed {successful_transcriptions}/{total_attempted} YouTube videos from article guideline "
            f"'{article_guideline_id}'. Results saved to database."
        ),
    }
