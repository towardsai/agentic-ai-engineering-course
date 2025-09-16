"""YouTube transcription operations and utilities."""

import asyncio
import logging
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from google import genai
from google.genai import errors, types
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.constants import (
    YOUTUBE_TRANSCRIPTION_MAX_RETRIES,
    YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MAX_SECONDS,
    YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MIN_SECONDS,
)
from ..config.prompts import PROMPT_YOUTUBE_TRANSCRIPTION
from ..config.settings import settings
from ..utils.opik_utils import track_genai_client

logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type(errors.ServerError),
    wait=wait_exponential(
        multiplier=1, min=YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MIN_SECONDS, max=YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MAX_SECONDS
    ),
    stop=stop_after_attempt(YOUTUBE_TRANSCRIPTION_MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def transcribe_youtube(
    url: str,
    output_path: Path,
    timestamp: int = 30,
) -> None:
    """
    Transcribes a public YouTube video using a Gemini model and saves the
    result to a file.

    Args:
        url: The public URL of the YouTube video.
        output_path: The path to save the transcription markdown file.
        timestamp: The interval in seconds for inserting timestamps in the
                   transcription.
    """
    # Create client internally using settings and track with Opik if configured
    base_client = genai.Client(api_key=settings.google_api_key.get_secret_value())
    client = track_genai_client(base_client)
    model_name = settings.youtube_transcription_model

    prompt = PROMPT_YOUTUBE_TRANSCRIPTION.format(timestamp=timestamp)

    parts: list[types.Part] = [
        types.Part(
            file_data=types.FileData(file_uri=url)  # YouTube URL - no download needed
        ),
        types.Part(text=prompt),
    ]

    logger.info(f"⏳ Processing transcription request for {url}, it may take a while.")
    start_time = time.monotonic()
    try:
        response: types.GenerateContentResponse = await client.aio.models.generate_content(
            model=model_name,
            contents=types.Content(parts=parts),
        )
    except errors.APIError as e:
        if isinstance(e, errors.ServerError):
            logger.warning(f"Server error for {url}, re-raising to trigger retry. Error: {e}", exc_info=True)
            raise

        msg = f"API Error during transcription for {url}: {e}"
        logger.error(msg, exc_info=True)
        output_path.write_text(msg, encoding="utf-8")
        logger.info(f"📄 Error details saved to {output_path}")
        return
    except Exception as e:
        msg = f"An unexpected error occurred for {url}: {e}"
        logger.error(msg, exc_info=True)
        output_path.write_text(msg, encoding="utf-8")
        logger.info(f"📄 Error details saved to {output_path}")
        return
    elapsed_time = time.monotonic() - start_time

    logger.info(f"✅ Transcription for {url} finished in {elapsed_time:.2f} seconds.")

    if not response.text:
        msg = f"Could not generate transcription for {url}.\n\nFull API response:\n" + str(response)
        logger.error(msg, exc_info=True)
        output_path.write_text(msg, encoding="utf-8")
        logger.info(f"📄 Error details saved to {output_path}")
        return

    output_path.write_text(response.text, encoding="utf-8")
    logger.debug(f"📄 Transcription saved to {output_path}")


def get_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from various URL formats.

    Args:
        url: The YouTube URL.

    Returns:
        The video ID, or None if it cannot be parsed.
    """
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")
    return None


async def process_youtube_url(
    url: str,
    dest_folder: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Processes a single YouTube URL by generating a filename and initiating
    the transcription.

    Args:
        url: The YouTube URL to process.
        dest_folder: The directory where the transcription file should be saved.
        semaphore: The asyncio.Semaphore to limit concurrency.
    """
    video_id = get_video_id(url)
    if not video_id:
        sanitized_url = url.replace("https://", "").replace("http://", "").replace("/", "_")
        logger.warning(f"⚠️ Could not extract video ID from URL: {url}. Using fallback name: {sanitized_url}")
        video_id = sanitized_url

    output_path = dest_folder / f"{video_id}.md"

    async with semaphore:
        await transcribe_youtube(
            url=url,
            output_path=output_path,
        )
