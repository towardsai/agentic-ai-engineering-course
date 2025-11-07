"""Scrape research URLs tool implementation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..app.scraping_handler import build_filename, scrape_urls_concurrently
from ..app.youtube_handler import process_youtube_url
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    GUIDELINES_FILENAMES_FILE,
    NOVA_FOLDER,
    URLS_FROM_RESEARCH_FOLDER,
    URLS_TO_SCRAPE_FROM_RESEARCH_FILE,
    YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS,
)
from ..utils.file_utils import read_file_safe, validate_research_folder

logger = logging.getLogger(__name__)


def validate_and_read_urls_file(
    urls_file_path: Path, research_directory: str
) -> tuple[list[str], Dict[str, Any] | None]:
    """
    Validate and read URLs from the research URLs file.

    Checks if the URLs file exists, reads its content, and parses the URLs.
    Returns either the list of URLs or an early return dict for error cases.

    Args:
        urls_file_path: Path to the URLs file
        research_directory: Research directory path for error messages

    Returns:
        Tuple of (urls_list, early_return_dict). If early_return_dict is not None,
        the caller should return it immediately. Otherwise, use the urls_list.
    """
    if not urls_file_path.exists():
        early_return = {
            "status": "success",
            "message": f"No URLs to scrape, file not found: {urls_file_path}",
            "urls_processed": 0,
            "urls_total": 0,
        }
        return [], early_return

    # Read the URLs file (one URL per line)
    try:
        urls_content = read_file_safe(urls_file_path)
        if not urls_content:
            early_return = {
                "status": "success",
                "message": f"No URLs found in {URLS_TO_SCRAPE_FROM_RESEARCH_FILE} in '{research_directory}'",
                "urls_processed": 0,
                "urls_total": 0,
            }
            return [], early_return

        # Split by lines and filter out empty lines
        urls = [url.strip() for url in urls_content.split("\n") if url.strip()]

        if not urls:
            early_return = {
                "status": "success",
                "message": f"No valid URLs found in {URLS_TO_SCRAPE_FROM_RESEARCH_FILE} in '{research_directory}'",
                "urls_processed": 0,
                "urls_total": 0,
            }
            return [], early_return

    except (IOError, OSError) as e:
        msg = f"Error reading {URLS_TO_SCRAPE_FROM_RESEARCH_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    return urls, None


def deduplicate_urls(nova_path: Path, urls: list[str]) -> tuple[list[str], int, int]:
    """
    Deduplicate URLs by checking against previously processed URLs from guidelines file.

    Reads the guidelines filenames JSON file to get already processed URLs and filters
    them out from the input URLs list.

    Args:
        nova_path: Path to the .nova directory
        urls: List of URLs to deduplicate

    Returns:
        Tuple of (filtered_urls, original_count, deduplicated_count)
    """
    # Read previously processed URLs from GUIDELINES_FILENAMES_FILE to deduplicate
    guidelines_json_path = nova_path / GUIDELINES_FILENAMES_FILE
    already_processed_urls = set()

    if guidelines_json_path.exists():
        try:
            with open(guidelines_json_path, "r", encoding="utf-8") as f:
                guidelines_data = json.load(f)

            # Collect URLs from steps 2.2 and 2.3 (other_urls and github_urls)
            other_urls_guidelines = guidelines_data.get("other_urls", [])
            github_urls_guidelines = guidelines_data.get("github_urls", [])
            already_processed_urls.update(other_urls_guidelines)
            already_processed_urls.update(github_urls_guidelines)

        except (IOError, OSError, json.JSONDecodeError) as e:
            msg = f"⚠️ Warning: Could not read {GUIDELINES_FILENAMES_FILE} for deduplication: {e}"
            logger.warning(msg, exc_info=True)

    # Filter out URLs that were already processed
    original_count = len(urls)
    urls_to_process = [url for url in urls if url not in already_processed_urls]
    deduplicated_count = original_count - len(urls_to_process)

    return urls_to_process, original_count, deduplicated_count


def categorize_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorize URLs into YouTube and other URLs.

    Separates the input URLs into two categories:
    - YouTube URLs (containing youtube.com or youtu.be)
    - Other URLs (all remaining URLs)

    Args:
        urls: List of URLs to categorize

    Returns:
        Tuple of (youtube_urls, other_urls)
    """
    youtube_urls = [url for url in urls if "youtube.com" in url or "youtu.be" in url]
    other_urls = [url for url in urls if url not in youtube_urls]

    return youtube_urls, other_urls


def write_scraped_results_to_files(completed_results: list[dict], output_dir: Path) -> tuple[list[str], int]:
    """
    Write scraped results to markdown files and return statistics.

    Args:
        completed_results: List of scraping results from scrape_urls_concurrently
        output_dir: Directory to write the markdown files to

    Returns:
        Tuple of (saved_files_list, successful_scrapes_count)
    """
    saved_files = []
    successful_scrapes = 0
    existing_names: set[str] = set()

    for res in completed_results:
        cleaned_markdown = res.get("markdown", "")
        title = res.get("title", "")
        url = res.get("url", "")

        if res.get("success", False):
            successful_scrapes += 1

        filename = build_filename(title, url, existing_names)
        output_path = output_dir / filename
        output_path.write_text(cleaned_markdown or "")
        saved_files.append(filename)

    return saved_files, successful_scrapes


async def process_and_save_urls(
    other_urls: list[str], youtube_urls: list[str], article_guidelines: str, output_dir: Path, concurrency_limit: int
) -> tuple[list[str], int, list[str]]:
    """
    Process and save both other URLs and YouTube URLs.

    Args:
        other_urls: List of non-YouTube URLs to scrape
        youtube_urls: List of YouTube URLs to transcribe
        article_guidelines: Guidelines content for cleaning scraped data
        output_dir: Directory to save the processed files
        concurrency_limit: Maximum number of concurrent tasks

    Returns:
        Tuple of (saved_files_list, successful_scrapes_count, report_parts_list)
    """
    saved_files = []
    successful_scrapes = 0
    report_parts = []

    # Process OTHER URLs
    if other_urls:
        logger.debug(
            f"Starting scraping of {len(other_urls)} web pages with a concurrency limit of {concurrency_limit}..."
        )

        # Scrape URLs concurrently
        completed_results = await scrape_urls_concurrently(other_urls, article_guidelines, concurrency_limit)

        # Write outputs
        saved_files_batch, successful_scrapes = write_scraped_results_to_files(completed_results, output_dir)
        saved_files.extend(saved_files_batch)

        report_parts.append(f"Scraped {successful_scrapes}/{len(other_urls)} web pages.")

    # Process YOUTUBE URLs
    if youtube_urls:
        logger.debug(f"Starting transcription of {len(youtube_urls)} YouTube video(s)...")
        try:
            yt_semaphore = asyncio.Semaphore(YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS)

            yt_tasks = [process_youtube_url(url, output_dir, yt_semaphore) for url in youtube_urls]
            await asyncio.gather(*yt_tasks)
            report_parts.append(f"Transcribed {len(youtube_urls)} YouTube videos.")
        except Exception as e:
            logger.error(f"Failed to initialize or run YouTube transcription: {e}", exc_info=True)
            report_parts.append(f"Failed to transcribe {len(youtube_urls)} YouTube videos.")

    return saved_files, successful_scrapes, report_parts


async def scrape_research_urls_tool(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape the selected research URLs for full content.

    Reads the URLs from urls_to_scrape_from_research.md and scrapes/cleans each URL's
    full content. The cleaned markdown files are saved to the urls_from_research
    subfolder with appropriate filenames. Deduplicates URLs that were already processed.

    Args:
        research_directory: Path to the research directory containing urls_to_scrape_from_research.md
        concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Scraping research URLs from directory: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Create NOVA_FOLDER directory if it doesn't exist
    nova_path.mkdir(parents=True, exist_ok=True)

    # Look for urls_to_scrape_from_research.md file
    urls_file_path = nova_path / URLS_TO_SCRAPE_FROM_RESEARCH_FILE

    # Validate and read URLs from file
    urls, early_return = validate_and_read_urls_file(urls_file_path, research_directory)
    if early_return is not None:
        return early_return

    # Deduplicate URLs against previously processed ones
    urls_to_process, original_count, deduplicated_count = deduplicate_urls(nova_path, urls)

    # Categorize URLs into YouTube and other types
    youtube_urls, other_urls = categorize_urls(urls_to_process)

    if not youtube_urls and not other_urls:
        return {
            "status": "success",
            "message": f"All {original_count} URLs were already processed. No new URLs to scrape.",
            "urls_processed": 0,
            "urls_total": original_count,
            "deduplicated_count": deduplicated_count,
        }

    # Look for ARTICLE_GUIDELINE_FILE file to get the guidelines content
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    article_guidelines = ""
    if guidelines_path.exists():
        try:
            article_guidelines = read_file_safe(guidelines_path)
        except Exception as e:
            msg = f"Error reading {ARTICLE_GUIDELINE_FILE}: {str(e)}"
            logger.error(msg, exc_info=True)
            raise ValueError(msg) from e
    else:
        logger.warning(f"{ARTICLE_GUIDELINE_FILE} not found in research folder: {research_directory}")

    # Prepare output directory
    output_dir = nova_path / URLS_FROM_RESEARCH_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process and save URLs
    saved_files, successful_scrapes, report_parts = await process_and_save_urls(
        other_urls, youtube_urls, article_guidelines, output_dir, concurrency_limit
    )

    # Final Report
    base_message = (
        f"Processed {len(youtube_urls) + len(other_urls)} new URLs "
        f"from {URLS_TO_SCRAPE_FROM_RESEARCH_FILE} in '{research_directory}'."
    )

    total_urls_processed = len(youtube_urls) + len(other_urls)
    return {
        "status": "success" if len(report_parts) > 0 else "warning",
        "urls_processed": successful_scrapes,
        "urls_total": total_urls_processed,
        "original_urls_count": original_count,
        "deduplicated_count": deduplicated_count,
        "files_saved": len(saved_files),
        "output_directory": str(output_dir.resolve()),
        "saved_files": saved_files,
        "message": f"{base_message} {' '.join(report_parts)}",
    }
