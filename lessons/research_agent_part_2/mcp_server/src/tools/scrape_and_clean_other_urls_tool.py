"""Scrape and clean other URLs tool implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..app.scraping_handler import build_filename, scrape_urls_concurrently
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    GUIDELINES_FILENAMES_FILE,
    NOVA_FOLDER,
    URLS_FROM_GUIDELINES_FOLDER,
)
from ..utils.file_utils import (
    read_file_safe,
    validate_guidelines_file,
    validate_guidelines_filenames_file,
    validate_research_folder,
)

logger = logging.getLogger(__name__)


def write_scraped_results_to_files(completed_results: List[dict], output_dir: Path) -> Tuple[List[str], int]:
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


async def scrape_and_clean_other_urls_tool(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape and clean other URLs from guidelines file in the research folder.

    Reads the guidelines file and scrapes/cleans each URL listed
    under 'other_urls'. The cleaned markdown content is saved to the
    URLS_FROM_GUIDELINES_FOLDER subfolder with appropriate filenames.

    Args:
        research_directory: Path to the research folder containing the guidelines file
        concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Scraping and cleaning other URLs from research folder: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Look for GUIDELINES_FILENAMES_FILE file
    guidelines_json_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(guidelines_json_path)

    # Read the guidelines JSON file
    try:
        with open(guidelines_json_path, "r", encoding="utf-8") as f:
            guidelines_data = json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        msg = f"Error reading {GUIDELINES_FILENAMES_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Get the other_urls list
    other_urls = guidelines_data.get("other_urls", [])

    if not other_urls:
        return {
            "status": "success",
            "message": f"No other URLs found in {GUIDELINES_FILENAMES_FILE} in '{research_directory}'",
            "urls_processed": 0,
            "urls_total": 0,
            "files_saved": 0,
        }

    # Look for ARTICLE_GUIDELINE_FILE file to get the guidelines content
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE

    # Validate the guidelines file
    validate_guidelines_file(guidelines_path)

    # Read the article guidelines
    try:
        article_guidelines = read_file_safe(guidelines_path)
    except (IOError, OSError) as e:
        msg = f"Error reading {ARTICLE_GUIDELINE_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Prepare output directory
    output_dir = nova_path / URLS_FROM_GUIDELINES_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scrape URLs concurrently
    completed_results = await scrape_urls_concurrently(other_urls, article_guidelines, concurrency_limit)

    # Write outputs
    saved_files, successful_scrapes = write_scraped_results_to_files(completed_results, output_dir)

    total_attempted = len(other_urls)
    return {
        "status": "success" if successful_scrapes > 0 else "warning",
        "urls_processed": successful_scrapes,
        "urls_total": total_attempted,
        "files_saved": len(saved_files),
        "output_directory": str(output_dir.resolve()),
        "saved_files": saved_files,
        "message": (
            f"Scraped and cleaned {successful_scrapes}/{total_attempted} other URLs from {GUIDELINES_FILENAMES_FILE} "
            f"in '{research_directory}'.\nSaved {len(saved_files)} files to {URLS_FROM_GUIDELINES_FOLDER} folder: "
            f"{', '.join(saved_files)}"
        ),
    }
