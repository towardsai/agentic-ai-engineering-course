"""GitHub URLs processing tool implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..app.github_handler import process_github_url
from ..config.constants import (
    GUIDELINES_FILENAMES_FILE,
    NOVA_FOLDER,
    URLS_FROM_GUIDELINES_CODE_FOLDER,
)
from ..config.settings import settings
from ..utils.file_utils import validate_guidelines_filenames_file, validate_research_folder

logger = logging.getLogger(__name__)


async def process_github_urls_tool(research_directory: str) -> Dict[str, Any]:
    """
    Process GitHub URLs from guidelines file in the research folder.

    Reads the guidelines file and processes each URL listed
    under 'github_urls' using gitingest to extract repository summaries, file trees,
    and content. The results are saved as markdown files in the
    URLS_FROM_GUIDELINES_CODE_FOLDER subfolder.

    Args:
        research_directory: Path to the research folder containing the guidelines file

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Processing GitHub URLs from research folder: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Ensure the NOVA_FOLDER directory exists
    nova_path.mkdir(parents=True, exist_ok=True)

    # Look for GUIDELINES_FILENAMES_FILE file
    metadata_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(metadata_path)

    # Read the guidelines JSON file
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (IOError, OSError, json.JSONDecodeError) as e:
        msg = f"Error reading {GUIDELINES_FILENAMES_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Get the github_urls list
    github_urls: list[str] = data.get("github_urls", [])

    if not github_urls:
        return {
            "status": "success",
            "message": f"No GitHub URLs found in {GUIDELINES_FILENAMES_FILE} in '{research_directory}'",
            "urls_processed": 0,
            "urls_total": 0,
            "files_saved": 0,
        }

    # Prepare output directory
    dest_folder = nova_path / URLS_FROM_GUIDELINES_CODE_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Processing {len(github_urls)} GitHub URLs...")

    # Process GitHub URLs sequentially
    success_count = 0
    for url in github_urls:
        try:
            result = await process_github_url(url, dest_folder, settings.github_token.get_secret_value())
            if result:
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing GitHub URL {url}: {e}")
            continue

    return {
        "status": "success" if success_count > 0 else "warning",
        "urls_processed": success_count,
        "urls_total": len(github_urls),
        "files_saved": success_count,
        "output_directory": str(dest_folder.resolve()),
        "message": (
            f"Processed {success_count}/{len(github_urls)} GitHub URLs from {GUIDELINES_FILENAMES_FILE} "
            f"in '{research_directory}'. Saved markdown summaries to {URLS_FROM_GUIDELINES_CODE_FOLDER} folder."
        ),
    }
