"""Guidelines URL extraction tool implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..app.guideline_extractions_handler import extract_local_paths, extract_urls
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    GUIDELINES_FILENAMES_FILE,
    NOVA_FOLDER,
)
from ..utils.file_utils import validate_guidelines_file, validate_research_folder

logger = logging.getLogger(__name__)


def extract_guidelines_urls_tool(research_folder: str) -> Dict[str, Any]:
    """
    Extract URLs and local file references from the article guidelines in the research folder.

    Reads the ARTICLE_GUIDELINE_FILE file and extracts:
    - GitHub URLs
    - YouTube video URLs
    - Other HTTP/HTTPS URLs
    - Local file references

    Results are saved to GUIDELINES_FILENAMES_FILE in the research folder.

    Args:
        research_folder: Path to the research folder containing ARTICLE_GUIDELINE_FILE

    Returns:
        Dict with status, extraction results, and output file path
    """
    logger.debug(f"Extracting URLs from article guidelines in: {research_folder}")

    # Convert to Path object
    research_path = Path(research_folder)
    nova_path = research_path / NOVA_FOLDER
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE

    # Validate folders and files
    validate_research_folder(research_path)
    validate_guidelines_file(guidelines_path)

    # Create NOVA_FOLDER directory if it doesn't exist
    nova_path.mkdir(parents=True, exist_ok=True)

    # Read the guidelines file
    try:
        text = guidelines_path.read_text(encoding="utf-8")
    except (IOError, OSError) as e:
        msg = f"Error reading {ARTICLE_GUIDELINE_FILE}: {e}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

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

    # Write to GUIDELINES_FILENAMES_FILE in the research folder
    output_path = nova_path / GUIDELINES_FILENAMES_FILE

    try:
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (IOError, OSError, TypeError) as e:
        msg = f"Error writing {GUIDELINES_FILENAMES_FILE}: {e}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    return {
        "status": "success",
        "github_sources_count": len(github_source_urls),
        "youtube_sources_count": len(youtube_source_urls),
        "web_sources_count": len(web_source_urls),
        "local_files_count": len(local_file_paths),
        "output_path": str(output_path.resolve()),
        "message": (
            f"Successfully extracted URLs from article guidelines in '{research_folder}'. "
            f"Found {len(github_source_urls)} GitHub URLs, {len(youtube_source_urls)} YouTube videos URLs, "
            f"{len(web_source_urls)} other URLs, and {len(local_file_paths)} local file references. "
            f"Results saved to: {output_path.resolve()}"
        ),
    }
