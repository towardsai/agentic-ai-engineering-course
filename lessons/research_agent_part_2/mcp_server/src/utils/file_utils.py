"""File and directory operations utilities."""

import logging
from pathlib import Path
from typing import List, Tuple

from ..config.constants import MARKDOWN_EXTENSION
from .markdown_utils import get_first_line_title

logger = logging.getLogger(__name__)


def read_file_safe(path: Path) -> str:
    """Return file content or empty string if the file doesn't exist."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return ""
    except (IOError, OSError) as e:
        logger.error(f"Error reading file {path}: {e}")
        return ""


def validate_research_folder(research_path: Path) -> None:
    """
    Validate that the research folder exists and is a directory.

    Args:
        research_path: Path to the research folder to validate

    Raises:
        ValueError: If the research folder doesn't exist or is not a directory
    """
    if not research_path.exists():
        msg = f"Research folder does not exist: {research_path}"
        logger.error(msg)
        raise ValueError(msg)

    if not research_path.is_dir():
        msg = f"Path is not a directory: {research_path}"
        logger.error(msg)
        raise ValueError(msg)


def validate_guidelines_file(guidelines_path: Path) -> None:
    """
    Validate that the guidelines file exists and is readable.

    Args:
        guidelines_path: Path to the guidelines file to validate

    Raises:
        ValueError: If the guidelines file does not exist or is not readable
    """
    if not guidelines_path.exists():
        msg = f"Guidelines file does not exist: {guidelines_path}"
        logger.error(msg)
        raise ValueError(msg)

    if not guidelines_path.is_file():
        msg = f"Path is not a file: {guidelines_path}"
        logger.error(msg)
        raise ValueError(msg)


def validate_guidelines_filenames_file(filenames_path: Path) -> None:
    """
    Validate that the guidelines filenames file exists and is readable.

    Args:
        filenames_path: Path to the guidelines filenames file to validate

    Raises:
        ValueError: If the guidelines filenames file does not exist or is not readable
    """
    if not filenames_path.exists():
        msg = f"Guidelines filenames file does not exist: {filenames_path}"
        logger.error(msg)
        raise ValueError(msg)

    if not filenames_path.is_file():
        msg = f"Path is not a file: {filenames_path}"
        logger.error(msg)
        raise ValueError(msg)


def validate_perplexity_results_selected_file(results_selected_path: Path) -> None:
    """
    Validate that the perplexity results selected file exists and is readable.

    Args:
        results_selected_path: Path to the perplexity results selected file to validate

    Raises:
        FileNotFoundError: If the perplexity results selected file does not exist
        ValueError: If the path is not a file
    """
    if not results_selected_path.exists():
        msg = f"Perplexity results selected file does not exist: {results_selected_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not results_selected_path.is_file():
        msg = f"Path is not a file: {results_selected_path}"
        logger.error(msg)
        raise ValueError(msg)


def collect_directory_markdowns(dir_path: Path) -> List[Tuple[str, str]]:
    """Return list of (title, content) for every .md file in directory, sorted alphabetically."""
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    items: List[Tuple[str, str]] = []
    for file in sorted(dir_path.glob(f"*{MARKDOWN_EXTENSION}")):
        title = file.stem  # filename without extension
        content = read_file_safe(file)
        if content:
            items.append((title, content))
    return items


def collect_directory_markdowns_with_titles(dir_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (title, content) for every .md file in directory, sorted alphabetically.
    Title is extracted from the first line of the file, with leading '#' removed.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    items = []
    for file in sorted(dir_path.glob(f"*{MARKDOWN_EXTENSION}")):
        content = read_file_safe(file)
        if content:
            title = get_first_line_title(content)
            items.append((title, content))
    return items
