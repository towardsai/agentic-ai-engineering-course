"""Research sources selection tool implementation."""

import logging
import re
from pathlib import Path
from typing import Any, Dict

from ..app.source_selection_handler import select_sources
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_FILE,
    PERPLEXITY_RESULTS_SELECTED_FILE,
    PERPLEXITY_SOURCES_SELECTED_FILE,
)
from ..utils.file_utils import read_file_safe, validate_research_folder

logger = logging.getLogger(__name__)


def extract_selected_blocks_content(selected_ids: list[int], md_results: str) -> str:
    """
    Extract selected source blocks from markdown results and return as content.

    Parses the markdown results to find source blocks by their IDs, filters
    to include only the selected sources, and returns the filtered content
    as a string.

    Args:
        selected_ids: List of source IDs to include in the output
        md_results: Raw markdown content containing all source blocks

    Returns:
        str: Filtered markdown content containing only selected source blocks
    """
    if selected_ids:
        # Split original results into blocks by source
        block_pattern = re.compile(r"(### Source \[(\d+)]:[\s\S]*?)(?=### Source \[|\Z)")
        selected_blocks: list[str] = []
        for block, sid_str in block_pattern.findall(md_results):
            if int(sid_str) in selected_ids:
                selected_blocks.append(block.rstrip())
        content_out = "\n\n".join(selected_blocks) + "\n" if selected_blocks else ""
    else:
        content_out = ""

    return content_out


async def select_research_sources_to_keep_tool(research_directory: str) -> Dict[str, Any]:
    """
    Automatically select high-quality sources from Perplexity results.

    Uses LLM to evaluate each source in perplexity_results.md for trustworthiness,
    authority, and relevance based on the article guidelines. Writes the comma-separated
    IDs of accepted sources to perplexity_sources_selected.md and saves a filtered
    markdown file perplexity_results_selected.md containing only the accepted sources.

    Args:
        research_directory: Path to the research directory containing article guidelines and research data

    Returns:
        Dict with status, selection results, and file paths
    """
    logger.debug(f"Selecting research sources to keep from: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Create NOVA_FOLDER directory if it doesn't exist
    nova_path.mkdir(parents=True, exist_ok=True)

    # Gather context from the research folder
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    results_path = nova_path / PERPLEXITY_RESULTS_FILE

    if not results_path.exists():
        msg = f"{results_path} not found. Run the research round first."
        logger.error(msg)
        raise FileNotFoundError(msg)

    article_guidelines = read_file_safe(guidelines_path)
    md_results = read_file_safe(results_path)

    selected_ids = await select_sources(article_guidelines, md_results)

    # Write the selected IDs (comma-separated) to file.
    selected_ids_path = nova_path / PERPLEXITY_SOURCES_SELECTED_FILE
    with selected_ids_path.open("w", encoding="utf-8") as f:
        f.write(",".join(map(str, selected_ids)))

    # Extract corresponding blocks and write to results_selected file
    results_selected_path = nova_path / PERPLEXITY_RESULTS_SELECTED_FILE
    content_out = extract_selected_blocks_content(selected_ids, md_results)
    with results_selected_path.open("w", encoding="utf-8") as f:
        f.write(content_out)

    return {
        "status": "success",
        "sources_selected_count": len(selected_ids),
        "selected_source_ids": selected_ids,
        "sources_selected_path": str(selected_ids_path.resolve()),
        "results_selected_path": str(results_selected_path.resolve()),
        "message": (
            f"✅ Selected {len(selected_ids)} source(s). IDs written to {selected_ids_path}. "
            f"Filtered results written to {results_selected_path}."
        ),
    }
