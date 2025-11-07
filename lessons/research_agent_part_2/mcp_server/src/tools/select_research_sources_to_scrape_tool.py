"""Research sources to scrape selection tool implementation."""

import logging
from pathlib import Path
from typing import Any, Dict

from ..app.source_selection_handler import load_scraped_guideline_context, select_top_sources
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_SELECTED_FILE,
    URLS_TO_SCRAPE_FROM_RESEARCH_FILE,
)
from ..utils.file_utils import read_file_safe, validate_perplexity_results_selected_file, validate_research_folder

logger = logging.getLogger(__name__)


async def select_research_sources_to_scrape_tool(research_directory: str, max_sources: int = 5) -> Dict[str, Any]:
    """
    Select up to max_sources priority research sources to scrape in full.

    Analyzes the filtered Perplexity results together with the article guidelines and
    the material already scraped from guideline URLs, then chooses up to max_sources diverse,
    authoritative sources whose full content will add most value. The chosen URLs are
    written (one per line) to urls_to_scrape_from_research.md.

    Args:
        research_directory: Path to the research directory containing all research data
        max_sources: Maximum number of sources to select (default: 5)

    Returns:
        Dict with status, selection results, file paths, and reasoning for the selection
    """
    logger.debug(f"Selecting research sources to scrape from: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Create NOVA_FOLDER directory if it doesn't exist
    nova_path.mkdir(parents=True, exist_ok=True)

    # Gather context from the research folder
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    results_selected_path = nova_path / PERPLEXITY_RESULTS_SELECTED_FILE
    urls_out_path = nova_path / URLS_TO_SCRAPE_FROM_RESEARCH_FILE

    validate_perplexity_results_selected_file(results_selected_path)

    article_guidelines = read_file_safe(guidelines_path)
    md_results_selected = read_file_safe(results_selected_path)
    guideline_ctx = load_scraped_guideline_context(research_directory)

    selection_result = await select_top_sources(article_guidelines, guideline_ctx, md_results_selected, max_sources)
    top_urls = selection_result["selected_urls"]
    reasoning = selection_result["reasoning"]

    # Write URLs one per line
    urls_out_path.parent.mkdir(parents=True, exist_ok=True)
    with urls_out_path.open("w", encoding="utf-8") as f:
        for url in top_urls:
            f.write(url + "\n")

    return {
        "status": "success",
        "urls_selected_count": len(top_urls),
        "selected_urls": top_urls,
        "selection_reasoning": reasoning,
        "urls_output_path": str(urls_out_path.resolve()),
        "message": f"✅ Saved {len(top_urls)} URL(s) to scrape to {urls_out_path}.\nReasoning: {reasoning}",
    }
