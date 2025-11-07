"""Research file creation tool implementation."""

import logging
from pathlib import Path
from typing import Any, Dict

from ..app.perplexity_handler import extract_perplexity_chunks, group_perplexity_by_query
from ..config.constants import (
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_FILE,
    PERPLEXITY_RESULTS_SELECTED_FILE,
    RESEARCH_MD_FILE,
    URLS_FROM_GUIDELINES_CODE_FOLDER,
    URLS_FROM_GUIDELINES_FOLDER,
    URLS_FROM_GUIDELINES_YOUTUBE_FOLDER,
    URLS_FROM_RESEARCH_FOLDER,
)
from ..utils.file_utils import (
    collect_directory_markdowns,
    collect_directory_markdowns_with_titles,
    read_file_safe,
    validate_research_folder,
)
from ..utils.markdown_utils import build_research_results_section, build_sources_section, combine_research_sections

logger = logging.getLogger(__name__)


def create_research_file_tool(research_directory: str) -> Dict[str, Any]:
    """
    Generate comprehensive research.md file from all research data.

    Combines all research data including filtered Perplexity results, scraped guideline
    sources, and full research sources into a comprehensive research.md file. The file
    is organized into sections with collapsible blocks for easy navigation.

    Args:
        research_directory: Path to the research directory containing all research data

    Returns:
        Dict with status, generated file path, and summary information
    """
    logger.debug(f"Creating research files for directory: {research_directory}")

    # Convert to Path object
    article_dir = Path(research_directory)
    nova_dir = article_dir / NOVA_FOLDER

    # Validate research folder exists
    validate_research_folder(article_dir)

    # Paths
    selected_results_file = nova_dir / PERPLEXITY_RESULTS_SELECTED_FILE
    original_results_file = nova_dir / PERPLEXITY_RESULTS_FILE

    urls_from_research_dir = nova_dir / URLS_FROM_RESEARCH_FOLDER
    code_sources_dir = nova_dir / URLS_FROM_GUIDELINES_CODE_FOLDER
    additional_sources_dir = nova_dir / URLS_FROM_GUIDELINES_FOLDER
    youtube_transcripts_dir = nova_dir / URLS_FROM_GUIDELINES_YOUTUBE_FOLDER

    # Load and parse perplexity results
    if selected_results_file.exists():
        # Use the already-filtered results directly
        results_md = read_file_safe(selected_results_file)
        chunks = extract_perplexity_chunks(results_md)
        selected_ids = list(chunks.keys())  # all chunks are accepted
    else:
        # Fallback to legacy behaviour (all sources in PERPLEXITY_RESULTS_FILE)
        results_md = read_file_safe(original_results_file)
        if not results_md:
            logger.warning(f"File not found or empty: {original_results_file}")
        chunks = extract_perplexity_chunks(results_md)
        selected_ids = list(chunks.keys())

    # Build Research Results section
    grouped = group_perplexity_by_query(chunks, selected_ids)
    research_results_section = build_research_results_section(grouped)

    # Build Sources Scraped From Research Results section
    scraped_sources = collect_directory_markdowns_with_titles(urls_from_research_dir)
    sources_scraped_section = build_sources_section(
        "## Sources Scraped From Research Results", scraped_sources, "No scraped sources found for research results."
    )

    # Build Code Sources section
    code_sources = collect_directory_markdowns_with_titles(code_sources_dir)
    code_sources_section = build_sources_section("## Code Sources", code_sources, "No code sources found.")

    # Build YouTube Video Transcripts section
    youtube_sources = collect_directory_markdowns_with_titles(youtube_transcripts_dir)
    youtube_transcripts_section = build_sources_section(
        "## YouTube Video Transcripts", youtube_sources, "No YouTube video transcripts found."
    )

    # Build Additional Sources Scraped section
    additional_sources = collect_directory_markdowns(additional_sources_dir)
    additional_sources_section = build_sources_section(
        "## Additional Sources Scraped", additional_sources, "No additional sources scraped."
    )

    # Combine all sections
    final_md = combine_research_sections(
        research_results_section,
        sources_scraped_section,
        code_sources_section,
        youtube_transcripts_section,
        additional_sources_section,
    )

    # Write markdown output
    md_output_path = article_dir / RESEARCH_MD_FILE
    md_output_path.write_text(final_md, encoding="utf-8")

    logger.debug(f"Generated {md_output_path.resolve()}")

    return {
        "status": "success",
        "markdown_file": str(md_output_path.resolve()),
        "research_results_count": len(grouped),
        "scraped_sources_count": len(scraped_sources),
        "code_sources_count": len(code_sources),
        "youtube_transcripts_count": len(youtube_sources),
        "additional_sources_count": len(additional_sources),
        "message": (f"✅ Generated research markdown file:\n  - {md_output_path.relative_to(article_dir)}"),
    }
