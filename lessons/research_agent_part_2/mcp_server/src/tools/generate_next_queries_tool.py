"""Query generation tool implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..app.generate_queries_handler import generate_queries_with_reasons
from ..config.constants import (
    ARTICLE_GUIDELINE_FILE,
    MARKDOWN_EXTENSION,
    NEXT_QUERIES_FILE,
    NOVA_FOLDER,
    PERPLEXITY_RESULTS_FILE,
    URLS_FROM_GUIDELINES_FOLDER,
)
from ..utils.file_utils import read_file_safe, validate_research_folder

logger = logging.getLogger(__name__)


def write_queries_to_file(next_q_path: Path, queries_and_reasons: List[Tuple[str, str]]) -> None:
    """
    Write the generated queries and reasons to a markdown file.

    Args:
        next_q_path: Path to the output file
        queries_and_reasons: List of tuples containing (query, reason) pairs
    """
    with next_q_path.open("w", encoding="utf-8") as f:
        f.write("### Candidate Web-Search Queries\n\n")
        for idx, (query, reason) in enumerate(queries_and_reasons, 1):
            f.write(f"{idx}. {query}\n")
            f.write(f"Reason: {reason}\n\n")


def format_queries_for_display(queries_and_reasons: List[Tuple[str, str]]) -> str:
    """
    Format the queries and reasons for display in the response message.

    Args:
        queries_and_reasons: List of tuples containing (query, reason) pairs

    Returns:
        Formatted string with all queries and reasons
    """
    formatted_queries = []
    for idx, (query, reason) in enumerate(queries_and_reasons, 1):
        formatted_queries.append(f"{idx}. {query}\nReason: {reason}")

    return "\n\n".join(formatted_queries)


async def generate_next_queries_tool(research_directory: str, n_queries: int = 5) -> Dict[str, Any]:
    """
    Generate candidate web-search queries for the next research round.

    Analyzes the article guidelines, already-scraped content, and existing Perplexity
    results to identify knowledge gaps and propose new web-search questions.
    Each query includes a rationale explaining why it's important for the article.
    Results are saved to next_queries.md in the research directory.

    Args:
        research_directory: Path to the research directory containing article data
        n_queries: Number of queries to generate (default: 5)

    Returns:
        Dict with status, generated queries, and output file path
    """
    logger.debug(f"Generating candidate web-search queries for {research_directory}")

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
    urls_from_guidelines_dir = nova_path / URLS_FROM_GUIDELINES_FOLDER

    article_guidelines = read_file_safe(guidelines_path)
    past_research = read_file_safe(results_path)

    scraped_ctx_parts: List[str] = []
    if urls_from_guidelines_dir.exists():
        for md_file in sorted(urls_from_guidelines_dir.glob(f"*{MARKDOWN_EXTENSION}")):
            scraped_ctx_parts.append(md_file.read_text(encoding="utf-8"))
    scraped_ctx_str = "\n\n".join(scraped_ctx_parts)

    if not article_guidelines:
        logger.warning(f"⚠️  Article guidelines not found at {guidelines_path}. Proceeding anyway.")

    queries_and_reasons = await generate_queries_with_reasons(
        article_guidelines, past_research, scraped_ctx_str, n_queries=n_queries
    )

    # Write to next_queries.md (overwrite)
    next_q_path = nova_path / NEXT_QUERIES_FILE

    # Write queries to file
    write_queries_to_file(next_q_path, queries_and_reasons)

    # Create the formatted queries string for display
    queries_string = format_queries_for_display(queries_and_reasons)

    return {
        "status": "success",
        "queries_count": len(queries_and_reasons),
        "queries": queries_and_reasons,
        "output_path": str(next_q_path.resolve()),
        "message": (
            f"Successfully generated {len(queries_and_reasons)} candidate queries for research folder "
            f"'{research_directory}'. Queries and reasons saved to: "
            f"{next_q_path.relative_to(research_path)}\n\nGenerated Queries:\n\n{queries_string}"
        ),
    }
