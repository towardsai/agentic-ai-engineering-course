"""Utility functions for research operations."""

from ..config.constants import (
    YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS,
    YOUTUBE_TRANSCRIPTION_MAX_RETRIES,
    YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MAX_SECONDS,
    YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MIN_SECONDS,
)
from ..utils.file_utils import collect_directory_markdowns, collect_directory_markdowns_with_titles
from ..utils.markdown_utils import get_first_line_title, markdown_collapsible
from .generate_queries_handler import PROMPT_GENERATE_QUERIES_AND_REASONS, generate_queries_with_reasons
from .github_handler import process_github_url
from .guideline_extractions_handler import extract_local_paths, extract_urls
from .notebook_handler import NotebookToMarkdownConverter
from .perplexity_handler import (
    PROMPT_WEB_SEARCH,
    append_perplexity_results,
    compute_next_source_id,
    extract_perplexity_chunks,
    group_perplexity_by_query,
    run_perplexity_search,
    run_queries,
)
from .scraping_handler import (
    PROMPT_CLEAN_MARKDOWN,
    build_filename,
    clean_markdown,
    convert_markdown_images_to_urls,
    scrape_and_clean,
    scrape_url,
    slugify,
)
from .source_selection_handler import (
    PROMPT_AUTO_SOURCE_SELECTION,
    PROMPT_SELECT_TOP_SOURCES,
    build_sources_data_text,
    load_scraped_guideline_context,
    parse_perplexity_results,
    parse_results_selected,
    select_sources,
    select_top_sources,
)
from .youtube_handler import (
    get_video_id,
    process_youtube_url,
    transcribe_youtube,
)

__all__ = [
    # File operations (business logic only)
    "collect_directory_markdowns",
    "collect_directory_markdowns_with_titles",
    # Text processing
    "extract_urls",
    "extract_local_paths",
    # Markdown processing
    "markdown_collapsible",
    "get_first_line_title",
    # LLM operations
    "generate_queries_with_reasons",
    "PROMPT_GENERATE_QUERIES_AND_REASONS",
    # Notebook processing
    "NotebookToMarkdownConverter",
    # Perplexity operations
    "run_perplexity_search",
    "compute_next_source_id",
    "append_perplexity_results",
    "run_queries",
    "extract_perplexity_chunks",
    "group_perplexity_by_query",
    "PROMPT_WEB_SEARCH",
    # Scraping utilities
    "slugify",
    "build_filename",
    "scrape_url",
    "convert_markdown_images_to_urls",
    "clean_markdown",
    "scrape_and_clean",
    "PROMPT_CLEAN_MARKDOWN",
    # GitHub utilities
    "process_github_url",
    # Source selection operations
    "parse_perplexity_results",
    "build_sources_data_text",
    "select_sources",
    "parse_results_selected",
    "load_scraped_guideline_context",
    "select_top_sources",
    "PROMPT_AUTO_SOURCE_SELECTION",
    "PROMPT_SELECT_TOP_SOURCES",
    # YouTube operations
    "transcribe_youtube",
    "get_video_id",
    "process_youtube_url",
    "YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS",
    "YOUTUBE_TRANSCRIPTION_MAX_RETRIES",
    "YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MIN_SECONDS",
    "YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MAX_SECONDS",
]
