"""MCP Tools registration for research operations."""

from typing import Any, Dict

from fastmcp import FastMCP

from ..tools import (
    create_research_file_tool,
    extract_guidelines_urls_tool,
    generate_next_queries_tool,
    process_github_urls_tool,
    process_local_files_tool,
    run_perplexity_research_tool,
    scrape_and_clean_other_urls_tool,
    scrape_research_urls_tool,
    select_research_sources_to_keep_tool,
    select_research_sources_to_scrape_tool,
    transcribe_youtube_videos_tool,
)


def register_mcp_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with the server instance."""

    # ============================================================================
    # URL AND FILE EXTRACTION TOOLS
    # ============================================================================

    @mcp.tool()
    async def extract_guidelines_urls(research_directory: str) -> Dict[str, Any]:
        """
        Extract URLs and local file references from article guidelines.

        Reads the ARTICLE_GUIDELINE_FILE file in the research directory and extracts:
        - GitHub URLs
        - Other HTTP/HTTPS URLs
        - Local file references (files mentioned in quotes with extensions)

        Results are saved to GUIDELINES_FILENAMES_FILE in the research directory.

        Args:
            research_directory: Path to the research directory containing ARTICLE_GUIDELINE_FILE

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - github_sources_count: Number of GitHub URLs found
                - youtube_sources_count: Number of YouTube URLs found
                - web_sources_count: Number of other web URLs found
                - local_files_count: Number of local file references found
                - output_path: Path to the generated GUIDELINES_FILENAMES_FILE
                - message: Human-readable success message
        """
        result = extract_guidelines_urls_tool(research_directory)
        return result

    @mcp.tool()
    async def process_local_files(research_directory: str) -> Dict[str, Any]:
        """
        Process local files referenced in the article guidelines.

        Reads the GUIDELINES_FILENAMES_FILE file and copies each referenced local file
        to the LOCAL_FILES_FROM_RESEARCH_FOLDER subfolder. Path separators in filenames are
        replaced with underscores to avoid creating nested folders.

        Args:
            research_directory: Path to the research directory containing GUIDELINES_FILENAMES_FILE

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - files_processed: List of successfully processed files
                - files_failed: List of files that failed to process
                - total_files: Total number of files processed
                - files_copied_count: Number of files successfully copied
                - files_failed_count: Number of files that failed to copy
                - output_directory: Path to the output directory
                - message: Human-readable success message with processing results
        """
        result = process_local_files_tool(research_directory)
        return result

    # ============================================================================
    # WEB SCRAPING AND CONTENT PROCESSING TOOLS
    # ============================================================================

    @mcp.tool()
    async def scrape_and_clean_other_urls(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
        """
        Scrape and clean other URLs from GUIDELINES_FILENAMES_FILE.

        Reads the GUIDELINES_FILENAMES_FILE file and scrapes/cleans each URL listed
        under 'other_urls'. The cleaned markdown content is saved to the
        URLS_FROM_GUIDELINES_FOLDER subfolder with appropriate filenames.

        Args:
            research_directory: Path to the research directory containing GUIDELINES_FILENAMES_FILE
            concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - urls_processed: List of successfully processed URLs
                - urls_failed: List of URLs that failed to process
                - total_urls: Total number of URLs processed
                - successful_urls_count: Number of URLs successfully scraped
                - failed_urls_count: Number of URLs that failed to scrape
                - output_directory: Path to the output directory
                - message: Human-readable success message with processing results
        """
        result = await scrape_and_clean_other_urls_tool(research_directory, concurrency_limit)
        return result

    @mcp.tool()
    async def process_github_urls(research_directory: str) -> Dict[str, Any]:
        """
        Process GitHub URLs from GUIDELINES_FILENAMES_FILE using gitingest.

        Reads the GUIDELINES_FILENAMES_FILE file and processes each URL listed
        under 'github_urls' using gitingest to extract repository summaries, file trees,
        and content. The results are saved as markdown files in the
        URLS_FROM_GUIDELINES_CODE_FOLDER subfolder.

        Args:
            research_directory: Path to the research directory containing GUIDELINES_FILENAMES_FILE

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - urls_processed: List of successfully processed GitHub URLs
                - urls_failed: List of GitHub URLs that failed to process
                - total_urls: Total number of GitHub URLs processed
                - successful_urls_count: Number of GitHub URLs successfully processed
                - failed_urls_count: Number of GitHub URLs that failed to process
                - output_directory: Path to the output directory
                - message: Human-readable success message with processing results
        """
        result = await process_github_urls_tool(research_directory)
        return result

    @mcp.tool()
    async def transcribe_youtube_urls(research_directory: str) -> Dict[str, Any]:
        """
        Transcribe YouTube video URLs from GUIDELINES_FILENAMES_FILE using an LLM.

        Reads the GUIDELINES_FILENAMES_FILE file and processes each URL listed
        under 'youtube_videos_urls'. Each video is transcribed, and the results are
        saved as markdown files in the URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder.

        Args:
            research_directory: Path to the research directory containing GUIDELINES_FILENAMES_FILE

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - urls_processed: List of successfully transcribed YouTube URLs
                - urls_failed: List of YouTube URLs that failed to transcribe
                - total_urls: Total number of YouTube URLs processed
                - successful_urls_count: Number of YouTube URLs successfully transcribed
                - failed_urls_count: Number of YouTube URLs that failed to transcribe
                - output_directory: Path to the output directory
                - message: Human-readable success message with processing results
        """
        result = await transcribe_youtube_videos_tool(research_directory)
        return result

    # ============================================================================
    # RESEARCH QUERY AND ANALYSIS TOOLS
    # ============================================================================

    @mcp.tool()
    async def generate_next_queries(research_directory: str, n_queries: int = 5) -> Dict[str, Any]:
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
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - queries_generated: List of generated query dictionaries with 'query' and 'rationale' keys
                - queries_count: Number of queries generated
                - output_path: Path to the generated next_queries.md file
                - message: Human-readable success message with generation results
        """
        result = await generate_next_queries_tool(research_directory, n_queries)
        return result

    @mcp.tool()
    async def run_perplexity_research(research_directory: str, queries: list[str]) -> Dict[str, Any]:
        """
        Run selected web-search queries with Perplexity and store the results.

        Executes the provided queries using Perplexity's Sonar-Pro model and appends
        the results to perplexity_results.md in the research directory. Each query
        result includes the answer and source citations.

        Args:
            research_directory: Path to the research directory where results will be saved
            queries: List of web-search queries to execute

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - queries_executed: List of queries that were successfully executed
                - queries_failed: List of queries that failed to execute
                - total_queries: Total number of queries processed
                - successful_queries_count: Number of queries successfully executed
                - failed_queries_count: Number of queries that failed to execute
                - total_sources: Total number of sources collected across all queries
                - output_path: Path to the updated perplexity_results.md file
                - message: Human-readable success message with processing results
        """
        result = await run_perplexity_research_tool(research_directory, queries)
        return result

    # ============================================================================
    # SOURCE SELECTION AND CURATION TOOLS
    # ============================================================================

    @mcp.tool()
    async def select_research_sources_to_keep(research_directory: str) -> Dict[str, Any]:
        """
        Automatically select high-quality sources from Perplexity results.

        Uses an LLM to evaluate each source in perplexity_results.md for trustworthiness,
        authority, and relevance based on the article guidelines. Writes the comma-separated
        IDs of accepted sources to perplexity_sources_selected.md and saves a filtered
        markdown file perplexity_results_selected.md containing only the accepted sources.

        Args:
            research_directory: Path to the research directory (e.g., "articles/1")

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - sources_selected_count: Number of sources selected
                - selected_source_ids: List of IDs of selected sources
                - sources_selected_path: Path to the perplexity_sources_selected.md file
                - results_selected_path: Path to the perplexity_results_selected.md file
                - message: Human-readable success message with selection results
        """
        result = await select_research_sources_to_keep_tool(research_directory)
        return result

    @mcp.tool()
    async def select_research_sources_to_scrape(research_directory: str, max_sources: int = 5) -> Dict[str, Any]:
        """
        Select up to max_sources priority research sources to scrape in full.

        Analyzes the filtered Perplexity results together with the article guidelines and
        the material already scraped from guideline URLs, then chooses up to max_sources diverse,
        authoritative sources whose full content will add most value. The chosen URLs are
        written (one per line) to urls_to_scrape_from_research.md.

        Args:
            research_directory: Path to the research directory (e.g., "articles/1")
            max_sources: Maximum number of sources to select (default: 5)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - sources_selected: List of selected source URLs
                - sources_selected_count: Number of sources selected
                - output_path: Path to the urls_to_scrape_from_research.md file
                - reasoning: AI reasoning for why these sources were selected
                - message: Human-readable success message with selection results and reasoning
        """
        result = await select_research_sources_to_scrape_tool(research_directory, max_sources)
        return result

    @mcp.tool()
    async def scrape_research_urls(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
        """
        Scrape the selected research URLs for full content.

        Reads the URLs from urls_to_scrape_from_research.md and scrapes/cleans each URL's
        full content. The cleaned markdown files are saved to the urls_from_research
        subfolder with appropriate filenames.

        Args:
            research_directory: Path to the research directory containing urls_to_scrape_from_research.md
            concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - urls_processed: List of successfully processed URLs
                - urls_failed: List of URLs that failed to process
                - total_urls: Total number of URLs processed
                - successful_urls_count: Number of URLs successfully scraped
                - failed_urls_count: Number of URLs that failed to scrape
                - output_directory: Path to the output directory
                - message: Human-readable success message with processing results
        """
        result = await scrape_research_urls_tool(research_directory, concurrency_limit)
        return result

    # ============================================================================
    # FINAL RESEARCH COMPILATION TOOLS
    # ============================================================================

    @mcp.tool()
    async def create_research_file(research_directory: str) -> Dict[str, Any]:
        """
        Generate the final comprehensive research.md file.

        Combines all research data including filtered Perplexity results, scraped guideline
        sources, and full research sources into a comprehensive research.md file. The file
        is organized into sections with collapsible blocks for easy navigation.

        Args:
            research_directory: Path to the research directory containing all research data

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - markdown_file: Path to the generated research.md file
                - research_results_count: Number of research result sections
                - scraped_sources_count: Number of scraped sources sections
                - code_sources_count: Number of code source sections
                - youtube_transcripts_count: Number of YouTube transcript sections
                - additional_sources_count: Number of additional source sections
                - message: Human-readable success message with file generation results
        """
        result = create_research_file_tool(research_directory)
        return result
