"""MCP Tools registration for research operations."""

from typing import Any, Dict

import opik
from fastmcp import FastMCP

from ..tools import (
    create_research_file_tool,
    extract_guidelines_urls_tool,
    generate_next_queries_tool,
    process_github_urls_tool,
    run_perplexity_research_tool,
    scrape_and_clean_other_urls_tool,
    scrape_research_urls_tool,
    select_research_sources_to_keep_tool,
    select_research_sources_to_scrape_tool,
    transcribe_youtube_videos_tool,
)
from ..utils.opik_utils import update_opik_thread_id
from ..utils.rate_limit_utils import rate_limited


def register_mcp_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with the server instance."""

    # ============================================================================
    # URL AND FILE EXTRACTION TOOLS
    # ============================================================================

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def extract_guidelines_urls(article_guideline_id: str) -> Dict[str, Any]:
        """
        Extract URLs and local file references from article guidelines.

        Reads the article guideline from the database and extracts:
        - GitHub URLs
        - YouTube video URLs
        - Other HTTP/HTTPS URLs
        - Local file references (files mentioned in quotes with extensions)

        Results are saved to the database in the extracted_urls column.

        Args:
            article_guideline_id: UUID of the article guideline in the database

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - github_sources_count: Number of GitHub URLs found
                - youtube_sources_count: Number of YouTube URLs found
                - web_sources_count: Number of other web URLs found
                - local_files_count: Number of local file references found
                - message: Human-readable success message
        """

        update_opik_thread_id(article_guideline_id)

        result = await extract_guidelines_urls_tool(article_guideline_id)
        return result

    # ============================================================================
    # WEB SCRAPING AND CONTENT PROCESSING TOOLS
    # ============================================================================

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def scrape_and_clean_other_urls(article_guideline_id: str, concurrency_limit: int = 4) -> Dict[str, Any]:
        """
        Scrape and clean other URLs from article guidelines.

        Reads the list of other URLs from the database (extracted in step 1.3)
        and scrapes/cleans each URL. The cleaned markdown content is saved to
        the database.

        Args:
            article_guideline_id: UUID of the article guideline in the database
            concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success" or "warning")
                - urls_processed: Number of URLs successfully scraped
                - urls_failed: Number of URLs that failed to scrape
                - urls_total: Total number of URLs attempted
                - message: Human-readable success message with processing results
        """

        update_opik_thread_id(article_guideline_id)

        result = await scrape_and_clean_other_urls_tool(article_guideline_id, concurrency_limit)
        return result

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def process_github_urls(article_guideline_id: str) -> Dict[str, Any]:
        """
        Process GitHub URLs from article guidelines using gitingest.

        Reads the list of GitHub URLs from the database (extracted in step 1.3)
        and processes each URL with GitIngest. The processed markdown content is
        saved to the database.

        Args:
            article_guideline_id: UUID of the article guideline in the database

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success" or "warning")
                - urls_processed: Number of GitHub URLs successfully processed
                - urls_failed: Number of GitHub URLs that failed to process
                - urls_total: Total number of GitHub URLs attempted
                - message: Human-readable success message with processing results
        """

        update_opik_thread_id(article_guideline_id)

        result = await process_github_urls_tool(article_guideline_id)
        return result

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def transcribe_youtube_urls(article_guideline_id: str) -> Dict[str, Any]:
        """
        Transcribe YouTube video URLs from article guidelines.

        Reads the list of YouTube URLs from the database (extracted in step 1.3)
        and transcribes each video. The transcription results are saved to the
        database.

        Args:
            article_guideline_id: UUID of the article guideline in the database

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success" or "warning")
                - videos_processed: Number of videos successfully transcribed
                - videos_failed: Number of videos that failed to transcribe
                - videos_total: Total number of videos attempted
                - message: Human-readable success message with processing results
        """

        update_opik_thread_id(article_guideline_id)

        result = await transcribe_youtube_videos_tool(article_guideline_id)
        return result

    # ============================================================================
    # RESEARCH QUERY AND ANALYSIS TOOLS
    # ============================================================================

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def generate_next_queries(article_guideline_id: str, n_queries: int = 5) -> Dict[str, Any]:
        """
        Generate candidate web-search queries for the next research round.

        Analyzes the article guidelines, already-scraped content from the database,
        and existing Perplexity results to identify knowledge gaps and propose new
        web-search questions. Each query includes a rationale.

        Args:
            article_guideline_id: UUID of the article guideline in the database
            n_queries: Number of queries to generate (default: 5)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - queries_generated: List of generated query tuples (query, rationale)
                - queries_count: Number of queries generated
                - message: Human-readable success message with queries
        """

        update_opik_thread_id(article_guideline_id)

        result = await generate_next_queries_tool(article_guideline_id, n_queries)
        return result

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def run_perplexity_research(article_guideline_id: str, queries: list[str]) -> Dict[str, Any]:
        """
        Run selected web-search queries with Perplexity and store the results.

        Executes the provided queries using Perplexity's Sonar-Pro model and appends
        the results to the perplexity_results field in the articles table.

        Args:
            article_guideline_id: UUID of the article guideline in the database
            queries: List of web-search queries to execute

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - queries_processed: Number of queries processed
                - sources_added: Number of sources added
                - message: Human-readable success message with processing results
        """

        update_opik_thread_id(article_guideline_id)

        result = await run_perplexity_research_tool(article_guideline_id, queries)
        return result

    # ============================================================================
    # SOURCE SELECTION AND CURATION TOOLS
    # ============================================================================

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def select_research_sources_to_keep(article_guideline_id: str) -> Dict[str, Any]:
        """
        Automatically select high-quality sources from Perplexity results.

        Uses an LLM to evaluate each source in the database for trustworthiness,
        authority, and relevance based on the article guidelines. Saves the
        comma-separated IDs and filtered results to the database.

        Args:
            article_guideline_id: UUID of the article guideline in the database

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - sources_selected_count: Number of sources selected
                - selected_source_ids: List of IDs of selected sources
                - message: Human-readable success message with selection results
        """

        update_opik_thread_id(article_guideline_id)

        result = await select_research_sources_to_keep_tool(article_guideline_id)
        return result

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def select_research_sources_to_scrape(article_guideline_id: str, max_sources: int = 5) -> Dict[str, Any]:
        """
        Select up to max_sources priority research sources to scrape in full.

        Analyzes the filtered Perplexity results from the database together with
        the article guidelines and material already scraped, then chooses diverse,
        authoritative sources. The chosen URLs are saved to the database.

        Args:
            article_guideline_id: UUID of the article guideline in the database
            max_sources: Maximum number of sources to select (default: 5)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - sources_selected: List of selected source URLs
                - sources_selected_count: Number of sources selected
                - reasoning: AI reasoning for why these sources were selected
                - message: Human-readable success message with selection results and reasoning
        """

        update_opik_thread_id(article_guideline_id)

        result = await select_research_sources_to_scrape_tool(article_guideline_id, max_sources)
        return result

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def scrape_research_urls(article_guideline_id: str, concurrency_limit: int = 4) -> Dict[str, Any]:
        """
        Scrape the selected research URLs for full content.

        Reads the URLs from the database (saved in step 5.1) and scrapes/cleans each URL's
        full content. YouTube URLs are ignored. The cleaned markdown content is saved to
        the database.

        Args:
            article_guideline_id: UUID of the article guideline in the database
            concurrency_limit: Maximum number of concurrent tasks for scraping (default: 4)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success" or "warning")
                - urls_processed: Number of URLs successfully scraped
                - urls_failed: Number of URLs that failed to scrape
                - urls_total: Total number of URLs attempted
                - youtube_urls_ignored: Number of YouTube URLs that were ignored
                - message: Human-readable success message with processing results
        """

        update_opik_thread_id(article_guideline_id)

        result = await scrape_research_urls_tool(article_guideline_id, concurrency_limit)
        return result

    # ============================================================================
    # FINAL RESEARCH COMPILATION TOOLS
    # ============================================================================

    @mcp.tool()
    @opik.track(type="tool")
    @rate_limited
    async def create_research_file(article_guideline_id: str) -> Dict[str, Any]:
        """
        Generate the final comprehensive research markdown.

        Combines all research data from the database including filtered Perplexity results,
        scraped guideline sources, GitHub ingests, YouTube transcripts, and scraped research
        sources. The final markdown is saved to the database and a download link is provided.

        Args:
            article_guideline_id: UUID of the article guideline in the database

        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Operation status ("success")
                - download_url: URL to download the research.md file
                - research_results_count: Number of research result sections
                - scraped_sources_count: Number of scraped sources sections
                - code_sources_count: Number of code source sections
                - youtube_transcripts_count: Number of YouTube transcript sections
                - additional_sources_count: Number of additional source sections
                - message: Human-readable success message with download instructions
        """

        update_opik_thread_id(article_guideline_id)

        result = await create_research_file_tool(article_guideline_id)
        return result
