"""
Composed MCP Server - Combines Nova and Brown servers into a single endpoint.

This server uses FastMCP's composition features to mount both the Nova research agent
and Brown writing workflow servers, exposing all their capabilities through a single
MCP server without prefixes.

Usage:
    python -m src.main
"""

import json
import logging
from pathlib import Path

from fastmcp import Client, FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_server_config(config_path: Path | None = None) -> dict:
    """Load the MCP servers configuration from JSON file.
    
    Args:
        config_path: Optional path to config file. If None, uses default mcp_servers_to_compose.json
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "mcp_servers_to_compose.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)


def create_composed_server(config_path: Path | None = None) -> FastMCP:
    """
    Create a composed MCP server by mounting Nova and Brown servers.
    
    Args:
        config_path: Optional path to config file. If None, uses default mcp_servers_to_compose.json
    
    Returns:
        FastMCP: The composed server instance with both servers mounted
    """
    # Create the main composed server
    mcp = FastMCP(
        name="Nova+Brown Composed Server",
        version="0.1.0",
    )
    
    logger.info("Loading server configuration...")
    config = load_server_config(config_path)
    
    servers_config = config.get("mcpServers", {})
    
    if not servers_config:
        raise ValueError("No servers found in configuration")
    
    logger.info(f"Found {len(servers_config)} servers to compose: {list(servers_config.keys())}")
    
    # Create proxies and mount each server
    for server_name, server_config in servers_config.items():
        logger.info(f"Creating proxy for {server_name}...")
        
        # Wrap the server config in the mcpServers structure expected by Client
        client_config = {"mcpServers": {server_name: server_config}}
        
        # Create a client for this server
        client = Client(client_config)
        
        # Create a proxy from the client
        proxy = FastMCP.as_proxy(client)
        
        logger.info(f"Mounting {server_name} without prefix...")
        mcp.mount(proxy)
    
    # Register the combined workflow prompt
    register_combined_prompt(mcp)
    
    logger.info("Composed server created successfully!")
    return mcp


def register_combined_prompt(mcp: FastMCP) -> None:
    """Register the combined research and writing workflow prompt."""
    
    @mcp.prompt()
    def full_research_and_writing_workflow(dir_path: Path) -> str:
        """Complete workflow for research and article generation.
        
        This prompt combines the Nova research agent workflow with the Brown
        article generation workflow, providing end-to-end instructions for
        conducting comprehensive research and generating an article from that research.
        
        Args:
            dir_path: Path to the directory that will contain research resources
                     and the final article.
        
        Returns:
            A formatted prompt string with complete workflow instructions.
        """
        return f"""
# Complete Research and Article Generation Workflow

This workflow combines two phases: research (Nova) and article generation (Brown).

---

## PHASE 1: Research (Nova)

Your job is to execute the workflow below.

All the tools require a research directory as input.
If the user doesn't provide a research directory, you should ask for it before executing any tool.

**Workflow:**

1. Setup:

    1.1. Explain to the user the numbered steps of the workflow. Be concise. Keep them numbered so that the user
    can easily refer to them later.
    
    1.2. Ask the user for the research directory, if not provided. Ask the user if any modification is needed for the
    workflow (e.g. running from a specific step, or adding user feedback to specific steps).

    1.3 Extract the URLs from the ARTICLE_GUIDELINE_FILE with the "extract_guidelines_urls" tool. This tool reads the
    ARTICLE_GUIDELINE_FILE and extracts three groups of references from the guidelines:
    • "github_urls" - all GitHub links;
    • "youtube_videos_urls" - all YouTube video links;
    • "other_urls" - all remaining HTTP/HTTPS links;
    • "local_files" - relative paths to local files mentioned in the guidelines (e.g. "code.py", "src/main.py").
    Only extensions allowed are: ".py", ".ipynb", and ".md".
    The extracted data is saved to the GUIDELINES_FILENAMES_FILE within the NOVA_FOLDER directory.

2. Process the extracted resources in parallel:

    You can run the following sub-steps (2.1 to 2.4) in parallel. In a single turn, you can call all the
    necessary tools for these steps.

    2.1 Local files - run the "process_local_files" tool to read every file path listed under "local_files" in the
    GUIDELINES_FILENAMES_FILE and copy its content into the LOCAL_FILES_FROM_RESEARCH_FOLDER subfolder within
    NOVA_FOLDER, giving each copy an appropriate filename (path separators are replaced with underscores).

    2.2 Other URL links - run the "scrape_and_clean_other_urls" tool to read the `other_urls` list from the
    GUIDELINES_FILENAMES_FILE and scrape/clean them. The tool writes the cleaned markdown files inside the
    URLS_FROM_GUIDELINES_FOLDER subfolder within NOVA_FOLDER.

    2.3 GitHub URLs - run the "process_github_urls" tool to process the `github_urls` list from the
    GUIDELINES_FILENAMES_FILE with gitingest and save a Markdown summary for each URL inside the
    URLS_FROM_GUIDELINES_CODE_FOLDER subfolder within NOVA_FOLDER.

    2.4 YouTube URLs - run the "transcribe_youtube_urls" tool to process the `youtube_videos_urls` list from the
    GUIDELINES_FILENAMES_FILE, transcribe each video, and save the transcript as a Markdown file inside the
    URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder within NOVA_FOLDER.
        Note: Please be aware that video transcription can be a time-consuming process. For reference,
        transcribing a 39-minute video can take approximately 4.5 minutes.

3. Repeat the following research loop for 3 rounds:

    3.1. Run the "generate_next_queries" tool to analyze the ARTICLE_GUIDELINE_FILE, the already-scraped guideline
    URLs, and the existing PERPLEXITY_RESULTS_FILE. The tool identifies knowledge gaps, proposes new web-search
    questions, and writes them - together with a short justification for each - to the NEXT_QUERIES_FILE within
    NOVA_FOLDER.

    3.2. Run the "run_perplexity_research" tool with the new queries. This tool executes the queries with
    Perplexity and appends the results to the PERPLEXITY_RESULTS_FILE within NOVA_FOLDER.

4. Filter Perplexity results by quality:

    4.1 Run the "select_research_sources_to_keep" tool. The tool reads the ARTICLE_GUIDELINE_FILE and the
    PERPLEXITY_RESULTS_FILE, automatically evaluates each source for trustworthiness, authority and relevance,
    writes the comma-separated IDs of the accepted sources to the PERPLEXITY_SOURCES_SELECTED_FILE **and** saves a
    filtered markdown file PERPLEXITY_RESULTS_SELECTED_FILE that contains only the full content blocks of the accepted
    sources. Both files are saved within NOVA_FOLDER.

5. Identify which of the accepted sources deserve a *full* scrape:

    5.1 Run the "select_research_sources_to_scrape" tool. It analyses the PERPLEXITY_RESULTS_SELECTED_FILE together
    with the ARTICLE_GUIDELINE_FILE and the material already scraped from guideline URLs, then chooses up to 5 diverse,
    authoritative sources whose full content will add most value. The chosen URLs are written (one per line) to the
    URLS_TO_SCRAPE_FROM_RESEARCH_FILE within NOVA_FOLDER.

    5.2 Run the "scrape_research_urls" tool. The tool reads the URLs from URLS_TO_SCRAPE_FROM_RESEARCH_FILE and
    scrapes/cleans each URL's full content. The cleaned markdown files are saved to the
    URLS_FROM_RESEARCH_FOLDER subfolder within NOVA_FOLDER with appropriate filenames.

6. Write final research file:

    6.1 Run the "create_research_file" tool. The tool combines all research data including filtered Perplexity results
    from PERPLEXITY_RESULTS_SELECTED_FILE, scraped guideline sources from URLS_FROM_GUIDELINES_FOLDER,
    URLS_FROM_GUIDELINES_CODE_FOLDER, and URLS_FROM_GUIDELINES_YOUTUBE_FOLDER, and full research sources from
    URLS_FROM_RESEARCH_FOLDER into a comprehensive RESEARCH_MD_FILE organized into sections with collapsible blocks
    for easy navigation. The final RESEARCH_MD_FILE is saved in the root of the research directory.

Depending on the results of previous steps, you may want to skip running a tool if not necessary.

**Critical Failure Policy:**

If a tool reports a complete failure, you are required to halt the entire workflow immediately. A complete failure
is defined as processing zero items successfully (e.g., scraped 0/7 URLs, processed 0 files).

If this occurs, your immediate and only action is to:
    1. State the exact tool that failed and quote the output message.
    2. Announce that you are stopping the workflow as per your instructions.
    3. Ask the user for guidance on how to proceed.

**File and Folder Structure:**

After running the complete workflow, the research directory will contain the following structure:

```
research_directory/
├── ARTICLE_GUIDELINE_FILE                           # Input: Article guidelines and requirements
├── NOVA_FOLDER/                                     # Hidden directory containing all research data
│   ├── GUIDELINES_FILENAMES_FILE                    # Extracted URLs and local files from guidelines
│   ├── LOCAL_FILES_FROM_RESEARCH_FOLDER/           # Copied local files referenced in guidelines
│   │   └── [processed_local_files...]
│   ├── URLS_FROM_GUIDELINES_FOLDER/               # Scraped content from other URLs in guidelines
│   │   └── [scraped_web_pages...]
│   ├── URLS_FROM_GUIDELINES_CODE_FOLDER/          # GitHub repository summaries and code analysis
│   │   └── [github_repo_summaries...]
│   ├── URLS_FROM_GUIDELINES_YOUTUBE_FOLDER/       # YouTube video transcripts
│   │   └── [youtube_transcripts...]
│   ├── NEXT_QUERIES_FILE                           # Generated web-search queries with justifications
│   ├── PERPLEXITY_RESULTS_FILE                     # Complete results from all Perplexity research rounds
│   ├── PERPLEXITY_SOURCES_SELECTED_FILE            # Comma-separated IDs of quality sources selected
│   ├── PERPLEXITY_RESULTS_SELECTED_FILE            # Filtered Perplexity results (only selected sources)
│   ├── URLS_TO_SCRAPE_FROM_RESEARCH_FILE          # URLs selected for full content scraping
│   └── URLS_FROM_RESEARCH_FOLDER/                 # Fully scraped content from selected research URLs
│       └── [full_research_sources...]
└── RESEARCH_MD_FILE                                 # Final comprehensive research compilation
```

This organized structure ensures all research artifacts are systematically collected, processed, and made easily
accessible for article writing and future reference.

---

## PHASE 2: Article Generation (Brown)

Once the research phase is complete, the research directory (specified as `{dir_path}`) will contain:
- `ARTICLE_GUIDELINE_FILE` - The original article guidelines and requirements
- `RESEARCH_MD_FILE` - The comprehensive research compilation from Phase 1

These files provide all the necessary context and information for article generation.

**Next Step:**

Using Brown hosted as an MCP server, generate an article using all the necessary resources from 
the following directory: `{dir_path}`. Don't check if any expected files are missing, just trigger 
the "generate_article" tool of the Brown MCP Server, which will take care of everything.
""".strip()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Composed MCP Server (Nova + Brown)")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (stdio or streamable-http)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8003,
        help="Port number for HTTP transport (default: 8003)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to MCP servers configuration file (overrides default mcp_servers_to_compose.json)",
    )
    args = parser.parse_args()
    
    logger.info("Starting composed MCP server...")
    
    try:
        composed_server = create_composed_server(config_path=args.config)
        logger.info("Running composed server...")
        
        # Run the server with the specified transport
        if args.transport == "streamable-http":
            composed_server.run(transport=args.transport, port=args.port)
        else:
            composed_server.run(transport=args.transport)
    except Exception as e:
        logger.error(f"Failed to start composed server: {e}")
        raise

