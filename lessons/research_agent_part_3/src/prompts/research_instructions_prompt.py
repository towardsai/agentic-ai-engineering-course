"""Full research instructions prompt implementation."""

import logging

from fastmcp.server.dependencies import get_access_token

from ..config.settings import settings

logger = logging.getLogger(__name__)


def get_upload_url(user_id: str) -> str:
    """
    Get the upload article guideline URL based on server configuration.

    Args:
        user_id: The user ID to include as a query parameter

    Returns:
        The full URL to the upload article guideline UI with user_id
    """
    if settings.server_base_url:
        base_url = settings.server_base_url.rstrip("/")
    else:
        base_url = f"http://{settings.server_host}:{settings.server_port}"
    return f"{base_url}/upload_article_guideline?user_id={user_id}"


def get_base_url() -> str:
    """
    Get the base URL for the server.

    Returns:
        The base URL without trailing slash
    """
    if settings.server_base_url:
        return settings.server_base_url.rstrip("/")
    else:
        return f"http://{settings.server_host}:{settings.server_port}"


async def full_research_instructions_prompt() -> str:
    """
    Return the complete Nova research agent instructions as a string.

    Returns:
        The complete research instructions as a string
    """
    # Get user ID from access token (if available)
    user_id = None
    try:
        token = get_access_token()
        user_id = token.claims.get("sub")
    except Exception as e:
        logger.warning(f"Could not get access token for prompt rendering: {e}")

    # Build URLs with user_id if available, otherwise use placeholder
    if user_id:
        upload_url = get_upload_url(user_id)
    else:
        upload_url = f"{get_base_url()}/upload_article_guideline?user_id=<USER_ID>"

    base_url = get_base_url()

    instructions_content = f"""
Your job is to execute the workflow below.

**Workflow:**

0. Get Article Guideline ID:

    Before starting, you need an article guideline ID to proceed with the research workflow.

    Please provide your article guideline ID, OR if you don't have one yet:
    - Visit {upload_url} to upload a new article guideline file
    - Select your article guideline markdown file (.md) and upload it
    - After uploading, you will receive an article guideline ID
    - Copy that ID and provide it here to continue

    Once you have the article guideline ID, provide it and we can proceed with the workflow.

1. Setup:

    1.1. Explain to the user the numbered steps of the workflow. Be concise. Keep them numbered so that the user
    can easily refer to them later.
    
    1.2. Ask the user if any modification is needed for the workflow (e.g. running from a specific step,
    or adding user feedback to specific steps).

    1.3 Extract the URLs from the article guideline (using the ID from step 0) with the "extract_guidelines_urls" tool.
    This tool reads the article guideline from the database and extracts four groups of references:
    • "github_urls" - all GitHub links;
    • "youtube_videos_urls" - all YouTube video links;
    • "other_urls" - all remaining HTTP/HTTPS links;
    • "local_files" - relative paths to local files mentioned in the guidelines (e.g. "code.py", "src/main.py").
    Only extensions allowed are: ".py", ".ipynb", and ".md".
    The extracted data is saved to the database.

2. Process the extracted resources:

    2.1 Local files upload:
    
    Provide the user with this upload URL to upload any local files they need for the research:
    
    {base_url}/upload_local_files?user_id={user_id}&article_guideline_id=<ARTICLE_GUIDELINE_ID>
    
    (Replace <ARTICLE_GUIDELINE_ID> with the actual article guideline ID from step 0)
    
    - If step 1.3 found local files ("local_files_count" > 0), those files will be shown as suggestions on the upload page
    - The user can upload any files they need, not just the suggested ones
    - Wait for the user to confirm they have completed their uploads (or confirm they have no files to upload)
    - Then proceed to the next steps
    
    After user confirmation, you can proceed to steps 2.2-2.4 in parallel.

    2.2 Other URL links - run the "scrape_and_clean_other_urls" tool with the article guideline ID to scrape and
    clean the URLs from the `other_urls` list (extracted in step 1.3). The cleaned content is saved to the database.

    2.3 GitHub URLs - run the "process_github_urls" tool with the article guideline ID to process the
    `github_urls` list from the database (extracted in step 1.3) with gitingest. The GitIngest results
    are saved to the database.

    2.4 YouTube URLs - run the "transcribe_youtube_urls" tool with the article guideline ID to process the
    `youtube_videos_urls` list from the database (extracted in step 1.3) and transcribe each video. The
    transcriptions are saved to the database.
        Note: Please be aware that video transcription can be a time-consuming process. For reference,
        transcribing a 39-minute video can take approximately 4.5 minutes.

3. Repeat the following research loop for 3 rounds:

    3.1. Run the "generate_next_queries" tool with the article guideline ID to analyze the article guidelines,
    the already-scraped content from the database, and existing Perplexity results. The tool identifies knowledge
    gaps and proposes new web-search questions with justifications. The generated queries are returned directly.

    3.2. Run the "run_perplexity_research" tool with the article guideline ID and the new queries. This tool
    executes the queries with Perplexity and appends the results to the perplexity_results field in the database.

4. Filter Perplexity results by quality:

    4.1 Run the "select_research_sources_to_keep" tool with the article guideline ID. The tool reads the article
    guidelines and Perplexity results from the database, automatically evaluates each source for trustworthiness,
    authority and relevance, then saves the comma-separated IDs of accepted sources and a filtered markdown file
    containing only the selected sources to the database.

5. Identify which of the accepted sources deserve a *full* scrape:

    5.1 Run the "select_research_sources_to_scrape" tool with the article guideline ID. It analyzes the filtered
    Perplexity results from the database together with the article guidelines and material already scraped from
    guideline URLs, then chooses up to 5 diverse, authoritative sources whose full content will add most value.
    The chosen URLs are saved to the database.

    5.2 Run the "scrape_research_urls" tool with the article guideline ID. The tool reads the URLs from the
    database (saved in step 5.1), scrapes and cleans each URL's full content (ignoring YouTube URLs), and saves
    the results to the database.

6. Write final research file:

    6.1 Run the "create_research_file" tool with the article guideline ID. The tool combines all research data from
    the database including filtered Perplexity results, scraped guideline sources, GitHub ingests, YouTube transcripts,
    and scraped research sources into a comprehensive research markdown organized into sections with collapsible blocks
    for easy navigation. The final research is saved to the database, and the tool returns a download URL.
    After the tool completes, inform the user to open the download URL in their browser to download the research.md file.

Depending on the results of previous steps, you may want to skip running a tool if not necessary.

**Critical Failure Policy:**

If a tool reports a complete failure, you are required to halt the entire workflow immediately. A complete failure
is defined as processing zero items successfully (e.g., scraped 0/7 URLs, processed 0 files).

If this occurs, your immediate and only action is to:
    1. State the exact tool that failed and quote the output message.
    2. Announce that you are stopping the workflow as per your instructions.
    3. Ask the user for guidance on how to proceed.

**File and Folder Structure:**

After running the complete workflow, all research data is stored in the database:

- Extracted URLs, local files, scraped web content from guidelines
- GitHub ingests, YouTube transcripts  
- Perplexity research results, selected source IDs, filtered results
- URLs to scrape and scraped research content
- Final research markdown

The final research file can be downloaded via the URL provided by the "create_research_file" tool.

This database-based approach ensures all research artifacts are systematically collected, processed, and made easily
accessible for article writing and future reference.
    """.strip()

    return instructions_content
