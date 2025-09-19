# Lesson 16: Data Ingestion — Building Essential Research Tools

In this lesson, we focus on building the first set of essential tools for data gathering in our research agent. We'll implement tools that read article guideline files, extract web URLs programmatically, and scrape their content in parallel. This lesson demonstrates how file-based approaches can save tokens for the orchestrating agent, which only needs to process simple success or failure messages rather than large content blocks.

Learning Objectives:
- Learn how to build tools that extract URLs and references from text files
- Understand the benefits of file-based tool outputs for token efficiency
- Implement robust web scraping tools using external services
- Handle error cases gracefully in tool implementations
- Use parallel processing for efficient data collection

## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:

```python
%load_ext autoreload
%autoreload 2
```

### Set Up Python Environment

To set up your Python virtual environment using `uv` and load it into the Notebook, follow the step-by-step instructions from the `Course Admin` lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.

### Configure API Keys

To run this lesson, you'll need several API keys configured in your `.env` file:

1. **Gemini API Key**: Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. **Firecrawl API Key**: Get your key from [Firecrawl](https://firecrawl.dev/).
3. **OpenAI API Key**: For content cleaning (optional, can use Gemini instead).

From the root of your project, run: `cp .env.example .env` and fill in the required variables.

Now, the code below will load the keys from the `.env` file:

```python
from utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
```

### Import Key Packages

```python
import nest_asyncio
nest_asyncio.apply()  # Allow nested async usage in notebooks
```

## 2. Understanding the Research Agent Workflow

The research agent follows a systematic workflow for data ingestion. The MCP prompt defines a clear two-phase approach:

- **Step 1**: Extract URLs and file references from the article guidelines
- **Step 2**: Process all the extracted resources in parallel (local files, web URLs, GitHub repos, YouTube videos)

Let's examine the MCP tools involved in these first two steps of the workflow.

Source: _research_agent_part_2/mcp_server/src/routers/tools.py_

```python
def register_mcp_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with the server instance."""
    
    # Step 1: Extract URLs and file references from guidelines
    @mcp.tool()
    async def extract_guidelines_urls(research_directory: str) -> Dict[str, Any]:
        """
        Extract URLs and local file references from article guidelines.
        
        Reads the ARTICLE_GUIDELINE_FILE file in the research directory and extracts:
        - GitHub URLs
        - Other HTTP/HTTPS URLs  
        - Local file references (files mentioned in quotes with extensions)
        
        Results are saved to GUIDELINES_FILENAMES_FILE in the research directory.
        """
        result = extract_guidelines_urls_tool(research_directory)
        return result

    # Step 2: Process local files
    @mcp.tool()
    async def process_local_files(research_directory: str) -> Dict[str, Any]:
        """Process local files referenced in the article guidelines."""
        result = process_local_files_tool(research_directory)
        return result
        
    # Step 2: Scrape web URLs
    @mcp.tool() 
    async def scrape_and_clean_other_urls(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
        """Scrape and clean other URLs from GUIDELINES_FILENAMES_FILE."""
        result = await scrape_and_clean_other_urls_tool(research_directory, concurrency_limit)
        return result
        
    # Step 2: Process GitHub repositories
    @mcp.tool()
    async def process_github_urls(research_directory: str) -> Dict[str, Any]:
        """
        Process GitHub URLs from GUIDELINES_FILENAMES_FILE using gitingest.
        
        Reads the GUIDELINES_FILENAMES_FILE file and processes each URL listed
        under 'github_urls' using gitingest to extract repository summaries, file trees,
        and content. The results are saved as markdown files in the
        URLS_FROM_GUIDELINES_CODE_FOLDER subfolder.
        """
        result = await process_github_urls_tool(research_directory)
        return result
        
    # Step 2: Transcribe YouTube videos
    @mcp.tool()
    async def transcribe_youtube_urls(research_directory: str) -> Dict[str, Any]:
        """
        Transcribe YouTube video URLs from GUIDELINES_FILENAMES_FILE using Gemini 2.5 Pro.
        
        Reads the GUIDELINES_FILENAMES_FILE file and processes each URL listed
        under 'youtube_videos_urls'. Each video is transcribed, and the results are
        saved as markdown files in the URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder.
        """
        result = await transcribe_youtube_videos_tool(research_directory)
        return result
```

These tools correspond to steps 1 and 2 of the research workflow. Step 1 focuses on extraction and categorization, while Step 2 processes all the different types of content sources in parallel for maximum efficiency.

## 3. Step 1: Extracting URLs from Guidelines

The first tool in our data ingestion pipeline reads an article guideline file and programmatically extracts all URLs and file references it contains.

### 3.1 Understanding the Tool Implementation

Source: _research_agent_part_2/mcp_server/src/tools/extract_guidelines_urls_tool.py_

```python
def extract_guidelines_urls_tool(research_folder: str) -> Dict[str, Any]:
    """
    Extract URLs and local file references from the article guidelines in the research folder.
    
    Reads the ARTICLE_GUIDELINE_FILE file and extracts:
    - GitHub URLs
    - YouTube video URLs  
    - Other HTTP/HTTPS URLs
    - Local file references
    
    Results are saved to GUIDELINES_FILENAMES_FILE in the research folder.
    """
    logger.debug(f"Extracting URLs from article guidelines in: {research_folder}")
    
    # Convert to Path object
    research_path = Path(research_folder)
    nova_path = research_path / NOVA_FOLDER
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    
    # Validate folders and files
    validate_research_folder(research_path)
    validate_guidelines_file(guidelines_path)
    
    # Create NOVA_FOLDER directory if it doesn't exist
    nova_path.mkdir(parents=True, exist_ok=True)
    
    # Read guidelines content
    guidelines_content = read_file_safe(guidelines_path)
    
    # Extract URLs and local file paths
    urls = extract_urls(guidelines_content)
    local_paths = extract_local_paths(guidelines_content)
    
    # Prepare the extracted data structure
    extracted_data = {
        "github_urls": urls["github_urls"],
        "youtube_videos_urls": urls["youtube_videos_urls"], 
        "other_urls": urls["other_urls"],
        "local_file_paths": local_paths,
    }
    
    # Save to JSON file
    output_path = nova_path / GUIDELINES_FILENAMES_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    return {
        "status": "success",
        "github_sources_count": len(urls["github_urls"]),
        "youtube_sources_count": len(urls["youtube_videos_urls"]),
        "web_sources_count": len(urls["other_urls"]),
        "local_files_count": len(local_paths),
        "output_path": str(output_path),
        "message": f"Successfully extracted URLs from article guidelines in '{research_folder}'. "
                  f"Found {len(urls['github_urls'])} GitHub URLs, {len(urls['youtube_videos_urls'])} YouTube videos URLs, "
                  f"{len(urls['other_urls'])} other URLs, and {len(local_paths)} local file references. "
                  f"Results saved to: {output_path}"
    }
```

### 3.2 URL Extraction and Categorization

The core URL extraction logic uses regular expressions to find and categorize different types of URLs. Let's examine how the tool identifies and separates GitHub URLs, YouTube URLs, and other web URLs.

Source: _research_agent_part_2/mcp_server/src/tools/extract_guidelines_urls_tool.py_

```python
# Extract URLs and categorize them
all_urls = extract_urls(text)
github_source_urls = [u for u in all_urls if "github.com" in u]
youtube_source_urls = [u for u in all_urls if "youtube.com" in u]
web_source_urls = [u for u in all_urls if "github.com" not in u and "youtube.com" not in u]
```

The `extract_urls` function from the guideline extractions handler finds all HTTP/HTTPS URLs:

Source: _research_agent_part_2/mcp_server/src/app/guideline_extractions_handler.py_

```python
def extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs from the given text."""
    url_pattern = re.compile(r"https?://[^\s)>\"',]+")
    return url_pattern.findall(text)
```

This regular expression pattern:
- `https?://` - Matches both HTTP and HTTPS protocols
- `[^\s)>\"',]+` - Matches any characters except whitespace, closing parentheses, greater-than signs, quotes, or commas
- This ensures URLs are extracted cleanly from markdown links, plain text, and various formatting contexts

After extraction, the URLs are categorized by domain to enable specialized processing for each type of content source.

### 3.3 Local File Path Extraction

Local file references are extracted using a more sophisticated approach that handles both quoted and unquoted file references.

Source: _research_agent_part_2/mcp_server/src/app/guideline_extractions_handler.py_

```python
def extract_local_paths(text: str) -> list[str]:
    """Extract local file paths that are referenced inside double quotes or as standalone filenames.

    We treat a reference as a local file path if it:
      • is wrapped in double quotes e.g. "code.py" or "src/main.py", OR
      • appears as a standalone filename with valid extension (e.g., at the start of a line or after whitespace)
      • does NOT start with an URL scheme such as http:// or https://
      • has a file extension that is one of: .py, .ipynb, .md
    """
    local_files = []

    # First, find anything inside double quotes
    candidate_pattern = re.compile(r'"([^"]+)"')
    quoted_candidates = candidate_pattern.findall(text)

    for c in quoted_candidates:
        c = c.strip()
        # Skip if it looks like a URL
        if re.match(r"https?://", c, re.IGNORECASE):
            continue
        # Must have one of the allowed file extensions
        if re.search(r"\.(py|ipynb|md)$", c, re.IGNORECASE):
            if c not in local_files:
                local_files.append(c)

    # Second, find standalone filenames (not wrapped in quotes)
    # Look for files that appear after whitespace or at start of line and have valid extensions
    standalone_pattern = re.compile(r'(?:^|\s)([^\s"]+\.(py|ipynb|md))(?:\s|$)', re.MULTILINE | re.IGNORECASE)
    standalone_matches = standalone_pattern.findall(text)

    for match in standalone_matches:
        filename = match[0].strip()

        # Skip if it looks like a URL
        if re.match(r"https?://", filename, re.IGNORECASE):
            continue

        # Skip if it contains URL-like patterns (has protocol or domain-like structure)
        if "://" in filename or filename.count(".") > 2:
            continue

        if filename not in local_files:
            local_files.append(filename)

    return local_files
```

This function uses a two-stage approach:

1. **Quoted References**: Finds filenames wrapped in double quotes like `"src/main.py"` or `"notebook.ipynb"`
2. **Standalone References**: Identifies standalone filenames that appear after whitespace or at line beginnings

The function includes several safety checks:
- Only allows specific file extensions (`.py`, `.ipynb`, `.md`)
- Excludes anything that looks like a URL
- Prevents duplicate entries
- Handles both relative paths (`src/file.py`) and simple filenames (`file.py`)

This robust extraction ensures that the agent can reliably identify local files referenced in article guidelines, regardless of how they're formatted in the text.

### 3.4 Benefits of Short Tool Outputs

Notice how this tool returns a concise summary rather than the full extracted content. This design choice has several advantages:

1. **Token Efficiency**: The agent receives only essential information (counts, status, file path) rather than large content blocks
2. **Context Management**: Keeps the agent's context window manageable for complex workflows  
3. **Selective Reading**: The agent can choose to read the output file only if needed for decision-making
4. **Error Handling**: Clear status messages help the agent understand what succeeded or failed

Let's test this tool programmatically:

```python
from research_agent_part_2.mcp_server.src.tools import extract_guidelines_urls_tool

# Update this path to your actual sample research folder
research_folder = "/Users/fabio/Desktop/course-ai-agents/lessons/research_agent_part_2/data/sample_research_folder"
result = extract_guidelines_urls_tool(research_folder=research_folder)
print(result)
```

The output will show a structured summary like:

```json
{
  "status": "success",
  "github_sources_count": 1,
  "youtube_sources_count": 2, 
  "web_sources_count": 6,
  "local_files_count": 0,
  "output_path": "/path/to/research_folder/.nova/guidelines_filenames.json",
  "message": "Successfully extracted URLs from article guidelines..."
}
```

## 4. Step 2: Processing Local Files

The second tool handles local file references found in the guidelines. It copies referenced files to an organized folder structure and formats them for LLM consumption.

### 4.1 Local File Processing Implementation

The `process_local_files_tool` handles local file references found in the guidelines, copying them to an organized folder structure and converting notebooks to LLM-friendly formats.

Source: _research_agent_part_2/mcp_server/src/tools/process_local_files_tool.py_

```python
def process_local_files_tool(research_directory: str) -> Dict[str, Any]:
    """
    Process local files referenced in the article guidelines.

    Reads the guidelines JSON file and copies each referenced local file
    to the local files subfolder. Path separators in filenames are
    replaced with underscores to avoid creating nested folders.

    Args:
        research_directory: Path to the research directory containing the guidelines JSON file

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Processing local files from research folder: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Look for GUIDELINES_FILENAMES_FILE
    metadata_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(metadata_path)

    # Load JSON metadata
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    local_files = data.get("local_files", [])

    if not local_files:
        return {
            "status": "success",
            "message": f"No local files to process in research folder '{research_directory}'.",
            "files_processed": 0,
            "files_total": 0,
            "warnings": [],
            "errors": [],
        }

    # Create destination folder if it doesn't exist
    dest_folder = nova_path / LOCAL_FILES_FROM_RESEARCH_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    warnings = []
    errors = []
    processed_files = []

    # Initialize notebook converter for .ipynb files
    notebook_converter = NotebookToMarkdownConverter(include_outputs=True, include_metadata=False)

    for rel_path in local_files:
        # Local files are relative to the research folder
        src_path = research_path / rel_path

        if not src_path.exists():
            warnings.append(f"Referenced local file not found: {rel_path}")
            continue

        # Sanitize destination filename (replace path separators with underscores)
        dest_name = rel_path.replace("/", "_").replace("\\", "_")

        try:
            # Handle .ipynb files specially by converting to markdown
            if src_path.suffix.lower() == ".ipynb":
                # Convert .ipynb to .md extension for destination
                dest_name = dest_name.rsplit(".ipynb", 1)[0] + ".md"
                dest_path = dest_folder / dest_name

                # Convert notebook to markdown string
                markdown_content = notebook_converter.convert_notebook_to_string(src_path)

                # Write markdown content to destination
                dest_path.write_text(markdown_content, encoding="utf-8")
            else:
                # For other file types, copy as before
                dest_path = dest_folder / dest_name
                shutil.copy2(src_path, dest_path)

            processed += 1
            processed_files.append(dest_name)
        except Exception as e:
            errors.append(f"Failed to process {rel_path}: {str(e)}")

    # Build result message using the dedicated function
    result_message = build_result_message(research_directory, processed, local_files, dest_folder, warnings, errors)

    return {
        "status": "success" if processed > 0 else "warning",
        "files_processed": processed,
        "files_total": len(local_files),
        "processed_files": processed_files,
        "warnings": warnings,
        "errors": errors,
        "output_directory": str(dest_folder.resolve()),
        "message": result_message,
    }
```

Key features of the local file processing tool:

1. **Path Sanitization**: Converts path separators (`/`, `\`) to underscores to create flat file structure
2. **Jupyter Notebook Conversion**: Uses `NotebookToMarkdownConverter` to transform `.ipynb` files into LLM-readable markdown format
3. **Graceful Error Handling**: Continues processing other files even if some fail, collecting warnings and errors
4. **Detailed Reporting**: Provides comprehensive feedback about what was processed, what failed, and why
5. **Relative Path Resolution**: Resolves local file paths relative to the research directory

The notebook conversion process includes:
- Cell content extraction with proper formatting
- Output preservation for executed cells
- Metadata filtering to focus on content
- Markdown structure that's optimized for LLM consumption

This approach ensures that referenced code files, notebooks, and documentation become part of the research context while maintaining their readability and structure.

## 5. Step 2: Web Scraping with Firecrawl and LLM Cleaning

The most complex tool in our data ingestion pipeline scrapes web URLs and cleans the content using both external services and LLM processing.

### 5.1 Understanding the Scraping Architecture

Source: _research_agent_part_2/mcp_server/src/tools/scrape_and_clean_other_urls_tool.py_

```python
async def scrape_and_clean_other_urls_tool(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape and clean other URLs from guidelines file in the research folder.
    
    Reads the guidelines file and scrapes/cleans each URL listed
    under 'other_urls'. The cleaned markdown content is saved to the
    URLS_FROM_GUIDELINES_FOLDER subfolder with appropriate filenames.
    """
    logger.debug(f"Scraping and cleaning other URLs from research folder: {research_directory}")
    
    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER
    
    # Validate folders and files
    validate_research_folder(research_path)
    
    # Look for GUIDELINES_FILENAMES_FILE file
    guidelines_file_path = nova_path / GUIDELINES_FILENAMES_FILE
    validate_guidelines_filenames_file(guidelines_file_path)
    
    # Read the guidelines filenames file
    guidelines_data = json.loads(read_file_safe(guidelines_file_path))
    urls_to_scrape = guidelines_data.get("other_urls", [])
    
    if not urls_to_scrape:
        return {
            "status": "success",
            "urls_processed": [],
            "urls_failed": [],
            "total_urls": 0,
            "successful_urls_count": 0,
            "failed_urls_count": 0,
            "output_directory": str(nova_path / URLS_FROM_GUIDELINES_FOLDER),
            "message": "No other URLs found to scrape in the guidelines filenames file."
        }
    
    # Create output directory
    output_dir = nova_path / URLS_FROM_GUIDELINES_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read article guidelines for context
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    guidelines_content = read_file_safe(guidelines_path)
    
    # Scrape URLs concurrently
    completed_results = await scrape_urls_concurrently(
        urls_to_scrape, 
        concurrency_limit, 
        guidelines_content
    )
    
    # Write results to files
    saved_files, successful_scrapes = write_scraped_results_to_files(completed_results, output_dir)
    
    # Calculate statistics
    failed_urls = [res["url"] for res in completed_results if not res.get("success", False)]
    successful_urls = [res["url"] for res in completed_results if res.get("success", False)]
    
    return {
        "status": "success",
        "urls_processed": successful_urls,
        "urls_failed": failed_urls,
        "total_urls": len(urls_to_scrape),
        "successful_urls_count": successful_scrapes,
        "failed_urls_count": len(failed_urls),
        "output_directory": str(output_dir),
        "message": f"Successfully processed {successful_scrapes}/{len(urls_to_scrape)} URLs. "
                  f"Results saved to: {output_dir}"
    }
```

### 5.2 The Two-Stage Cleaning Process

The scraping process uses a sophisticated two-stage approach:

1. **Firecrawl for Initial Scraping**: Firecrawl is a specialized service that handles the complexity of modern web scraping, including:
   - JavaScript rendering
   - Dynamic content loading  
   - Anti-bot protection
   - Clean markdown extraction

2. **LLM for Content Refinement**: After Firecrawl extracts the raw content, an LLM (GPT-4o-mini) further cleans and structures the content by:
   - Removing irrelevant sections (ads, navigation, footers)
   - Focusing on content relevant to the article guidelines
   - Maintaining proper markdown formatting
   - Preserving important links and references

The LLM cleaning process is handled by the `clean_markdown` function:

Source: _research_agent_part_2/mcp_server/src/app/scraping_handler.py_

```python
async def clean_markdown(
    markdown_content: str, article_guidelines: str, url_for_log: str, chat_model: BaseChatModel
) -> str:
    """Clean markdown content via LLM and convert image syntax to URLs."""
    if not markdown_content.strip():
        return markdown_content

    prompt_text = PROMPT_CLEAN_MARKDOWN.format(article_guidelines=article_guidelines, markdown_content=markdown_content)
    timeout_seconds = 180  # 3 minutes timeout for LLM call

    try:
        # Add timeout to LLM API call
        response = await asyncio.wait_for(chat_model.ainvoke(prompt_text), timeout=timeout_seconds)
        cleaned_content = response.content if hasattr(response, "content") else str(response)

        if isinstance(cleaned_content, list):
            cleaned_content = "".join(str(part) for part in cleaned_content)

        # Post-process: convert markdown images to just URLs
        cleaned_content = convert_markdown_images_to_urls(cleaned_content)

        return cleaned_content
    except asyncio.TimeoutError:
        logger.error(f"LLM API call timed out after {timeout_seconds}s for {url_for_log}. Using original content.")
        return markdown_content
    except Exception as e:
        logger.error(f"Error cleaning markdown for {url_for_log}: {e}. Using original content.", exc_info=True)
        return markdown_content
```

This function demonstrates several important patterns:

1. **Context-Aware Cleaning**: Uses the article guidelines as context to help the LLM understand what content is relevant
2. **Robust Error Handling**: Falls back to original content if LLM processing fails
3. **Timeout Management**: Prevents hanging on slow LLM responses
4. **Post-Processing**: Converts markdown image syntax to plain URLs for better LLM consumption

The cleaning process significantly reduces token count while preserving the most relevant information for research purposes.

The concurrent scraping orchestration is handled by the `scrape_urls_concurrently` function:

Source: _research_agent_part_2/mcp_server/src/app/scraping_handler.py_

```python
async def scrape_urls_concurrently(
    other_urls: List[str], article_guidelines: str, concurrency_limit: int = 4
) -> List[dict]:
    """
    Scrape and clean multiple URLs concurrently.

    Args:
        other_urls: List of URLs to scrape
        article_guidelines: Guidelines content for cleaning scraped data
        concurrency_limit: Maximum number of concurrent tasks

    Returns:
        List of scraping results, each containing the scraped data or error information
    """
    # Initialize clients
    firecrawl_app = AsyncFirecrawl(api_key=settings.firecrawl_api_key.get_secret_value())
    chat_model = get_chat_model(settings.scraping_model)
    logger.debug(f"Starting scraping of {len(other_urls)} URL(s) with a concurrency limit of {concurrency_limit}...")

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def scrape_with_semaphore(url: str, guidelines: str) -> dict:
        async with semaphore:
            return await scrape_and_clean(url, guidelines, firecrawl_app, chat_model)

    # Process URLs concurrently
    tasks = [scrape_with_semaphore(url, article_guidelines) for url in other_urls]
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    processed_results = []
    for i, res in enumerate(completed_results):
        url_for_processing = other_urls[i]

        if isinstance(res, Exception):
            logger.error(f"✗ Unexpected error processing {url_for_processing}: {res}", exc_info=True)
            res = {
                "url": url_for_processing,
                "title": "Unexpected Processing Error",
                "markdown": f"An unexpected error occurred for {url_for_processing}:\n\n{res}",
                "success": False,
            }
        assert isinstance(res, dict)
        processed_results.append(res)

    return processed_results
```

The individual URL processing combines Firecrawl scraping with LLM cleaning:

```python
async def scrape_and_clean(url: str, article_guidelines: str, firecrawl_app: AsyncFirecrawl, chat_model) -> dict:
    """Scrape and clean a single URL, returning dict with url, title, markdown."""
    scraped = await scrape_url(url, firecrawl_app)
    status_marker = "✓" if scraped["success"] else "✗"
    number_of_tokens = chat_model.get_num_tokens(scraped["markdown"])
    token_info = f" ({number_of_tokens} tokens)"
    logger.debug(f"📥 Scraped: {url} {status_marker}{token_info}")
    if scraped["success"]:
        cleaned_md = await clean_markdown(scraped["markdown"], article_guidelines, url, chat_model)
        scraped["markdown"] = cleaned_md
        number_of_tokens = chat_model.get_num_tokens(scraped["markdown"])
        token_info = f" (tokens reduced to {number_of_tokens})"
        logger.debug(f"🧼 Cleaned: {url} {token_info}")
    return scraped
```

The Firecrawl scraping function handles the complexity of modern web scraping:

```python
async def scrape_url(url: str, firecrawl_app: AsyncFirecrawl) -> dict:
    """
    Scrape a URL using Firecrawl with retries and return a dict with url, title, markdown.

    Uses maxAge=1 week for 500% faster scraping by leveraging cached data when available.
    This optimization significantly improves performance for documentation, articles, and
    relatively static content while maintaining freshness within acceptable limits.
    """
    max_retries = 3
    base_delay = 5  # seconds
    timeout_seconds = 120000  # 2 minutes timeout per request

    for attempt in range(max_retries):
        try:
            # Add timeout to individual Firecrawl request
            # Use maxAge=1 week for 500% faster scraping with cached data
            res = await firecrawl_app.scrape(
                url, formats=["markdown"], maxAge=MAX_AGE_ONE_WEEK, timeout=timeout_seconds
            )
            title = res.metadata.title if res and res.metadata and res.metadata.title else "N/A"
            markdown_content = res.markdown if res and res.markdown else ""
            return {"url": url, "title": title, "markdown": markdown_content, "success": True}
        except asyncio.TimeoutError:
            error_msg = f"⚠️ Firecrawl request timed out after {timeout_seconds}s for {url}"
            logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"{error_msg} after {max_retries} attempts")
                return {
                    "url": url,
                    "title": "Scraping Timeout",
                    "markdown": f"{error_msg} after {max_retries} attempts.",
                    "success": False,
                }
        except Exception as e:
            # print the error with traceback
            logger.error(f"Error scraping {url}: {e}", exc_info=True)

            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"⚠️ Error scraping {url} (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                msg = f"⚠️ Error scraping {url} after {max_retries} attempts: {e}"
                logger.error(msg, exc_info=True)
                return {
                    "url": url,
                    "title": "Scraping Failed",
                    "markdown": msg,
                    "success": False,
                }
    
    return {
        "url": url,
        "title": "Scraping Failed",
        "markdown": f"⚠️ Error scraping {url} after {max_retries} attempts.",
        "success": False,
    }
```

Key features of the scraping architecture:

1. **Concurrency Control**: Uses semaphores to limit concurrent requests and respect API limits
2. **Retry Logic**: Implements exponential backoff for failed requests
3. **Caching Optimization**: Uses week-long caching for 500% performance improvement on static content
4. **Comprehensive Error Handling**: Gracefully handles timeouts, network errors, and API failures
5. **Token Tracking**: Monitors content size before and after cleaning to show efficiency gains
6. **Status Logging**: Provides detailed logging for debugging and monitoring

### 5.3 Why Use External Scraping Services?

Web scraping is notoriously complex due to:

- **Dynamic Content**: Modern websites heavily use JavaScript
- **Anti-Bot Measures**: CAPTCHAs, rate limiting, IP blocking
- **Varied Formats**: Inconsistent HTML structures across sites
- **Performance Issues**: Slow loading, timeouts, redirects

Rather than building a robust scraper from scratch (which would require significant effort and still fall short), using a specialized service like Firecrawl allows us to:

- Focus on our core research logic
- Get reliable results across diverse websites  
- Benefit from ongoing improvements to the scraping infrastructure
- Handle edge cases that would be time-consuming to solve ourselves

Let's test the scraping tool:

```python
from research_agent_part_2.mcp_server.src.tools import scrape_and_clean_other_urls_tool

# Test the scraping tool
result = await scrape_and_clean_other_urls_tool(research_directory=research_folder, concurrency_limit=2)
print(result)
```

## 6. Step 2: Processing GitHub URLs

For GitHub repositories, we use a different approach optimized for code analysis.

### 6.1 Using GitIngest for Repository Processing

The `process_github_urls_tool` leverages the `gitingest` library to extract comprehensive information from GitHub repositories, making code and documentation available for research purposes.

Source: _research_agent_part_2/mcp_server/src/tools/process_github_urls_tool.py_

```python
async def process_github_urls_tool(research_directory: str) -> Dict[str, Any]:
    """
    Process GitHub URLs from guidelines file in the research folder.

    Reads the guidelines file and processes each URL listed
    under 'github_urls' using gitingest to extract repository summaries, file trees,
    and content. The results are saved as markdown files in the
    URLS_FROM_GUIDELINES_CODE_FOLDER subfolder.

    Args:
        research_directory: Path to the research folder containing the guidelines file

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Processing GitHub URLs from research folder: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Ensure the NOVA_FOLDER directory exists
    nova_path.mkdir(parents=True, exist_ok=True)

    # Look for GUIDELINES_FILENAMES_FILE file
    metadata_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(metadata_path)

    # Read the guidelines JSON file
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (IOError, OSError, json.JSONDecodeError) as e:
        msg = f"Error reading {GUIDELINES_FILENAMES_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    # Get the github_urls list
    github_urls: list[str] = data.get("github_urls", [])

    if not github_urls:
        return {
            "status": "success",
            "message": f"No GitHub URLs found in {GUIDELINES_FILENAMES_FILE} in '{research_directory}'",
            "urls_processed": 0,
            "urls_total": 0,
            "files_saved": 0,
        }

    # Prepare output directory
    dest_folder = nova_path / URLS_FROM_GUIDELINES_CODE_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Processing {len(github_urls)} GitHub URLs...")

    # Process GitHub URLs sequentially
    success_count = 0
    for url in github_urls:
        try:
            result = await process_github_url(url, dest_folder, settings.github_token.get_secret_value())
            if result:
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing GitHub URL {url}: {e}")
            continue

    return {
        "status": "success" if success_count > 0 else "warning",
        "urls_processed": success_count,
        "urls_total": len(github_urls),
        "files_saved": success_count,
        "output_directory": str(dest_folder.resolve()),
        "message": (
            f"Processed {success_count}/{len(github_urls)} GitHub URLs from {GUIDELINES_FILENAMES_FILE} "
            f"in '{research_directory}'. Saved markdown summaries to {URLS_FROM_GUIDELINES_CODE_FOLDER} folder."
        ),
    }
```

Key features of the GitHub processing tool:

1. **Sequential Processing**: Unlike web scraping, GitHub processing is done sequentially to respect API rate limits
2. **Token Authentication**: Uses GitHub tokens for accessing private repositories when available
3. **Error Resilience**: Continues processing other URLs even if one fails
4. **Comprehensive Output**: Each repository generates a detailed markdown file with:
   - Repository metadata and structure
   - File trees showing project organization
   - Key source files with syntax highlighting
   - Converted notebook content in LLM-friendly format
   - Documentation and README files

```python
from research_agent_part_2.mcp_server.src.tools import process_github_urls_tool

# Test GitHub URL processing
result = process_github_urls_tool(research_directory=research_folder)
print(result)
```

The `gitingest` library handles the complexity of:
- Cloning repositories efficiently
- Parsing different file types appropriately
- Converting Jupyter notebooks to markdown
- Organizing content by importance and relevance

## 7. Step 2: YouTube Video Transcription

For YouTube videos referenced in guidelines, we use Gemini's multimodal capabilities.

### 7.1 Gemini-Based Video Transcription

The `transcribe_youtube_videos_tool` leverages Gemini's multimodal capabilities to process video content directly and generate structured transcripts for research purposes.

Source: _research_agent_part_2/mcp_server/src/tools/transcribe_youtube_videos_tool.py_

```python
async def transcribe_youtube_videos_tool(research_directory: str) -> Dict[str, Any]:
    """
    Transcribe YouTube video URLs from GUIDELINES_JSON_FILE using Gemini 2.5 Pro.

    Reads the GUIDELINES_JSON_FILE file and processes each URL listed
    under 'youtube_videos_urls'. Each video is transcribed, and the results are
    saved as markdown files in the URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder.

    Args:
        research_directory: Path to the research directory containing GUIDELINES_JSON_FILE

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Starting transcription of YouTube videos from {research_directory}")

    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Look for GUIDELINES_FILENAMES_FILE
    metadata_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(metadata_path)

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (IOError, OSError, json.JSONDecodeError) as e:
        msg = f"Error reading {GUIDELINES_FILENAMES_FILE}: {str(e)}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e

    youtube_urls: list[str] = data.get("youtube_videos_urls", [])

    if not youtube_urls:
        return {
            "status": "success",
            "videos_processed": 0,
            "videos_total": 0,
            "message": f"No YouTube URLs found in {GUIDELINES_FILENAMES_FILE} in '{research_directory}'",
        }

    dest_folder = nova_path / URLS_FROM_GUIDELINES_YOUTUBE_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Processing {len(youtube_urls)} YouTube URL(s)...")

    semaphore = asyncio.Semaphore(YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS)
    tasks = [process_youtube_url(url, dest_folder, semaphore) for url in youtube_urls]
    await asyncio.gather(*tasks)

    return {
        "status": "success",
        "videos_processed": len(youtube_urls),
        "videos_total": len(youtube_urls),
        "output_directory": str(dest_folder.resolve()),
        "message": (
            f"Processed {len(youtube_urls)} YouTube URLs from {GUIDELINES_FILENAMES_FILE} "
            f"in '{research_directory}'. Saved transcriptions to {dest_folder.name} folder."
        ),
    }
```

Key features of the YouTube transcription tool:

1. **Concurrent Processing**: Uses a semaphore to limit concurrent requests while maintaining efficiency
2. **Gemini Integration**: Leverages Gemini 2.0 Flash's ability to process video content directly
3. **Structured Output**: Generates organized transcripts with:
   - Video metadata (title, duration, channel)
   - Timestamped sections for easy navigation
   - Key points and topics identified
   - Content relevant to the article guidelines

4. **Rate Limiting**: Respects API limits with `YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS`

The transcription process:
1. **Extracts video content** using Gemini's multimodal input
2. **Generates structured transcripts** with timestamps and key points  
3. **Identifies relevant segments** based on article guidelines
4. **Formats for research use** with clear section breaks

```python
from research_agent_part_2.mcp_server.src.tools import transcribe_youtube_videos_tool

# Test YouTube transcription (note: this can be time-consuming)
result = await transcribe_youtube_videos_tool(research_directory=research_folder)
print(result)
```

**Note**: Video transcription is time-intensive. A 39-minute video typically takes about 4.5 minutes to process. The tool processes videos concurrently but with controlled concurrency to respect API limits and avoid overwhelming the service.

## 8. Running the Full Agent with MCP Prompt

Now let's see how these tools work together in the complete research workflow using the MCP client.

```python
# Run the MCP client in-kernel
from research_agent_part_2.mcp_client.src.client import main as client_main
import sys

async def run_client():
    _argv_backup = sys.argv[:]
    sys.argv = ["client"]
    try:
        await client_main()
    finally:
        sys.argv = _argv_backup

# Start client with in-memory server 
await run_client()
```

Once the client is running, you can:

1. **Start the workflow**: Type `/prompt/full_research_instructions_prompt` to load the complete research workflow
2. **Provide research directory**: Give the path to your sample research folder
3. **Watch the agent work**: Observe how it runs the tools in sequence
4. **Examine outputs**: Check the `.nova` folder for generated files

Try these commands in sequence:
- `/prompt/full_research_instructions_prompt`
- Provide your research directory path when asked
- Watch the agent execute steps 1 and 2 of the workflow

## 9. Error Handling and Failure Cases

The research agent is designed to handle various error scenarios gracefully. Let's explore how it behaves with different types of failures.

### 9.1 Non-Critical Failures

When some URLs fail to scrape but others succeed, the agent continues the workflow:

```python
# Example of partial failure handling
result = {
    "status": "success",
    "urls_processed": ["https://example1.com", "https://example2.com"],
    "urls_failed": ["https://broken-link.com"],
    "total_urls": 3,
    "successful_urls_count": 2,
    "failed_urls_count": 1,
    "message": "Successfully processed 2/3 URLs. Results saved to output directory."
}
```

The agent recognizes this as a partial success and continues with available data.

### 9.2 Critical Failures

According to the MCP prompt instructions, the agent stops only for "complete failures" - when zero items are processed successfully:

```python
# Example of critical failure
result = {
    "status": "success", 
    "urls_processed": [],
    "urls_failed": ["https://url1.com", "https://url2.com", "https://url3.com"],
    "total_urls": 3,
    "successful_urls_count": 0,
    "failed_urls_count": 3,
    "message": "Failed to process any URLs. All 3 URLs failed to scrape."
}
```

In this case, the agent would:
1. State the exact tool that failed and quote the output message
2. Announce that it's stopping the workflow per its instructions  
3. Ask the user for guidance on how to proceed

### 9.3 MCP Prompt Failure Policy

The research agent follows specific instructions defined in the MCP prompt for handling failures. Let's examine the exact policy from the prompt:

Source: _research_agent_part_2/mcp_server/src/prompts/research_instructions_prompt.py_

```
**Critical Failure Policy:**

If a tool reports a complete failure, you are required to halt the entire workflow immediately. A complete failure
is defined as processing zero items successfully (e.g., scraped 0/7 URLs, processed 0 files).

If this occurs, your immediate and only action is to:
    1. State the exact tool that failed and quote the output message.
    2. Announce that you are stopping the workflow as per your instructions.
    3. Ask the user for guidance on how to proceed.
```

This policy demonstrates several important design principles:

1. **Clear Failure Definition**: "Complete failure" is precisely defined as processing zero items successfully, not just encountering some errors.

2. **Immediate Halt**: The agent must stop the entire workflow immediately when a critical failure occurs, preventing wasted resources on subsequent steps.

3. **Transparent Communication**: The agent must quote the exact error message and explain why it's stopping, ensuring the user understands the situation.

4. **Human Escalation**: Rather than attempting to recover automatically, the agent asks for human guidance, recognizing that critical failures often require human judgment.

5. **Graceful Degradation**: Partial failures (e.g., 5/7 URLs scraped successfully) are acceptable and allow the workflow to continue with available data.

This approach balances automation with reliability—the agent continues working through minor issues but escalates major problems that could compromise the entire research effort.

### 9.4 Testing Error Scenarios

To test error handling, you can modify the sample article guideline to include:
- Non-existent local files
- Invalid URLs
- Private repositories without proper tokens

The agent will demonstrate different responses based on the severity of failures:

- **Non-Critical**: Some files fail to process, but others succeed → workflow continues
- **Critical**: All items in a processing step fail → workflow halts and asks for guidance

## 10. Exploring Generated Files

After running the tools, examine the organized file structure in your research directory:

```
research_directory/
├── article_guideline.md                     # Input guidelines
├── .nova/                                   # Hidden folder with all data
│   ├── guidelines_filenames.json           # Extracted URLs and files
│   ├── local_files_from_research/          # Copied local files  
│   ├── urls_from_guidelines/               # Scraped web content
│   ├── urls_from_guidelines_code/          # GitHub repo summaries
│   └── urls_from_guidelines_youtube/       # Video transcripts
```

Each folder contains processed content ready for the next stages of the research workflow. The file-based approach ensures that:

- **Content is persistent** across agent sessions
- **Large content blocks** don't overwhelm the agent's context
- **Selective access** allows the agent to read only relevant files
- **Human inspection** is possible for debugging and verification

## 11. Key Takeaways

This lesson demonstrated several important principles for building robust data ingestion tools:

1. **File-Based Outputs**: Keep tool responses concise and save detailed content to files
2. **External Services**: Use specialized services (Firecrawl, GitIngest) for complex tasks
3. **Parallel Processing**: Implement concurrency for efficient data collection
4. **Graceful Degradation**: Handle partial failures without stopping the entire workflow
5. **LLM Enhancement**: Use LLMs to clean and structure scraped content
6. **Organized Storage**: Create clear folder structures for different content types

These patterns form the foundation for scalable research automation and can be adapted for various data ingestion scenarios.

In the next lesson, we'll explore how the agent continues the workflow by generating targeted research queries and using advanced search capabilities to fill knowledge gaps.
