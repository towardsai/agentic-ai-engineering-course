# Lesson 18: Final Outputs and Agent Completion

In our final lesson, we will complete the research agent's workflow by implementing the remaining tools that filter research results, scrape additional sources, and compile everything into a final research file. We'll test the complete end-to-end agent with different article guidelines and analyze the quality of outputs. We'll conclude with a discussion on extensibility and deployment considerations for production use.

Learning Objectives:
- Learn how to filter and validate research sources for quality and trustworthiness
- Understand how to select the most valuable sources for full content scraping
- Implement the final research compilation tool that creates structured outputs
- Test the complete agent workflow and analyze output quality
- Explore extensibility options and deployment considerations

## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:

```python
%load_ext autoreload
%autoreload 2
```

### Set Up Python Environment

To set up your Python virtual environment using `uv` and load it into the Notebook, follow the step-by-step instructions from the `Course Admin` lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.

### Configure Required APIs

To run this lesson, you'll need several API keys configured:

1. **Gemini API Key**, `GOOGLE_API_KEY` variable: Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. **Perplexity API Key**, `PPLX_API_KEY` variable: Get your key from [Perplexity](https://www.perplexity.ai/settings/api).
3. **Firecrawl API Key**, `FIRECRAWL_API_KEY` variable: Get your key from [Firecrawl](https://firecrawl.dev/). They have a free tier that allows you to scrape 500 pages.

```python
from utils import env

env.load(required_env_vars=["GOOGLE_API_KEY", "PPLX_API_KEY", "FIRECRAWL_API_KEY"])
```

### Import Key Packages

```python
import nest_asyncio
nest_asyncio.apply() # Allow nested async usage in notebooks
```

## 2. Completing the Research Workflow

As we've seen in previous lessons, our research agent follows a systematic workflow. We've covered the initial data ingestion (lesson 16) and the research loop with query generation (lesson 17). Now we need to implement the final steps that ensure quality and compile the results.

The complete workflow includes these final steps:

```markdown
4. Filter Perplexity results by quality:
    4.1 Run the "select_research_sources_to_keep" tool to automatically evaluate each source 
    for trustworthiness, authority and relevance.

5. Identify which of the accepted sources deserve a full scrape:
    5.1 Run the "select_research_sources_to_scrape" tool to choose up to 5 diverse, 
    authoritative sources whose full content will add most value.
    5.2 Run the "scrape_research_urls" tool to scrape/clean each selected URL's full content.

6. Write final research file:
    6.1 Run the "create_research_file" tool to combine all research data into a 
    comprehensive research.md file.
```

Let's examine each of these final tools and understand their purpose in the workflow.

## 3. Filtering Research Sources for Quality

The `select_research_sources_to_keep` tool addresses a critical problem we discovered during development: Perplexity results often include sources from untrustworthy blogs, SEO spam, or low-quality content that would pollute our research.

### 3.1 Understanding the Tool Implementation

This tool takes a research directory as input and automatically filters Perplexity results for quality. It reads the article guidelines and raw Perplexity results, then uses an LLM to evaluate each source based on trustworthiness, authority, and relevance criteria. The tool outputs two files: a list of selected source IDs and a filtered markdown file containing only the approved sources. This automated filtering saves time while ensuring research quality.

Source: _mcp_server/src/tools/select_research_sources_to_keep_tool.py_

```python
async def select_research_sources_to_keep_tool(research_directory: str) -> Dict[str, Any]:
    """
    Automatically select high-quality sources from Perplexity results.
    
    Uses GPT-4.1 to evaluate each source in perplexity_results.md for trustworthiness,
    authority, and relevance based on the article guidelines. Writes the comma-separated
    IDs of accepted sources to perplexity_sources_selected.md and saves a filtered
    markdown file perplexity_results_selected.md containing only the accepted sources.
    """
    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER
    
    # Gather context from the research folder
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    results_path = nova_path / PERPLEXITY_RESULTS_FILE
    
    article_guidelines = read_file_safe(guidelines_path)
    perplexity_results = read_file_safe(results_path)
    
    # Use LLM to select sources
    selected_ids = await select_sources(
        article_guidelines, perplexity_results, settings.source_selection_model
    )
    
    # Write selected source IDs to file
    sources_selected_path = nova_path / PERPLEXITY_SOURCES_SELECTED_FILE
    sources_selected_path.write_text(",".join(map(str, selected_ids)), encoding="utf-8")
    
    # Extract and save filtered content
    filtered_content = extract_selected_blocks_content(selected_ids, perplexity_results)
    results_selected_path = nova_path / PERPLEXITY_RESULTS_SELECTED_FILE
    results_selected_path.write_text(filtered_content, encoding="utf-8")
    
    return {
        "status": "success",
        "sources_selected_count": len(selected_ids),
        "selected_source_ids": selected_ids,
        "sources_selected_path": str(sources_selected_path.resolve()),
        "results_selected_path": str(results_selected_path.resolve()),
        "message": f"Successfully selected {len(selected_ids)} high-quality sources..."
    }
```

### 3.2 The Source Evaluation Prompt

The tool uses a sophisticated prompt to evaluate source quality:

Source: _mcp_server/src/config/prompts.py_

```python
PROMPT_SELECT_SOURCES = """
You are a research quality evaluator. Your task is to evaluate web sources for an upcoming article
and select only the high-quality, trustworthy sources that are relevant to the article guidelines.

<article_guidelines>
{article_guidelines}
</article_guidelines>

Here are the sources to evaluate:
<sources_to_evaluate>
{sources_data}
</sources_to_evaluate>

**Selection Criteria:**
- ACCEPT sources from trustworthy domains (e.g., .edu, .gov, established news sites,
official documentation, reputable organizations)
- ACCEPT sources with high-quality, relevant content that directly supports the article guidelines
- REJECT sources from obscure, untrustworthy, or potentially biased websites
- REJECT sources with low-quality, irrelevant, or superficial content
- REJECT sources that seem to be marketing materials, advertisements, or self-promotional content

Return your decision as a structured output with:
1. selection_type: "none" if no sources meet the quality standards, "all" if all sources are acceptable,
or "specific" for specific source IDs
2. source_ids: List of selected source IDs
""".strip()
```

This prompt serves as a quality gatekeeper, automatically filtering out unreliable sources that could compromise research integrity. The key innovation is the structured selection criteria that balance domain reputation, content quality, and relevance. The three-tier selection system (none/all/specific) provides flexibility for different quality scenarios. The prompt explicitly targets common quality issues like SEO spam, marketing content, and biased sources that often pollute web search results, ensuring only authoritative sources proceed to the next stage.

### 3.3 Testing the Source Selection Tool

Let's test the source filtering tool to see how it evaluates and selects high-quality sources from our Perplexity results. The tool will analyze each source and provide feedback on which ones meet our quality standards.

```python
from research_agent_part_2.mcp_server.src.tools import select_research_sources_to_keep_tool

# Test the source selection tool
research_folder = "/path/to/research_folder"
result = await select_research_sources_to_keep_tool(research_directory=research_folder)
print(result)
```

The tool provides feedback on which sources were selected, allowing users to review the decisions if needed.

## 4. Selecting Sources for Full Content Scraping

After filtering for quality, we need to identify which sources deserve a full scrape. While Perplexity provides summaries and excerpts, some high-quality sources contain much more valuable content in their full form. The `select_research_sources_to_scrape` tool analyzes the filtered results and strategically chooses the most valuable sources for comprehensive content extraction. This full content will provide the writing agent with richer context, detailed examples, and comprehensive coverage that brief excerpts cannot capture.

### 4.1 Understanding the Selection Logic

This tool takes filtered Perplexity results and selects the most valuable sources for full scraping. It analyzes the article guidelines, accepted sources, and already-scraped guideline content to avoid duplication. The tool uses an LLM to evaluate sources based on relevance, authority, quality, and uniqueness, then outputs a prioritized list of URLs. The default limit of 5 sources balances comprehensive coverage with processing efficiency and API costs.

Source: _mcp_server/src/tools/select_research_sources_to_scrape_tool.py_

```python
async def select_research_sources_to_scrape_tool(research_directory: str, max_sources: int = 5) -> Dict[str, Any]:
    """
    Select up to max_sources priority research sources to scrape in full.
    
    Analyzes the filtered Perplexity results together with the article guidelines and
    the material already scraped from guideline URLs, then chooses up to max_sources diverse,
    authoritative sources whose full content will add most value. The chosen URLs are
    written (one per line) to urls_to_scrape_from_research.md.
    """
    # Gather context from the research folder
    guidelines_path = research_path / ARTICLE_GUIDELINE_FILE
    results_selected_path = nova_path / PERPLEXITY_RESULTS_SELECTED_FILE
    
    article_guidelines = read_file_safe(guidelines_path)
    accepted_sources_data = read_file_safe(results_selected_path)
    scraped_guideline_ctx = load_scraped_guideline_context(nova_path)
    
    # Use LLM to select top sources for scraping
    selected_urls, reasoning = await select_top_sources(
        article_guidelines, accepted_sources_data, scraped_guideline_ctx, max_sources
    )
    
    # Write selected URLs to file
    urls_out_path = nova_path / URLS_TO_SCRAPE_FROM_RESEARCH_FILE
    urls_out_path.write_text("\n".join(selected_urls) + "\n", encoding="utf-8")
    
    return {
        "status": "success",
        "sources_selected": selected_urls,
        "sources_selected_count": len(selected_urls),
        "output_path": str(urls_out_path.resolve()),
        "reasoning": reasoning,
        "message": f"Successfully selected {len(selected_urls)} sources for full scraping..."
    }
```

### 4.2 The Source Selection Prompt

The tool uses an intelligent prompt to choose the most valuable sources:

```python
PROMPT_SELECT_TOP_SOURCES = """
You are assisting with research for an upcoming article.

Your task is to select the most relevant and trustworthy sources from the web search results.
You should consider:
1. **Relevance**: How well each source addresses the article guidelines
2. **Authority**: The credibility and reputation of the source
3. **Quality**: The depth and accuracy of the information provided
4. **Uniqueness**: Sources that provide unique insights not covered by the scraped guideline URLs

Please select the top {top_n} sources that would be most valuable for the article research.

Return your selection with the following structure:
- **selected_urls**: A list of the most valuable URLs to scrape in full, ordered by priority
- **reasoning**: A short explanation summarizing why these specific URLs were chosen
""".strip()
```

This prompt optimizes resource allocation by strategically selecting sources for expensive full-content scraping. The four-dimensional evaluation framework (relevance, authority, quality, uniqueness) ensures maximum research value while avoiding duplication with already-scraped guideline content. The uniqueness criterion is particularly important as it prevents redundant scraping of similar information. The reasoning requirement provides transparency for human oversight and helps identify potential gaps in the selection logic, making the process auditable and improvable.

### 4.3 Testing the Source Selection Tool

Now let's test the source selection tool to see which URLs it chooses for full content scraping. The tool will analyze our filtered sources and select the most valuable ones based on their potential contribution to the final research.

```python
from research_agent_part_2.mcp_server.src.tools import select_research_sources_to_scrape_tool

# Test selecting sources to scrape
result = await select_research_sources_to_scrape_tool(research_directory=research_folder, max_sources=3)
print("Selected sources:")
print(result)
```

## 5. Scraping Selected Research URLs

The `scrape_research_urls_tool` handles the full content extraction from our selected sources. It works similarly to the guideline URL scraping tool we saw in lesson 16, using Firecrawl for robust web scraping and GPT-5-mini for content cleaning.

### 5.1 Understanding the Scraping Process

This tool reads the selected URLs from the previous step and performs full content scraping using the same robust infrastructure as guideline URL processing. It takes the research directory and concurrency limit as inputs, then uses Firecrawl for web scraping and GPT for content cleaning. The tool outputs cleaned markdown files in the research URLs folder and returns statistics about successful vs. failed scraping attempts, enabling quality monitoring.

Source: _mcp_server/src/tools/scrape_research_urls_tool.py_

```python
async def scrape_research_urls_tool(research_directory: str, concurrency_limit: int = 4) -> Dict[str, Any]:
    """
    Scrape the selected research URLs for full content.
    
    Reads the URLs from urls_to_scrape_from_research.md and scrapes/cleans each URL's
    full content. The cleaned markdown files are saved to the urls_from_research
    subfolder with appropriate filenames.
    """
    # Read URLs to scrape
    urls_file_path = nova_path / URLS_TO_SCRAPE_FROM_RESEARCH_FILE
    urls_to_scrape = urls_file_path.read_text(encoding="utf-8").strip().split("\n")
    
    # Read article guidelines for context
    guidelines_content = read_file_safe(guidelines_path)
    
    # Scrape URLs concurrently using the same infrastructure as guideline scraping
    completed_results = await scrape_urls_concurrently(
        urls_to_scrape, concurrency_limit, guidelines_content
    )
    
    # Write results to files in the research URLs folder
    output_dir = nova_path / URLS_FROM_RESEARCH_FOLDER
    saved_files, successful_scrapes = write_scraped_results_to_files(completed_results, output_dir)
    
    return {
        "status": "success",
        "urls_processed": successful_urls,
        "total_urls": len(urls_to_scrape),
        "successful_urls_count": successful_scrapes,
        "output_directory": str(output_dir),
        "message": f"Successfully scraped {successful_scrapes}/{len(urls_to_scrape)} research URLs..."
    }
```

This tool reuses the same robust scraping infrastructure we developed for guideline URLs, ensuring consistent quality and reliability.

### 5.2 Testing the URL Scraping Tool

Let's test the URL scraping tool to see how it processes our selected research sources. This tool will extract the full content from each selected URL and clean it for optimal use by the writing agent.

```python
from research_agent_part_2.mcp_server.src.tools import scrape_research_urls_tool

# Test scraping the selected research URLs
result = await scrape_research_urls_tool(research_directory=research_folder, concurrency_limit=2)
print("Scraping results:")
print(result)
```

## 6. Creating the Final Research File

The `create_research_file_tool` is the culmination of our entire workflow. It takes all the accumulated research data and formats it into a comprehensive, well-organized markdown file that serves as input for the writing agent we'll build in the next part of the course.

### 6.1 Understanding the Compilation Process

This tool serves as the final orchestrator, combining all research data into a comprehensive markdown file. It takes the research directory as input and collects content from multiple sources: filtered Perplexity results, scraped guideline sources, code repositories, YouTube transcripts, and additional research sources. The tool organizes everything into collapsible sections and outputs a structured research.md file with detailed statistics about the compilation process.

Source: _mcp_server/src/tools/create_research_file_tool.py_

```python
def create_research_file_tool(research_directory: str) -> Dict[str, Any]:
    """
    Generate comprehensive research.md file from all research data.
    
    Combines all research data including filtered Perplexity results, scraped guideline
    sources, and full research sources into a comprehensive research.md file. The file
    is organized into sections with collapsible blocks for easy navigation.
    """
    # Convert to Path object
    article_dir = Path(research_directory)
    nova_dir = article_dir / NOVA_FOLDER
    
    # Collect all research data
    perplexity_results = read_file_safe(nova_dir / PERPLEXITY_RESULTS_SELECTED_FILE)
    
    # Collect scraped sources from guidelines
    scraped_sources = collect_directory_markdowns_with_titles(nova_dir / URLS_FROM_GUIDELINES_FOLDER)
    code_sources = collect_directory_markdowns_with_titles(nova_dir / URLS_FROM_GUIDELINES_CODE_FOLDER)
    youtube_transcripts = collect_directory_markdowns_with_titles(nova_dir / URLS_FROM_GUIDELINES_YOUTUBE_FOLDER)
    
    # Collect full research sources
    additional_sources = collect_directory_markdowns_with_titles(nova_dir / URLS_FROM_RESEARCH_FOLDER)
    
    # Build comprehensive research sections
    research_results_section = build_research_results_section(perplexity_results)
    scraped_sources_section = build_sources_section("Scraped Sources from Guidelines", scraped_sources)
    code_sources_section = build_sources_section("Code Sources from Guidelines", code_sources)
    youtube_section = build_sources_section("YouTube Transcripts from Guidelines", youtube_transcripts)
    additional_section = build_sources_section("Additional Research Sources", additional_sources)
    
    # Combine all sections into final research file
    research_content = combine_research_sections([
        research_results_section,
        scraped_sources_section,
        code_sources_section,
        youtube_section,
        additional_section
    ])
    
    # Write final research file
    research_file_path = article_dir / RESEARCH_MD_FILE
    research_file_path.write_text(research_content, encoding="utf-8")
    
    return {
        "status": "success",
        "markdown_file": str(research_file_path.resolve()),
        "research_results_count": len(extract_perplexity_chunks(perplexity_results)),
        "scraped_sources_count": len(scraped_sources),
        "code_sources_count": len(code_sources),
        "youtube_transcripts_count": len(youtube_transcripts),
        "additional_sources_count": len(additional_sources),
        "message": f"Successfully created comprehensive research file: {research_file_path.name}"
    }
```

### 6.2 The Research File Structure

The final research file is organized into collapsible sections for easy navigation:

```markdown
# Research Results

## Research Results from Web Search
<details>
<summary>Query: [Original Query]</summary>

### Source [1]: [URL]
[Content from source]

### Source [2]: [URL]
[Content from source]
</details>

## Scraped Sources from Guidelines
<details>
<summary>Source: [Filename]</summary>
[Full scraped content]
</details>

## Code Sources from Guidelines
<details>
<summary>Repository: [Repository Name]</summary>
[Repository analysis and code content]
</details>

## YouTube Transcripts from Guidelines
<details>
<summary>Video: [Video Title]</summary>
[Full video transcript]
</details>

## Additional Research Sources
<details>
<summary>Source: [URL]</summary>
[Full scraped research content]
</details>
```

This structure provides comprehensive coverage while remaining navigable for both humans and AI writing agents.

### 6.3 Testing the Research File Creation

Now let's test the final compilation tool to see how it brings together all our research data into a comprehensive, well-structured file. This represents the culmination of our entire research workflow.

```python
from research_agent_part_2.mcp_server.src.tools import create_research_file_tool

# Test creating the final research file
result = create_research_file_tool(research_directory=research_folder)
print("Research file creation results:")
print(result)

# Read and display a sample of the generated research file
with open(result["markdown_file"], "r") as f:
    content = f.read()
    print("\nFirst 1000 characters of the research file:")
    print(content[:1000] + "...")
```

## 7. Human-in-the-Loop Feedback Integration

Both the source selection tools support human feedback integration. When running the full workflow, users can instruct the agent to:

- Show the sources selected by `select_research_sources_to_keep` and ask for approval
- Display the URLs chosen by `select_research_sources_to_scrape` and allow modifications
- Pause after any step for human review and guidance

This flexibility allows users to maintain control over the research quality while benefiting from the agent's analytical capabilities.

## 8. Testing the Complete Agent Workflow

Now let's test the complete end-to-end research agent workflow. We'll use the MCP client to run the full workflow and examine the results.

```python
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

1. **Start the complete workflow**: Type `/prompt/full_research_instructions_prompt` to load the complete research workflow
2. **Provide the research directory**: Give the path to your research folder
3. **Run the full workflow**: Let the agent execute all 6 steps of the workflow
4. **Examine the final output**: Check the generated `research.md` file

Try these commands in sequence:
- `/prompt/full_research_instructions_prompt`
- `The research folder is /path/to/research_folder. Run the complete workflow from start to finish.`
- `/quit` after the agent completes all steps

## 9. Using Cursor with the MCP Server

Our research agent can also be used directly within Cursor IDE through the MCP protocol. The `mcp_server` folder contains a `.mcp.json.sample` file that shows how to configure Cursor to use our research agent.

Here's the content of the `.mcp.json.sample` file:

```json
{
  "mcpServers": {
    "nova-research": {
      "command": "/path/to/uv",
      "args": [
        "--directory",
        "/path/to/mcp_server",
        "run",
        "-m",
        "src.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "GOOGLE_API_KEY": "...",
        "OPENAI_API_KEY": "...",
        "PPLX_API_KEY": "...",
        "FIRECRAWL_API_KEY": "...",
        "OPIK_API_KEY": "...",
        "OPIK_WORKSPACE": "...",
        "OPIK_PROJECT_NAME": "...",
        "GITHUB_TOKEN": "..."
      }
    }
  }
}
```

To set this up:
1. Copy `.mcp.json.sample` to `.mcp.json` in your Cursor workspace
2. Update the paths to point to your research agent installation
3. Fill in your actual API keys in the `env` section
4. Restart Cursor to load the MCP server
5. Use the research tools directly in Cursor's chat interface

This integration allows you to conduct research directly within your development environment, making it easy to incorporate findings into your coding projects.