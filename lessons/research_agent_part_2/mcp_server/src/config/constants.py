"""Constants used throughout the MCP server."""

# File and folder names
ARTICLE_GUIDELINE_FILE = "article_guideline.md"
GUIDELINES_FILENAMES_FILE = "guidelines_filenames.json"
PERPLEXITY_RESULTS_FILE = "perplexity_results.md"
PERPLEXITY_RESULTS_SELECTED_FILE = "perplexity_results_selected.md"
PERPLEXITY_SOURCES_SELECTED_FILE = "perplexity_sources_selected.md"
NEXT_QUERIES_FILE = "next_queries.md"
URLS_TO_SCRAPE_FROM_RESEARCH_FILE = "urls_to_scrape_from_research.md"
RESEARCH_MD_FILE = "research.md"
NOVA_FOLDER = ".nova"
URLS_FROM_GUIDELINES_FOLDER = "urls_from_guidelines"
URLS_FROM_GUIDELINES_CODE_FOLDER = "urls_from_guidelines_code"
URLS_FROM_GUIDELINES_YOUTUBE_FOLDER = "urls_from_guidelines_youtube_videos"
LOCAL_FILES_FROM_RESEARCH_FOLDER = "local_files_from_research"
URLS_FROM_RESEARCH_FOLDER = "urls_from_research"

# File extensions
MARKDOWN_EXTENSION = ".md"
JSON_EXTENSION = ".json"

# YouTube operation constants
YOUTUBE_TRANSCRIPTION_MAX_CONCURRENT_REQUESTS = 2
YOUTUBE_TRANSCRIPTION_MAX_RETRIES = 3
YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MIN_SECONDS = 5
YOUTUBE_TRANSCRIPTION_RETRY_WAIT_MAX_SECONDS = 60
