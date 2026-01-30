"""GitHub-specific utilities."""

import logging
import re
from typing import Any, Dict

from gitingest import ingest_async

logger = logging.getLogger(__name__)


async def process_github_url(url: str, token: str | None) -> Dict[str, Any]:
    """
    Fetch a GitHub repository (or file) with gitingest and return the markdown content.

    Args:
        url: GitHub URL to process
        token: GitHub token for authentication (optional)

    Returns:
        Dict with keys:
            - success (bool): Whether the ingestion succeeded
            - url (str): The GitHub URL that was processed
            - markdown (str): The markdown content from GitIngest
    """
    ingestion_succeeded = False
    try:
        summary, tree, content = await ingest_async(url, exclude_patterns="*.lock", token=token)
        ingestion_succeeded = True
        md = f"# Repository analysis for {url}\n\n## Summary\n{summary}\n\n## File tree\n```{tree}\n```\n\n## Extracted content\n{content}"
    except Exception as e:
        md = f"# Error processing {url}\n\n{e}"
        logger.error(f"Error processing repository {url}: {e}", exc_info=True)

    # Regex for markdown-style base64 images: ![...](data:image/...)
    md = re.sub(r"!\[[^\]]*\]\(data:image/[^;]+;base64,[^\)]+\)", "[... base64 image removed ...]", md)
    # Regex for HTML-style base64 images: <img src="data:image/...">
    md = re.sub(r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>', "[... base64 image removed ...]", md)
    # Regex for naked base64 image data starting with common magic numbers.
    md = re.sub(r"(?:iVBOR|/9j/|R0lGOD|UklGR)[A-Za-z0-9+/=\s]{100,}", "[... base64 image removed ...]", md, flags=re.IGNORECASE)

    # Check if content is too long and truncate if necessary
    MAX_CHARS = 65_000
    if len(md) > MAX_CHARS:
        logger.warning(f"⚠️ Warning: Content for {url} is {len(md)} characters, truncating to {MAX_CHARS} characters")
        md = md[:MAX_CHARS] + "\n\n[... Content truncated due to length ...]"

    if ingestion_succeeded:
        logger.debug(f"Successfully processed repository {url}")

    return {
        "success": ingestion_succeeded,
        "url": url,
        "markdown": md,
    }
