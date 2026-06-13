"""GitHub-specific utilities."""

import asyncio
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from gitingest import ingest_async

logger = logging.getLogger(__name__)

INGEST_TIMEOUT_SECONDS = 300

# GitIngest shells out to git. MCP tool calls do not have an interactive stdin,
# so credential prompts would otherwise block the whole parallel ingestion step.
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ["GCM_INTERACTIVE"] = "never"


async def process_github_url(url: str, dest_folder: Path, token: str | None) -> bool:
    """Fetch a GitHub repository (or file) with gitingest and write a Markdown report."""
    ingestion_succeeded = False
    try:
        summary, tree, content = await asyncio.wait_for(
            ingest_async(url, exclude_patterns="*.lock", token=token), timeout=INGEST_TIMEOUT_SECONDS
        )
        ingestion_succeeded = True
        md = f"# Repository analysis for {url}\n\n## Summary\n{summary}\n\n## File tree\n```{tree}\n```\n\n## Extracted content\n{content}"
    except TimeoutError:
        md = f"# Error processing {url}\n\nGitHub ingestion timed out after {INGEST_TIMEOUT_SECONDS} seconds."
        logger.error("GitHub ingestion timed out after %s seconds for %s", INGEST_TIMEOUT_SECONDS, url)
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

    # Construct a filename that reflects the repo (owner_repo.md) or fallback to sanitized URL
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) >= 2:
        dest_name = f"{parts[0]}_{parts[1]}.md"
    else:
        dest_name = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".md"

    dest_path = dest_folder / dest_name
    dest_path.write_text(md, encoding="utf-8")
    if ingestion_succeeded:
        logger.debug(f"Successfully processed repository {url} and wrote  {dest_path}")

    return ingestion_succeeded
