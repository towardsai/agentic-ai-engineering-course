"""Web scraping utilities."""

import asyncio
import logging
import re
from typing import List
from urllib.parse import urlparse

from firecrawl import AsyncFirecrawl
from langchain.chat_models.base import BaseChatModel

from ..config.prompts import PROMPT_CLEAN_MARKDOWN
from ..config.settings import settings
from ..utils.llm_utils import get_chat_model

logger = logging.getLogger(__name__)

# Cache settings for faster scraping
# maxAge values in milliseconds:
# 5 minutes: 300000, 1 hour: 3600000, 1 day: 86400000, 1 week: 604800000
MAX_AGE_ONE_WEEK = 604800000  # 1 week in milliseconds for 500% faster scraping


def slugify(text: str, max_length: int = 60) -> str:
    """Convert text to a filesystem-friendly slug."""
    text = text.lower()
    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_length] or "untitled"


def build_filename(title: str, url: str, existing_names: set) -> str:
    """Generate a unique filename for a scraped source."""
    base_name = slugify(title) if title and title.lower() != "n/a" else slugify(urlparse(url).netloc)
    candidate = base_name
    counter = 1
    while candidate in existing_names:
        candidate = f"{base_name}-{counter}"
        counter += 1
    existing_names.add(candidate)
    return f"{candidate}.md"


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


def convert_markdown_images_to_urls(text: str) -> str:
    """Convert markdown image and link syntax to just URLs for image content."""
    # Convert markdown images ![alt](url) to just url
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", text)

    # Convert markdown links [text](url) to just url when url appears to be an image
    # Common image extensions
    image_extensions = r"\.(jpg|jpeg|png|gif|bmp|svg|webp|ico|tiff|tif)(\?[^)]*)?$"
    text = re.sub(rf"\[([^\]]*)\]\(([^)]+{image_extensions})\)", r"\2", text, flags=re.IGNORECASE)

    return text


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
