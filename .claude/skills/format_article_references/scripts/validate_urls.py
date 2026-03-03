#!/usr/bin/env python3
"""Validate URLs and extract HTML metadata for APA reference formatting.

Usage:
    uv run python scripts/validate_urls.py <url1> <url2> ...

For each URL, performs a streaming GET (reads only the <head> section,
up to 32 KB) to check validity and extract metadata:
  - title (from <title> or og:title)
  - author (from meta author or article:author)
  - published_date (from article:published_time or meta date)
  - site_name (from og:site_name)

Outputs a JSON array with validation + metadata results.
"""

import asyncio
import json
import sys
from html.parser import HTMLParser

import httpx


class HeadMetadataParser(HTMLParser):
    """Parse only the <head> section of an HTML document for metadata."""

    def __init__(self):
        super().__init__()
        self.metadata: dict[str, str] = {}
        self._in_title = False
        self._title_text = ""
        self._in_head = False
        self._done = False

    def handle_starttag(self, tag, attrs):
        if self._done:
            return
        tag = tag.lower()
        if tag == "head":
            self._in_head = True
        elif tag == "body":
            self._done = True
        elif tag == "title" and self._in_head:
            self._in_title = True
        elif tag == "meta" and self._in_head:
            attrs_dict = {k.lower(): v for k, v in attrs if v is not None}
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            content = attrs_dict.get("content", "")
            if not content:
                return
            # Title
            if prop == "og:title" and "og_title" not in self.metadata:
                self.metadata["og_title"] = content
            # Site name
            elif prop == "og:site_name" and "site_name" not in self.metadata:
                self.metadata["site_name"] = content
            # Published date
            elif prop in ("article:published_time", "article:modified_time"):
                if "published_date" not in self.metadata:
                    self.metadata["published_date"] = content
            elif name == "date" and "published_date" not in self.metadata:
                self.metadata["published_date"] = content
            elif name == "publish_date" and "published_date" not in self.metadata:
                self.metadata["published_date"] = content
            # Author
            elif prop == "article:author" and "author" not in self.metadata:
                self.metadata["author"] = content
            elif name == "author" and "author" not in self.metadata:
                self.metadata["author"] = content
            elif name == "citation_author" and "author" not in self.metadata:
                self.metadata["author"] = content

    def handle_data(self, data):
        if self._in_title and not self._done:
            self._title_text += data

    def handle_endtag(self, tag):
        if self._done:
            return
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            text = self._title_text.strip()
            if text and "title" not in self.metadata:
                self.metadata["title"] = text
        elif tag == "head":
            self._in_head = False
            self._done = True

    def error(self, message):
        pass  # Silently ignore malformed HTML


def extract_metadata(html_text: str) -> dict[str, str]:
    """Extract metadata from an HTML string (only the <head> portion)."""
    parser = HeadMetadataParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass
    meta = parser.metadata
    # Prefer og:title over <title> (og:title is usually cleaner)
    result = {}
    result["title"] = meta.get("og_title") or meta.get("title") or None
    result["author"] = meta.get("author") or None
    result["published_date"] = meta.get("published_date") or None
    result["site_name"] = meta.get("site_name") or None
    return result


MAX_HEAD_BYTES = 32768  # 32 KB — enough for any <head> section


async def check_url(client: httpx.AsyncClient, url: str) -> dict:
    """Validate a URL and extract its <head> metadata via streaming GET."""
    try:
        async with client.stream(
            "GET", url, follow_redirects=True, timeout=15.0
        ) as response:
            status = response.status_code
            valid = status < 400

            # Read only the head portion for metadata extraction
            content = b""
            if valid:
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    content += chunk
                    if len(content) >= MAX_HEAD_BYTES:
                        break

            html_text = content.decode("utf-8", errors="ignore")
            metadata = extract_metadata(html_text) if valid else {}

            return {
                "url": url,
                "status_code": status,
                "valid": valid,
                "final_url": str(response.url),
                "metadata": metadata,
            }
    except httpx.TimeoutException:
        return {
            "url": url,
            "status_code": None,
            "valid": False,
            "error": "timeout",
            "metadata": {},
        }
    except httpx.ConnectError:
        return {
            "url": url,
            "status_code": None,
            "valid": False,
            "error": "connection_error",
            "metadata": {},
        }
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "valid": False,
            "error": str(e),
            "metadata": {},
        }


async def main(urls: list[str]) -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [check_url(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: validate_urls.py <url1> <url2> ...",
            file=sys.stderr,
        )
        sys.exit(1)

    urls = sys.argv[1:]
    results = asyncio.run(main(urls))
    print(json.dumps(results, indent=2))
