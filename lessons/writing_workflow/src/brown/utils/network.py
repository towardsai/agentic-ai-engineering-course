import httpx

INVALID_IMAGE_DOMAINS = ["github"]


async def is_image_url_valid(url: str, timeout: float = 5.0) -> bool:
    return is_image_domain_accepted(url) and await ping(url, timeout)


def is_image_domain_accepted(url: str) -> bool:
    return not any(domain in url for domain in INVALID_IMAGE_DOMAINS)


async def ping(url: str, timeout: float = 5.0) -> bool:
    """
    Check if a URL is valid and reachable (returns HTTP 200) without redirects.

    Args:
        url: The URL to check.
        timeout: Timeout in seconds for the request.

    Returns:
        True if the URL is reachable and returns HTTP 200 with no redirect,
        False otherwise.

    Example:
        >>> import asyncio
        >>> asyncio.run(is_url_valid("https://www.google.com"))
        True
    """

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            # Prefer HEAD to avoid downloading bodies when possible
            response = await client.head(url)

            # Some servers don't implement HEAD correctly or omit headers
            # Fallback to GET with an image-only Accept header
            if response.status_code == 405 or not response.headers.get("Content-Type"):
                response = await client.get(
                    url,
                    headers={
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                        "User-Agent": "brown-bot/1.0",
                    },
                )

            # Must be a successful response
            if response.status_code != 200:
                return False

            # Validate content type to catch HTML pages that "redirect" via meta/JS
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                return False

            return True
    except (httpx.RequestError, httpx.HTTPStatusError, httpx.TimeoutException, httpx.InvalidURL):
        return False
