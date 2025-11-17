import re
from functools import cached_property

from loguru import logger
from pydantic import BaseModel

from brown.entities.mixins import ContextMixin
from brown.utils.a import asyncio_run, run_jobs
from brown.utils.network import is_image_url_valid


class Research(BaseModel, ContextMixin):
    content: str
    max_image_urls: int = 30

    @cached_property
    def image_urls(self) -> list[str]:
        # TODO: Add support for SVG images (now Gemini fails to process them).
        image_urls = re.findall(
            r"(?!data:image/)https?://[^\s]+\.(?:jpg|jpeg|png|bmp|webp)",
            self.content,
            re.IGNORECASE,
        )
        jobs = [is_image_url_valid(url) for url in image_urls]
        results = asyncio_run(run_jobs(jobs))

        urls = [url for url, valid in zip(image_urls, results) if valid]
        if len(urls) > self.max_image_urls:
            logger.warning(f"Found `{len(urls)} > {self.max_image_urls}` image URLs in research. Trimming to first {self.max_image_urls}.")
            urls = urls[: self.max_image_urls]

        return urls

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    {self.content}
</{self.xml_tag}>
"""

    def __str__(self) -> str:
        return f"Research(len_content={len(self.content)}, len_image_urls={len(self.image_urls)})"
