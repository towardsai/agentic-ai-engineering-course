"""Pydantic models for query generation and research operations."""

from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Literal


class QueryAndReason(BaseModel):
    """A single web-search query and the reason for it."""

    question: str = Field(description="The web-search question to research.")
    reason: str = Field(description="The reason why this question is important for the research.")


class GeneratedQueries(BaseModel):
    """A list of generated web-search queries and their reasons."""

    queries: List[QueryAndReason] = Field(description="A list of web-search queries and their reasons.")


class SourceSelection(BaseModel):
    """Structured response expected from the LLM."""

    selection_type: Literal["none", "all", "specific"] = Field(
        description=(
            "Type of selection made: 'none' for no sources, 'all' for all sources, "
            "or 'specific' for specific source IDs"
        )
    )
    source_ids: List[int] = Field(
        description="List of selected source IDs. Empty for 'none', all IDs for 'all', or specific IDs for 'specific'"
    )


class TopSourceSelection(BaseModel):
    """Expected structure from the LLM when choosing the top sources."""

    selected_urls: List[str] = Field(description="List of URLs to scrape in full, ordered by priority.")
    reasoning: str = Field(description="Short explanation summarising why these URLs were chosen.")
