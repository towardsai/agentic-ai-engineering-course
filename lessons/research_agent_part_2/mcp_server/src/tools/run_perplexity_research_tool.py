"""Backward-compatible Perplexity research tool."""

from typing import Any, Dict, List

from .run_web_research_tool import run_web_research_tool


async def run_perplexity_research_tool(research_directory: str, queries: List[str]) -> Dict[str, Any]:
    """Run research explicitly with Perplexity, preserving the original tool API."""
    return await run_web_research_tool(research_directory, queries, provider="perplexity")
