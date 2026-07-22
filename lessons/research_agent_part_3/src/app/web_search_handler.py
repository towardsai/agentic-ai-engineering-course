"""Provider-neutral web-search dispatch."""

from typing import Dict, Tuple

from ..config.settings import settings
from .perplexity_handler import run_perplexity_search
from .tavily_handler import run_tavily_search

SUPPORTED_WEB_SEARCH_PROVIDERS = ("tavily", "perplexity")


def resolve_web_search_provider(provider: str | None = None) -> str:
    """Return and validate the requested provider, using configuration by default."""
    selected_provider = (provider or settings.web_search_provider).strip().lower()
    if selected_provider not in SUPPORTED_WEB_SEARCH_PROVIDERS:
        supported = ", ".join(SUPPORTED_WEB_SEARCH_PROVIDERS)
        raise ValueError(f"Unsupported web search provider: '{selected_provider}'. Supported providers: {supported}.")
    return selected_provider


async def run_web_search(query: str, provider: str | None = None) -> Tuple[str, Dict[int, str], Dict[int, str]]:
    """Run one web search with Tavily by default or Perplexity when selected."""
    selected_provider = resolve_web_search_provider(provider)
    if selected_provider == "tavily":
        return await run_tavily_search(query)
    return await run_perplexity_search(query)
