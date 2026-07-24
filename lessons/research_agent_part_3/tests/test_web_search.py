"""Tests for the deployable server's provider-neutral web search."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.tavily_handler import parse_tavily_response
from src.app.web_search_handler import resolve_web_search_provider, run_web_search
from src.tools.run_web_research_tool import run_web_research_tool


class TavilyResponseTests(unittest.TestCase):
    def test_normalizes_tavily_sources(self) -> None:
        response = {
            "results": [
                {"title": "Source", "url": "https://example.com", "content": "Evidence"},
            ]
        }

        full_answer, answers, citations = parse_tavily_response(response)

        self.assertEqual(answers, {1: "Source\n\nEvidence"})
        self.assertEqual(citations, {1: "https://example.com"})
        self.assertIn("https://example.com", full_answer)


class ProviderDispatchTests(unittest.IsolatedAsyncioTestCase):
    def test_default_provider_is_perplexity(self) -> None:
        with patch("src.app.web_search_handler.settings.web_search_provider", "perplexity"):
            self.assertEqual(resolve_web_search_provider(), "perplexity")

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported web search provider"):
            resolve_web_search_provider("unknown")

    async def test_dispatches_to_perplexity_by_default(self) -> None:
        expected = ("answer", {1: "content"}, {1: "https://example.com"})
        with (
            patch("src.app.web_search_handler.settings.web_search_provider", "perplexity"),
            patch(
                "src.app.web_search_handler.run_perplexity_search",
                new=AsyncMock(return_value=expected),
            ) as search,
        ):
            result = await run_web_search("query")

        self.assertEqual(result, expected)
        search.assert_awaited_once_with("query")


class WebResearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_persists_successful_results_when_one_query_fails(self) -> None:
        article = SimpleNamespace(perplexity_results=None)
        session = AsyncMock()
        session.get.return_value = article
        session_context = AsyncMock()
        session_context.__aenter__.return_value = session
        session_factory = MagicMock(return_value=session_context)
        successful_result = (
            "answer",
            {1: "Source title\n\nSource content"},
            {1: "https://example.com/source"},
        )

        with (
            patch(
                "src.tools.run_web_research_tool.get_async_session_factory",
                new=AsyncMock(return_value=session_factory),
            ),
            patch(
                "src.tools.run_web_research_tool.run_web_search",
                new=AsyncMock(side_effect=[successful_result, RuntimeError("rate limited")]),
            ),
        ):
            result = await run_web_research_tool(
                "d2719c4d-9a31-4e39-bb41-1baf9b9e97d7",
                ["successful query", "failed query"],
                provider="tavily",
            )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["queries_succeeded"], ["successful query"])
        self.assertEqual(result["queries_failed"][0]["query"], "failed query")
        self.assertEqual(result["sources_added"], 1)
        self.assertIn("https://example.com/source", article.perplexity_results)
        session.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
