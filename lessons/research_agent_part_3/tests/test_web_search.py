"""Tests for the deployable server's provider-neutral web search."""

import unittest
from unittest.mock import AsyncMock, patch

from src.app.tavily_handler import parse_tavily_response
from src.app.web_search_handler import resolve_web_search_provider, run_web_search


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
    def test_default_provider_is_tavily(self) -> None:
        with patch("src.app.web_search_handler.settings.web_search_provider", "tavily"):
            self.assertEqual(resolve_web_search_provider(), "tavily")

    async def test_dispatches_to_tavily_by_default(self) -> None:
        expected = ("answer", {1: "content"}, {1: "https://example.com"})
        with (
            patch("src.app.web_search_handler.settings.web_search_provider", "tavily"),
            patch(
                "src.app.web_search_handler.run_tavily_search",
                new=AsyncMock(return_value=expected),
            ) as search,
        ):
            result = await run_web_search("query")

        self.assertEqual(result, expected)
        search.assert_awaited_once_with("query")


if __name__ == "__main__":
    unittest.main()
