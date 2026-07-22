"""Tests for provider-neutral web search."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.app.tavily_handler import parse_tavily_response
from src.app.web_search_handler import resolve_web_search_provider, run_web_search
from src.tools.run_web_research_tool import run_web_research_tool


class TavilyResponseTests(unittest.TestCase):
    def test_normalizes_ranked_sources_and_skips_incomplete_results(self) -> None:
        response = {
            "results": [
                {"title": "Official docs", "url": "https://example.com/docs", "content": "Useful content."},
                {"title": "Missing content", "url": "https://example.com/empty", "content": ""},
                {"title": "Second source", "url": "https://example.com/two", "content": "More evidence."},
            ]
        }

        full_answer, answers, citations = parse_tavily_response(response)

        self.assertEqual(citations, {1: "https://example.com/docs", 2: "https://example.com/two"})
        self.assertEqual(answers[1], "Official docs\n\nUseful content.")
        self.assertIn("### [2]: https://example.com/two", full_answer)


class ProviderDispatchTests(unittest.IsolatedAsyncioTestCase):
    def test_default_provider_is_tavily(self) -> None:
        with patch("src.app.web_search_handler.settings.web_search_provider", "tavily"):
            self.assertEqual(resolve_web_search_provider(), "tavily")

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported web search provider"):
            resolve_web_search_provider("unknown")

    async def test_dispatches_to_perplexity_when_overridden(self) -> None:
        expected = ("answer", {1: "content"}, {1: "https://example.com"})
        with patch(
            "src.app.web_search_handler.run_perplexity_search",
            new=AsyncMock(return_value=expected),
        ) as search:
            result = await run_web_search("query", provider="perplexity")

        self.assertEqual(result, expected)
        search.assert_awaited_once_with("query")


class WebResearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_successful_results_when_one_query_fails(self) -> None:
        successful_result = (
            "answer",
            {1: "Source title\n\nSource content"},
            {1: "https://example.com/source"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            research_path = Path(temp_dir)
            (research_path / ".nova").mkdir()
            with patch(
                "src.tools.run_web_research_tool.run_web_search",
                new=AsyncMock(side_effect=[successful_result, RuntimeError("rate limited")]),
            ):
                result = await run_web_research_tool(
                    str(research_path),
                    ["successful query", "failed query"],
                    provider="tavily",
                )

            self.assertEqual(result["status"], "partial_success")
            self.assertEqual(result["queries_succeeded"], ["successful query"])
            self.assertEqual(result["queries_failed"][0]["query"], "failed query")
            self.assertEqual(result["sources_added"], 1)
            results_text = (research_path / ".nova" / "perplexity_results.md").read_text()
            self.assertIn("https://example.com/source", results_text)


if __name__ == "__main__":
    unittest.main()
