"""Tests for provider-directed credential selection in lessons/utils/env.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))

from utils.env import _apply_credential_selectors  # noqa: E402

NOTEBOOK_REQUIRED = ["GOOGLE_API_KEY", "PPLX_API_KEY", "FIRECRAWL_API_KEY"]


class ApplyCredentialSelectorsTests(unittest.TestCase):
    def test_defaults_keep_required_vars_unchanged(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_apply_credential_selectors(NOTEBOOK_REQUIRED), NOTEBOOK_REQUIRED)

    def test_vertex_mode_swaps_google_api_key_for_project(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_GENAI_USE_VERTEXAI": "true"}, clear=True):
            required = _apply_credential_selectors(NOTEBOOK_REQUIRED)
        self.assertNotIn("GOOGLE_API_KEY", required)
        self.assertIn("GOOGLE_CLOUD_PROJECT", required)
        self.assertIn("PPLX_API_KEY", required)

    def test_tavily_mode_swaps_pplx_key_for_tavily_key(self) -> None:
        with patch.dict("os.environ", {"WEB_SEARCH_PROVIDER": "tavily"}, clear=True):
            required = _apply_credential_selectors(NOTEBOOK_REQUIRED)
        self.assertNotIn("PPLX_API_KEY", required)
        self.assertIn("TAVILY_API_KEY", required)
        self.assertIn("GOOGLE_API_KEY", required)

    def test_perplexity_mode_still_requires_pplx_key(self) -> None:
        with patch.dict("os.environ", {"WEB_SEARCH_PROVIDER": "perplexity"}, clear=True):
            required = _apply_credential_selectors(NOTEBOOK_REQUIRED)
        self.assertIn("PPLX_API_KEY", required)
        self.assertNotIn("TAVILY_API_KEY", required)

    def test_both_selectors_combine(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_GENAI_USE_VERTEXAI": "1", "WEB_SEARCH_PROVIDER": "tavily"}, clear=True):
            required = _apply_credential_selectors(NOTEBOOK_REQUIRED)
        self.assertEqual(set(required), {"GOOGLE_CLOUD_PROJECT", "TAVILY_API_KEY", "FIRECRAWL_API_KEY"})

    def test_vertex_disabled_values_do_not_trigger_swap(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_GENAI_USE_VERTEXAI": "false"}, clear=True):
            required = _apply_credential_selectors(NOTEBOOK_REQUIRED)
        self.assertIn("GOOGLE_API_KEY", required)
        self.assertNotIn("GOOGLE_CLOUD_PROJECT", required)


if __name__ == "__main__":
    unittest.main()
