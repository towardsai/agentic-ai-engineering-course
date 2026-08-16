"""Tests for Gemini authentication selection (Gemini Developer API vs. Vertex AI)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.youtube_handler import transcribe_youtube
from src.config.settings import Settings
from src.utils.llm_utils import get_chat_model


def make_settings(**env_overrides) -> Settings:
    """Build Settings from explicit values only, ignoring .env and the process environment."""
    with patch.dict(os.environ, {}, clear=True):
        return Settings(_env_file=None, **env_overrides)


class GoogleClientKwargsTests(unittest.TestCase):
    def test_settings_construct_without_any_google_credentials(self) -> None:
        settings = make_settings()
        self.assertIsNone(settings.google_api_key)
        self.assertFalse(settings.google_genai_use_vertexai)

    def test_default_mode_requires_api_key_lazily(self) -> None:
        settings = make_settings()
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
            _ = settings.google_client_kwargs

    def test_default_mode_returns_unwrapped_api_key(self) -> None:
        settings = make_settings(GOOGLE_API_KEY="test-key")
        self.assertEqual(settings.google_client_kwargs, {"api_key": "test-key"})

    def test_vertex_mode_does_not_require_api_key(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project")
        self.assertEqual(
            settings.google_client_kwargs,
            {"vertexai": True, "project": "my-project", "location": "global"},
        )

    def test_vertex_mode_requires_project(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True)
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_CLOUD_PROJECT"):
            _ = settings.google_client_kwargs

    def test_vertex_mode_honors_custom_location(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project", GOOGLE_CLOUD_LOCATION="europe-west1")
        self.assertEqual(settings.google_client_kwargs["location"], "europe-west1")


class GetChatModelKwargsTests(unittest.TestCase):
    def test_gemini_default_mode_passes_api_key(self) -> None:
        settings = make_settings(GOOGLE_API_KEY="test-key")
        with patch("src.utils.llm_utils.settings", settings), patch("src.utils.llm_utils.init_chat_model") as init_mock:
            get_chat_model("gemini-2.5-pro")
        _, kwargs = init_mock.call_args
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertNotIn("vertexai", kwargs)

    def test_gemini_vertex_mode_passes_vertex_kwargs_without_api_key(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project")
        with patch("src.utils.llm_utils.settings", settings), patch("src.utils.llm_utils.init_chat_model") as init_mock:
            get_chat_model("gemini-2.5-pro")
        _, kwargs = init_mock.call_args
        self.assertTrue(kwargs["vertexai"])
        self.assertEqual(kwargs["project"], "my-project")
        self.assertEqual(kwargs["location"], "global")
        self.assertNotIn("api_key", kwargs)

    def test_perplexity_path_is_unaffected_by_vertex_mode(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project", PPLX_API_KEY="pplx-key")
        with patch("src.utils.llm_utils.settings", settings), patch("src.utils.llm_utils.init_chat_model") as init_mock:
            get_chat_model("perplexity")
        _, kwargs = init_mock.call_args
        self.assertEqual(kwargs["api_key"], "pplx-key")
        self.assertNotIn("vertexai", kwargs)

    def test_gemini_default_mode_without_key_raises(self) -> None:
        settings = make_settings()
        with patch("src.utils.llm_utils.settings", settings):
            with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
                get_chat_model("gemini-2.5-pro")


class RawClientKwargsTests(unittest.IsolatedAsyncioTestCase):
    """Verify the raw genai.Client call site receives the selected auth kwargs."""

    def _tracked_client(self, text: str = "transcript") -> MagicMock:
        tracked = MagicMock()
        tracked.aio.models.generate_content = AsyncMock(return_value=MagicMock(text=text))
        return tracked

    async def test_transcribe_youtube_builds_client_with_vertex_kwargs(self) -> None:
        settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project")
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "transcript.md"
            with (
                patch("src.app.youtube_handler.settings", settings),
                patch("src.app.youtube_handler.genai.Client") as client_mock,
                patch("src.app.youtube_handler.track_genai_client", return_value=self._tracked_client()),
            ):
                await transcribe_youtube("https://www.youtube.com/watch?v=abc", output_path)
            client_mock.assert_called_once_with(vertexai=True, project="my-project", location="global")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "transcript")

    async def test_transcribe_youtube_builds_client_with_api_key(self) -> None:
        settings = make_settings(GOOGLE_API_KEY="test-key")
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "transcript.md"
            with (
                patch("src.app.youtube_handler.settings", settings),
                patch("src.app.youtube_handler.genai.Client") as client_mock,
                patch("src.app.youtube_handler.track_genai_client", return_value=self._tracked_client()),
            ):
                await transcribe_youtube("https://www.youtube.com/watch?v=abc", output_path)
            client_mock.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
