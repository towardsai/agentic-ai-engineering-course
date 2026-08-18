"""Tests for provider-directed credential selection in lessons/utils/env.py."""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))

from utils.env import _apply_credential_selectors, _resolve_selector_env_vars, _setup_vertex_ai  # noqa: E402

NOTEBOOK_REQUIRED = ["GOOGLE_API_KEY", "PPLX_API_KEY", "FIRECRAWL_API_KEY"]
VERTEX_ON = {"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_CLOUD_PROJECT": "my-project"}


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


class ResolveFromColabSecretsTests(unittest.TestCase):
    """Selector and optional Vertex config should come from Colab Secrets, never a prompt."""

    def test_reads_selectors_and_optional_location_from_colab_secrets(self) -> None:
        secrets = {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "WEB_SEARCH_PROVIDER": "tavily",
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
        }
        colab_user_data = MagicMock()
        colab_user_data.get.side_effect = lambda name: secrets.get(name)
        with patch.dict("os.environ", {}, clear=True):
            _resolve_selector_env_vars(colab_user_data)
            self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "true")
            self.assertEqual(os.environ["WEB_SEARCH_PROVIDER"], "tavily")
            self.assertEqual(os.environ["GOOGLE_CLOUD_LOCATION"], "europe-west4")

    def test_optional_github_token_comes_from_colab_secrets(self) -> None:
        """The Colab setup cells used to load GITHUB_TOKEN by hand; env.load now does it."""
        colab_user_data = MagicMock()
        colab_user_data.get.side_effect = lambda name: {"GITHUB_TOKEN": "ghp_test"}.get(name)
        with patch.dict("os.environ", {}, clear=True):
            _resolve_selector_env_vars(colab_user_data)
            self.assertEqual(os.environ["GITHUB_TOKEN"], "ghp_test")

    def test_missing_optional_secret_is_not_fatal(self) -> None:
        colab_user_data = MagicMock()
        colab_user_data.get.side_effect = KeyError("no such secret")
        with patch.dict("os.environ", {}, clear=True):
            _resolve_selector_env_vars(colab_user_data)  # must not raise
            self.assertNotIn("GITHUB_TOKEN", os.environ)

    def test_colab_location_survives_vertex_setup(self) -> None:
        colab_user_data = MagicMock()
        colab_user_data.get.side_effect = lambda name: {"GOOGLE_CLOUD_LOCATION": "europe-west4"}.get(name)
        with patch.dict("os.environ", VERTEX_ON, clear=True), patch("google.auth.default") as auth_mock:
            auth_mock.return_value = (MagicMock(), "my-project")
            _resolve_selector_env_vars(colab_user_data)
            _setup_vertex_ai(is_colab=False)
            self.assertEqual(os.environ["GOOGLE_CLOUD_LOCATION"], "europe-west4")


class SetupVertexAiTests(unittest.TestCase):
    """Vertex AI authentication should be prepared automatically, not left to the student."""

    def test_no_op_when_vertex_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch("google.auth.default") as auth_mock:
            _setup_vertex_ai(is_colab=False)
        auth_mock.assert_not_called()
        self.assertNotIn("GOOGLE_CLOUD_LOCATION", os.environ)

    def test_local_mode_defaults_location_and_accepts_existing_adc(self) -> None:
        with patch.dict("os.environ", VERTEX_ON, clear=True), patch("google.auth.default") as auth_mock:
            auth_mock.return_value = (MagicMock(), "my-project")
            _setup_vertex_ai(is_colab=False)
            self.assertEqual(os.environ["GOOGLE_CLOUD_LOCATION"], "global")
        auth_mock.assert_called_once()

    def test_local_mode_keeps_explicit_location(self) -> None:
        env = {**VERTEX_ON, "GOOGLE_CLOUD_LOCATION": "europe-west1"}
        with patch.dict("os.environ", env, clear=True), patch("google.auth.default") as auth_mock:
            auth_mock.return_value = (MagicMock(), "my-project")
            _setup_vertex_ai(is_colab=False)
            self.assertEqual(os.environ["GOOGLE_CLOUD_LOCATION"], "europe-west1")

    def test_local_mode_without_adc_explains_the_gcloud_command(self) -> None:
        with patch.dict("os.environ", VERTEX_ON, clear=True), patch("google.auth.default") as auth_mock:
            auth_mock.side_effect = RuntimeError("no credentials")
            with self.assertRaises(RuntimeError) as context:
                _setup_vertex_ai(is_colab=False)
        self.assertIn("gcloud auth application-default login", str(context.exception))

    def test_colab_mode_authenticates_automatically(self) -> None:
        colab_auth = MagicMock()
        fake_colab = types.ModuleType("google.colab")
        fake_colab.auth = colab_auth
        with (
            patch.dict("os.environ", VERTEX_ON, clear=True),
            patch.dict("sys.modules", {"google.colab": fake_colab, "google.colab.auth": colab_auth}),
        ):
            _setup_vertex_ai(is_colab=True)
        colab_auth.authenticate_user.assert_called_once()

    def test_colab_mode_failure_is_actionable(self) -> None:
        colab_auth = MagicMock()
        colab_auth.authenticate_user.side_effect = RuntimeError("sign-in cancelled")
        fake_colab = types.ModuleType("google.colab")
        fake_colab.auth = colab_auth
        with (
            patch.dict("os.environ", VERTEX_ON, clear=True),
            patch.dict("sys.modules", {"google.colab": fake_colab, "google.colab.auth": colab_auth}),
        ):
            with self.assertRaises(RuntimeError) as context:
                _setup_vertex_ai(is_colab=True)
        self.assertIn("Colab authentication failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
