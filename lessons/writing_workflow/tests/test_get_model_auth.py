"""Tests for Gemini authentication selection in get_model (Developer API vs. Vertex AI)."""

from unittest.mock import patch

import pytest

from brown.config import Settings
from brown.models.config import SupportedModels
from brown.models.get_model import get_model


def make_settings(**overrides) -> Settings:
    with patch.dict("os.environ", {"CONFIG_FILE": "configs/debug.yaml"}, clear=True):
        return Settings(_env_file=None, **overrides)


def test_default_mode_passes_unwrapped_api_key():
    settings = make_settings(GOOGLE_API_KEY="test-key")
    with (
        patch("brown.models.get_model.get_settings", return_value=settings),
        patch("brown.models.get_model.init_chat_model") as init_mock,
    ):
        get_model(SupportedModels.GOOGLE_GEMINI_37_FLASH)
    kwargs = init_mock.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert "vertexai" not in kwargs


def test_vertex_mode_passes_vertex_kwargs_without_api_key():
    settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT="my-project")
    with (
        patch("brown.models.get_model.get_settings", return_value=settings),
        patch("brown.models.get_model.init_chat_model") as init_mock,
    ):
        get_model(SupportedModels.GOOGLE_GEMINI_37_FLASH)
    kwargs = init_mock.call_args.kwargs
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "my-project"
    assert kwargs["location"] == "global"
    assert "api_key" not in kwargs


def test_default_mode_without_api_key_raises():
    settings = make_settings()
    with patch("brown.models.get_model.get_settings", return_value=settings):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            get_model(SupportedModels.GOOGLE_GEMINI_37_FLASH)


def test_vertex_mode_without_project_raises():
    settings = make_settings(GOOGLE_GENAI_USE_VERTEXAI=True)
    with patch("brown.models.get_model.get_settings", return_value=settings):
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            get_model(SupportedModels.GOOGLE_GEMINI_37_FLASH)


def test_fake_model_needs_no_credentials():
    get_model(SupportedModels.FAKE_MODEL)
