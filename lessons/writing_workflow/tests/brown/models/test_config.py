"""Tests for brown.models.config (ModelConfig thinking parameters)."""

import pytest
from pydantic import ValidationError

from brown.models.config import ModelConfig


class TestModelConfigThinking:
    """Thinking-parameter behaviour: thinking_budget (2.5) vs thinking_level (3.x)."""

    def test_thinking_budget_only(self) -> None:
        config = ModelConfig(thinking_budget=1000)
        assert config.thinking_budget == 1000
        assert config.thinking_level is None

    def test_thinking_level_only(self) -> None:
        config = ModelConfig(thinking_level="low")
        assert config.thinking_level == "low"
        assert config.thinking_budget is None

    def test_neither_is_allowed(self) -> None:
        config = ModelConfig()
        assert config.thinking_budget is None
        assert config.thinking_level is None

    def test_budget_and_level_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ModelConfig(thinking_budget=1000, thinking_level="low")

    def test_invalid_thinking_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(thinking_level="ultra")

    def test_model_dump_drops_top_k_top_p(self) -> None:
        dumped = ModelConfig(thinking_level="medium").model_dump()
        assert "top_k" not in dumped
        assert "top_p" not in dumped

    def test_model_dump_excludes_unset_params(self) -> None:
        # Unset params are omitted (exclude_none) so the provider applies its own defaults,
        # e.g. Gemini 3.x uses temperature=1.0 when temperature is not sent.
        dumped = ModelConfig(thinking_level="low").model_dump()
        assert dumped.get("thinking_level") == "low"
        assert "thinking_budget" not in dumped
        assert "temperature" not in dumped

    def test_model_dump_includes_set_params(self) -> None:
        dumped = ModelConfig(temperature=0.0, thinking_budget=512).model_dump()
        assert dumped["temperature"] == 0.0
        assert dumped["thinking_budget"] == 512
        assert "thinking_level" not in dumped
