from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class SupportedModels(StrEnum):
    GOOGLE_GEMINI_25_PRO = "google_genai:gemini-2.5-pro"
    GOOGLE_GEMINI_35_FLASH = "google_genai:gemini-3.5-flash"
    GOOGLE_GEMINI_31_FLASH_LITE = "google_genai:gemini-3.1-flash-lite"
    GOOGLE_GEMINI_25_FLASH = "google_genai:gemini-2.5-flash"
    GOOGLE_GEMINI_25_FLASH_LITE = "google_genai:gemini-2.5-flash-lite"
    FAKE_MODEL = "fake"


class ModelConfig(BaseModel):
    temperature: float | None = None
    n: int = 1
    response_modalities: list[str] | None = None
    include_thoughts: bool = False
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        description="If reasoning is available, the maximum number of tokens the model can use for thinking. "
        "Used by Gemini 2.5 models. Mutually exclusive with `thinking_level`.",
    )
    thinking_level: ThinkingLevel | None = Field(
        default=None,
        description="Reasoning depth for Gemini 3.x models (one of minimal/low/medium/high). Mutually exclusive with `thinking_budget`.",
    )
    max_output_tokens: int | None = None
    max_retries: int = 6

    mocked_response: Any | None = None

    @model_validator(mode="after")
    def _check_thinking_exclusive(self) -> "ModelConfig":
        if self.thinking_budget is not None and self.thinking_level is not None:
            raise ValueError("`thinking_budget` and `thinking_level` are mutually exclusive; set only one.")
        return self

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        # exclude_none so unset params (e.g. temperature, thinking_*) are NOT forwarded to the
        # provider. This lets Gemini 3.x apply its recommended defaults (e.g. temperature=1.0).
        return super().model_dump(
            include={
                "temperature",
                "n",
                "response_modalities",
                "thinking_budget",
                "thinking_level",
                "max_output_tokens",
                "max_retries",
            },
            mode=kwargs.pop("mode", "json"),
            exclude_none=kwargs.pop("exclude_none", True),
            *args,
            **kwargs,
        )


DEFAULT_MODEL_CONFIGS = {
    "google_genai:gemini-2.5-pro": ModelConfig(
        temperature=0.7,
        include_thoughts=False,
        thinking_budget=1000,
        max_retries=3,
    ),
    "google_genai:gemini-3.5-flash": ModelConfig(
        temperature=1,
        include_thoughts=False,
        thinking_level="low",
        max_retries=3,
    ),
    "google_genai:gemini-3.1-flash-lite": ModelConfig(
        temperature=1,
        include_thoughts=False,
        thinking_level="low",
        max_retries=3,
    ),
    "google_genai:gemini-2.5-flash": ModelConfig(
        temperature=1,
        include_thoughts=False,
        thinking_budget=1000,
        max_retries=3,
    ),
}
