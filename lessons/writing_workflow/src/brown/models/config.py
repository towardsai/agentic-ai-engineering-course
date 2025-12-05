from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SupportedModels(StrEnum):
    GOOGLE_GEMINI_30_PRO = "google_genai:gemini-3-pro-preview"
    GOOGLE_GEMINI_25_PRO = "google_genai:gemini-2.5-pro"
    GOOGLE_GEMINI_25_FLASH = "google_genai:gemini-2.5-flash"
    GOOGLE_GEMINI_25_FLASH_LITE = "google_genai:gemini-2.5-flash-lite"
    FAKE_MODEL = "fake"


class ModelConfig(BaseModel):
    temperature: float = 0.7
    top_k: int | None = None
    n: int = 1
    response_modalities: list[str] | None = None
    include_thoughts: bool = False
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        description="If reasoning is available, the maximum number of tokens the model can use for thinking.",
    )
    max_output_tokens: int | None = None
    max_retries: int = 6

    mocked_response: Any | None = None

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        return super().model_dump(
            include={
                "temperature",
                "top_k",
                "n",
                "response_modalities",
                "thinking_budget",
                "max_output_tokens",
                "max_retries",
            },
            mode=kwargs.pop("mode", "json"),
            *args,
            **kwargs,
        )


DEFAULT_MODEL_CONFIGS = {
    "google_genai:gemini-3-pro-preview": ModelConfig(
        temperature=0.7,
        include_thoughts=False,
        thinking_budget=1000,
        max_retries=3,
    ),
    "google_genai:gemini-2.5-pro": ModelConfig(
        temperature=0.7,
        include_thoughts=False,
        thinking_budget=1000,
        max_retries=3,
    ),
    "google_genai:gemini-2.5-flash": ModelConfig(
        temperature=1,
        thinking_budget=1000,
        include_thoughts=False,
        max_retries=3,
    ),
}
