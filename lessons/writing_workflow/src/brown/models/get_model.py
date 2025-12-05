import json

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from brown.config import get_settings

from .config import (
    DEFAULT_MODEL_CONFIGS,
    ModelConfig,
    SupportedModels,
)
from .fake_model import FakeModel

MODEL_TO_REQUIRED_API_KEY = {
    SupportedModels.GOOGLE_GEMINI_30_PRO: "GOOGLE_API_KEY",
    SupportedModels.GOOGLE_GEMINI_25_PRO: "GOOGLE_API_KEY",
    SupportedModels.GOOGLE_GEMINI_25_FLASH: "GOOGLE_API_KEY",
    SupportedModels.GOOGLE_GEMINI_25_FLASH_LITE: "GOOGLE_API_KEY",
}


def get_model(model: SupportedModels, config: ModelConfig | None = None) -> BaseChatModel:
    if model == SupportedModels.FAKE_MODEL:
        if config and config.mocked_response is not None:
            if hasattr(config.mocked_response, "model_dump"):
                mocked_response_json = config.mocked_response.model_dump(mode="json")
            else:
                mocked_response_json = json.dumps(config.mocked_response)
            return FakeModel(responses=[mocked_response_json])
        else:
            return FakeModel(responses=[])

    config = config or DEFAULT_MODEL_CONFIGS.get(model) or ModelConfig()
    model_kwargs = {
        "model": model.value,
        **config.model_dump(),
    }

    required_api_key = MODEL_TO_REQUIRED_API_KEY.get(model)
    if required_api_key:
        settings = get_settings()
        if not getattr(settings, required_api_key):
            raise ValueError(f"Required environment variable `{required_api_key}` is not set")
        else:
            model_kwargs["api_key"] = getattr(settings, required_api_key)

    return init_chat_model(**model_kwargs)
