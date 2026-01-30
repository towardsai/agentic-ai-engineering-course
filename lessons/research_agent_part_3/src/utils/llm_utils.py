"""LLM utilities for creating and managing chat models."""

import logging
from typing import Optional, Type

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

from ..config.settings import settings
from .opik_utils import TrackedLangChainModel, is_opik_enabled, track_langchain_model

logger = logging.getLogger(__name__)


def get_chat_model(model_id: str, schema: Optional[Type[BaseModel]] = None) -> BaseChatModel | TrackedLangChainModel:
    """
    Initializes and returns a chat model from the centralized configuration.

    Args:
        model_id: The model identifier to use from llm_configs

    Returns:
        An instance of a LangChain chat model.
    """
    selected_model_config = settings.llm_configs[model_id]
    model_identifier = selected_model_config["identifier"]
    model_params = selected_model_config.get("params", {})
    api_key_env_var = selected_model_config.get("api_key_env_var")

    init_kwargs = model_params.copy()

    if api_key_env_var:
        # Get the appropriate API key based on the environment variable name
        api_key = None
        if api_key_env_var == "GOOGLE_API_KEY" and settings.google_api_key:
            api_key = settings.google_api_key.get_secret_value()
        elif api_key_env_var == "OPENAI_API_KEY" and settings.openai_api_key:
            api_key = settings.openai_api_key.get_secret_value()
        elif api_key_env_var == "PPLX_API_KEY" and settings.perplexity_api_key:
            api_key = settings.perplexity_api_key.get_secret_value()

        if not api_key:
            msg = f"{api_key_env_var} environment variable not set."
            logger.error(msg)
            raise RuntimeError(msg)
        init_kwargs["api_key"] = api_key

    model = init_chat_model(model_identifier, **init_kwargs)

    if schema is not None:
        model = model.with_structured_output(schema)

    # Apply Opik tracking if configured
    if is_opik_enabled():
        return track_langchain_model(model, model_id)

    return model
