"""Opik configuration and tracking utilities for MCP Server."""

import opik
from google import genai
from opik.integrations.genai import track_genai
from opik.integrations.langchain import OpikTracer
from opik.integrations.openai import track_openai
from opik.integrations.openai.opik_tracker import OpenAIClient

from ..config.settings import settings


def configure_opik():
    """Configure Opik monitoring if all required settings are provided.

    Returns:
        bool: True if Opik was configured, False otherwise
    """
    if settings.opik_api_key and settings.opik_workspace and settings.opik_project_name:
        opik.configure(
            api_key=settings.opik_api_key.get_secret_value(),
            workspace=settings.opik_workspace,
            use_local=False,
            force=True,
        )
        return True
    return False


def track_genai_client(client: genai.Client) -> genai.Client:
    """Track a Gemini client with Opik if configured.

    Args:
        client: The Gemini client to track

    Returns:
        The tracked client if Opik is configured, otherwise the original client
    """
    # Apply Opik tracking if all required settings are configured
    if settings.opik_api_key and settings.opik_workspace and settings.opik_project_name:
        return track_genai(client, project_name=settings.opik_project_name)
    else:
        return client


def track_openai_client(client: OpenAIClient) -> OpenAIClient:
    """Track an OpenAI client with Opik if configured.

    Args:
        client: The OpenAI client to track

    Returns:
        The tracked client if Opik is configured, otherwise the original client
    """
    # Apply Opik tracking if all required settings are configured
    if settings.opik_api_key and settings.opik_workspace and settings.opik_project_name:
        return track_openai(client)
    else:
        return client


class TrackedLangChainModel:
    """Wrapper for LangChain models to enable Opik tracking using OpikTracer.

    This class wraps any LangChain BaseChatModel and automatically injects
    the OpikTracer into the config for invoke/ainvoke calls, following the
    pattern from the Opik documentation.
    """

    def __init__(self, model, model_name: str, project_name: str):
        self._model = model
        self._model_name = model_name
        self._project_name = project_name

    def __getattr__(self, name):
        """Delegate all attribute access to the wrapped model."""
        return getattr(self._model, name)

    def _inject_tracer(self, kwargs):
        """Inject OpikTracer into the config callbacks if not already present."""
        config = kwargs.get("config", {})
        callbacks = config.get("callbacks", [])

        opik_tracer: OpikTracer = OpikTracer(project_name=self._project_name)

        # Check if our tracer is already in the callbacks
        if opik_tracer not in callbacks:
            config["callbacks"] = callbacks + [opik_tracer]
            kwargs["config"] = config

        return kwargs

    async def ainvoke(self, *args, **kwargs):
        """Track async invocations with OpikTracer."""
        kwargs = self._inject_tracer(kwargs)
        return await self._model.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """Track sync invocations with OpikTracer."""
        kwargs = self._inject_tracer(kwargs)
        return self._model.invoke(*args, **kwargs)


def track_langchain_model(model, model_name: str) -> TrackedLangChainModel:
    """Track a LangChain model with Opik."""
    # Return a wrapped model that uses OpikTracer for tracking
    return TrackedLangChainModel(model, model_name, settings.opik_project_name)


def is_opik_enabled() -> bool:
    """Check if Opik monitoring is enabled.

    Returns:
        bool: True if Opik is configured and enabled
    """
    return (
        settings.opik_api_key is not None
        and settings.opik_workspace is not None
        and settings.opik_project_name is not None
    )
