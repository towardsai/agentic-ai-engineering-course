"""Client configuration settings."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    """Application settings for the MCP Client."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Server settings and paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent, description="The root directory of the mcp_client project"
    )
    mcp_config_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "mcp_servers_config.json",
        description="Path to the MCP servers configuration file",
    )
    log_level: int = Field(default=logging.INFO, alias="LOG_LEVEL", description="The log level")
    log_level_dependencies: int = Field(
        default=logging.WARNING, alias="LOG_LEVEL_DEPENDENCIES", description="The log level for dependencies"
    )

    # LLM Configuration
    orchestrator_key: str = Field(default="gemini-3.5-flash", description="Default orchestrator model key")
    model_id: str = Field(default="gemini-3.5-flash", description="Default model ID for LLM operations")
    thinking_budget: int | None = Field(
        default=None, description="Thinking token budget for Gemini 2.5 models. Mutually exclusive with thinking_level."
    )
    thinking_level: ThinkingLevel | None = Field(
        default="low", description="Reasoning depth for Gemini 3.x models. Mutually exclusive with thinking_budget."
    )

    # Agent configuration
    recursion_limit: int = Field(default=100, description="The recursion limit for the agent")

    # Gemini authentication. The Gemini Developer API (GOOGLE_API_KEY) is the default.
    # Set GOOGLE_GENAI_USE_VERTEXAI=true to call Gemini through Vertex AI instead, using
    # Application Default Credentials and GOOGLE_CLOUD_PROJECT.
    google_genai_use_vertexai: bool = Field(
        default=False,
        alias="GOOGLE_GENAI_USE_VERTEXAI",
        description="Call Gemini through Vertex AI instead of the Gemini Developer API",
    )
    google_cloud_project: str | None = Field(
        default=None, alias="GOOGLE_CLOUD_PROJECT", description="Google Cloud project ID used for Vertex AI"
    )
    google_cloud_location: str = Field(
        default="global", alias="GOOGLE_CLOUD_LOCATION", description="Vertex AI location for Gemini requests"
    )

    # API Keys
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY", description="The API key for the Google API")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY", description="The API key for the OpenAI API")

    # Opik Configuration
    opik_api_key: SecretStr | None = Field(default=None, alias="OPIK_API_KEY", description="The API key for Opik")
    opik_workspace: str | None = Field(default=None, alias="OPIK_WORKSPACE", description="The Opik workspace name")
    opik_project_name: str | None = Field(default=None, alias="OPIK_PROJECT_NAME", description="The Opik project name")

    @model_validator(mode="after")
    def _check_thinking_exclusive(self) -> "Settings":
        if self.thinking_budget is not None and self.thinking_level is not None:
            raise ValueError("`thinking_budget` and `thinking_level` are mutually exclusive; set only one.")
        return self

    @property
    def google_client_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for authenticating a Gemini client.

        Validation is lazy on purpose: it only runs when a Gemini client is
        actually constructed, so non-Gemini code paths never need Google credentials.
        """
        if self.google_genai_use_vertexai:
            if not self.google_cloud_project:
                raise RuntimeError(
                    "GOOGLE_GENAI_USE_VERTEXAI is enabled but GOOGLE_CLOUD_PROJECT is not set. "
                    "Set GOOGLE_CLOUD_PROJECT and authenticate with Application Default Credentials "
                    "(for example, run `gcloud auth application-default login`)."
                )
            return {
                "vertexai": True,
                "project": self.google_cloud_project,
                "location": self.google_cloud_location,
            }

        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable not set. Set it, or switch to Vertex AI by setting "
                "GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT."
            )
        return {"api_key": self.google_api_key.get_secret_value()}

    @property
    def orchestrator_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get the orchestrator configurations."""
        return {
            "gemini-2.5-pro": {
                "identifier": "google_genai:gemini-2.5-pro",
                "params": {
                    "temperature": 0.7,
                    "thinking_budget": 1000,
                    "include_thoughts": True,
                    "max_retries": 3,
                },
            },
            "gemini-3.5-flash": {
                "identifier": "google_genai:gemini-3.5-flash",
                "params": {
                    "temperature": 1,
                    "thinking_level": "low",
                    "include_thoughts": True,
                    "max_retries": 3,
                },
            },
            "gpt-4.1": {
                "identifier": "openai:gpt-4.1",
                "params": {
                    "temperature": 1.0,
                },
            },
        }


# Global settings instance
settings = Settings()
