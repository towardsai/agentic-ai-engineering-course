"""Client configuration settings."""

import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings for the MCP Client."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Server settings and paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent, description="The root directory of the mcp_client project"
    )
    server_main_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "mcp_server",
        description="The path to the server's main.py file",
    )
    log_level: int = Field(default=logging.INFO, alias="LOG_LEVEL", description="The log level")
    log_level_dependencies: int = Field(
        default=logging.WARNING, alias="LOG_LEVEL_DEPENDENCIES", description="The log level for dependencies"
    )

    # LLM Configuration
    orchestrator_key: str = Field(default="gemini-2.5-flash", description="Default orchestrator model key")
    model_id: str = Field(default="gemini-2.5-flash", description="Default model ID for LLM operations")
    thinking_budget: int = Field(default=1024, description="Thinking budget for latency vs. depth tradeoff")

    # Agent configuration
    recursion_limit: int = Field(default=100, description="The recursion limit for the agent")

    # API Keys
    google_api_key: SecretStr | None = Field(
        default=None, alias="GOOGLE_API_KEY", description="The API key for the Google API"
    )
    openai_api_key: SecretStr | None = Field(
        default=None, alias="OPENAI_API_KEY", description="The API key for the OpenAI API"
    )

    # Opik Configuration
    opik_api_key: SecretStr | None = Field(default=None, alias="OPIK_API_KEY", description="The API key for Opik")
    opik_workspace: str | None = Field(default=None, alias="OPIK_WORKSPACE", description="The Opik workspace name")
    opik_project_name: str | None = Field(default=None, alias="OPIK_PROJECT_NAME", description="The Opik project name")

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
            "gemini-2.5-flash": {
                "identifier": "google_genai:gemini-2.5-flash",
                "params": {
                    "temperature": 1,
                    "thinking_budget": -1,
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
