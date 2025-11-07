"""Server configuration settings."""

import logging
from typing import Any, Dict

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings for the Research MCP Server."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Server settings
    server_name: str = Field(default="Nova Research MCP Server", description="The name of the server")
    version: str = Field(default="0.1.0", description="The version of the server")
    log_level: int = Field(default=logging.INFO, alias="LOG_LEVEL", description="The log level")
    log_level_dependencies: int = Field(
        default=logging.WARNING, alias="LOG_LEVEL_DEPENDENCIES", description="The log level for dependencies"
    )

    # LLM Configuration
    youtube_transcription_model: str = Field(default="gemini-2.5-flash", description="Model for YouTube transcription")
    scraping_model: str = Field(default="gemini-2.5-flash", description="Model for web scraping")
    query_generation_model: str = Field(default="gemini-2.5-pro", description="Model for query generation")
    source_selection_model: str = Field(default="gemini-2.5-flash", description="Model for source selection")

    # API Keys
    google_api_key: SecretStr | None = Field(
        default=None, alias="GOOGLE_API_KEY", description="The API key for the Google API"
    )
    openai_api_key: SecretStr | None = Field(
        default=None, alias="OPENAI_API_KEY", description="The API key for the OpenAI API"
    )
    perplexity_api_key: SecretStr | None = Field(
        default=None, alias="PPLX_API_KEY", description="The API key for the Perplexity API"
    )
    firecrawl_api_key: SecretStr | None = Field(
        default=None, alias="FIRECRAWL_API_KEY", description="The API key for the Firecrawl API"
    )
    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN", description="The GitHub token")

    # Opik Monitoring Configuration
    opik_api_key: SecretStr | None = Field(
        default=None, alias="OPIK_API_KEY", description="The API key for Opik monitoring"
    )
    opik_workspace: str | None = Field(default=None, alias="OPIK_WORKSPACE", description="The Opik workspace name")
    opik_project_name: str | None = Field(
        default="nova", alias="OPIK_PROJECT_NAME", description="The Opik project name"
    )

    @property
    def llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get the LLM configurations."""
        return {
            "gemini-2.5-pro": {
                "identifier": "google_genai:gemini-2.5-pro",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 0.7,
                    "thinking_budget": 1000,
                    "include_thoughts": False,
                    "max_retries": 3,
                },
            },
            "gemini-2.5-flash": {
                "identifier": "google_genai:gemini-2.5-flash",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 1,
                    "thinking_budget": 1000,
                    "include_thoughts": False,
                    "max_retries": 3,
                },
            },
            "gpt-5": {
                "identifier": "openai:gpt-5",
                "api_key_env_var": "OPENAI_API_KEY",
                "params": {
                    "temperature": 1,
                },
            },
            "gpt-5-mini": {
                "identifier": "openai:gpt-5-mini",
                "api_key_env_var": "OPENAI_API_KEY",
                "params": {
                    "temperature": 1,
                },
            },
            "perplexity": {
                "identifier": "perplexity:sonar-pro",
                "api_key_env_var": "PPLX_API_KEY",
                "params": {
                    "temperature": 0.7,
                    "max_retries": 3,
                },
            },
        }


# Global settings instance
settings = Settings()
