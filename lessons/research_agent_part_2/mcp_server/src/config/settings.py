"""Server configuration settings."""

import logging
from typing import Any, Dict, Literal

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
    youtube_transcription_model: str = Field(default="gemini-3.5-flash", description="Model for YouTube transcription")
    scraping_model: str = Field(default="gemini-3.5-flash", description="Model for web scraping")
    query_generation_model: str = Field(default="gemini-2.5-pro", description="Model for query generation")
    source_selection_model: str = Field(default="gemini-3.5-flash", description="Model for source selection")

    # Web search configuration
    web_search_provider: Literal["tavily", "perplexity"] = Field(
        default="perplexity",
        alias="WEB_SEARCH_PROVIDER",
        description="Web search provider used by run_web_research",
    )
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        alias="TAVILY_SEARCH_DEPTH",
        description="Tavily search depth",
    )
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="TAVILY_MAX_RESULTS",
        description="Maximum Tavily results returned for each query",
    )
    tavily_chunks_per_source: int = Field(
        default=3,
        ge=1,
        le=3,
        alias="TAVILY_CHUNKS_PER_SOURCE",
        description="Relevant content chunks returned per source for advanced Tavily search",
    )

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
    perplexity_api_key: SecretStr | None = Field(default=None, alias="PPLX_API_KEY", description="The API key for the Perplexity API")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY", description="The API key for the Tavily Search API")
    firecrawl_api_key: SecretStr | None = Field(default=None, alias="FIRECRAWL_API_KEY", description="The API key for the Firecrawl API")
    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN", description="The GitHub token")

    # Opik Monitoring Configuration
    opik_api_key: SecretStr | None = Field(default=None, alias="OPIK_API_KEY", description="The API key to authenticate with Opik")
    opik_workspace: str | None = Field(
        default=None,
        alias="OPIK_WORKSPACE",
        description="The Opik workspace name. If not set, the default workspace will be used.",
    )
    opik_project_name: str = Field(default="nova", alias="OPIK_PROJECT_NAME", description="Opik's project name")

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
            "gemini-3.5-flash": {
                "identifier": "google_genai:gemini-3.5-flash",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 1,
                    "thinking_level": "low",
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
