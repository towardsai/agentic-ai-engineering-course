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
    log_level: str = Field(default="INFO", alias="LOG_LEVEL", description="The log level")
    log_level_dependencies: str = Field(default="WARNING", alias="LOG_LEVEL_DEPENDENCIES", description="The log level for dependencies")

    # LLM Configuration
    youtube_transcription_model: str = Field(default="gemini-2.5-flash", description="Model for YouTube transcription")
    scraping_model: str = Field(default="gemini-2.5-flash", description="Model for web scraping")
    query_generation_model: str = Field(default="gemini-2.5-pro", description="Model for query generation")
    source_selection_model: str = Field(default="gemini-2.5-flash", description="Model for source selection")

    # API Keys
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY", description="The API key for the Google API")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY", description="The API key for the OpenAI API")
    perplexity_api_key: SecretStr | None = Field(default=None, alias="PPLX_API_KEY", description="The API key for the Perplexity API")
    firecrawl_api_key: SecretStr | None = Field(default=None, alias="FIRECRAWL_API_KEY", description="The API key for the Firecrawl API")
    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN", description="The GitHub token")

    # Opik Monitoring Configuration
    opik_api_key: SecretStr | None = Field(default=None, alias="OPIK_API_KEY", description="The API key to authenticate with Opik")
    opik_workspace: str | None = Field(
        default=None, alias="OPIK_WORKSPACE", description="The Opik workspace name. If not set, the default workspace will be used."
    )
    opik_project_name: str = Field(default="nova", alias="OPIK_PROJECT_NAME", description="Opik's project name")

    # Descope Authentication
    descope_project_id: str | None = Field(default=None, alias="DESCOPE_PROJECT_ID", description="Descope project ID for authentication")
    descope_base_url: str = Field(default="https://api.descope.com", alias="DESCOPE_BASE_URL", description="Descope API base URL")
    server_base_url: str | None = Field(default=None, alias="SERVER_URL", description="Public URL of this MCP server for OAuth callbacks")

    # HTTP Server settings
    server_host: str = Field(default="localhost", alias="SERVER_HOST", description="Host to bind the HTTP server")
    server_port: int = Field(default=8000, alias="SERVER_PORT", description="Port for the HTTP server")

    # Database settings (local PostgreSQL)
    database_url: str = Field(
        default="postgresql+asyncpg://nova:nova_dev_password@localhost:5432/nova_research",
        alias="DATABASE_URL",
        description="PostgreSQL connection URL (asyncpg driver) - used for local development",
    )

    # Cloud SQL settings (GCP deployment)
    # When CLOUD_SQL_INSTANCE is set, the app uses Cloud SQL Python Connector instead of DATABASE_URL
    cloud_sql_instance: str | None = Field(
        default=None,
        alias="CLOUD_SQL_INSTANCE",
        description="Cloud SQL instance connection name (e.g., project:region:instance)",
    )
    db_user: str = Field(default="nova", alias="DB_USER", description="Database username for Cloud SQL")
    db_pass: SecretStr | None = Field(default=None, alias="DB_PASS", description="Database password for Cloud SQL")
    db_name: str = Field(default="nova_research", alias="DB_NAME", description="Database name for Cloud SQL")

    # Rate limiting settings
    monthly_tool_call_limit: int = Field(
        default=100,
        alias="MONTHLY_TOOL_CALL_LIMIT",
        description="Maximum MCP tool calls per user per month (0 = unlimited)",
    )

    @property
    def is_cloud_sql(self) -> bool:
        """Check if running with Cloud SQL configuration."""
        return self.cloud_sql_instance is not None

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
