from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, FilePath, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # --- Gemini ---

    GOOGLE_API_KEY: SecretStr | None = Field(default=None, description="The API key for the Gemini API.")

    # The Gemini Developer API (GOOGLE_API_KEY) is the default. Set GOOGLE_GENAI_USE_VERTEXAI=true
    # to call Gemini through Vertex AI instead, using Application Default Credentials.
    GOOGLE_GENAI_USE_VERTEXAI: bool = Field(default=False, description="Call Gemini through Vertex AI instead of the Gemini Developer API.")
    GOOGLE_CLOUD_PROJECT: str | None = Field(default=None, description="Google Cloud project ID used for Vertex AI.")
    GOOGLE_CLOUD_LOCATION: str = Field(default="global", description="Vertex AI location for Gemini requests.")

    @property
    def google_client_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for authenticating a Gemini model.

        Validation is lazy on purpose: it only runs when a Gemini model is
        actually constructed, so non-Gemini code paths never need Google credentials.
        """
        if self.GOOGLE_GENAI_USE_VERTEXAI:
            if not self.GOOGLE_CLOUD_PROJECT:
                raise ValueError(
                    "GOOGLE_GENAI_USE_VERTEXAI is enabled but GOOGLE_CLOUD_PROJECT is not set. "
                    "Set GOOGLE_CLOUD_PROJECT and authenticate with Application Default Credentials "
                    "(for example, run `gcloud auth application-default login`)."
                )
            return {
                "vertexai": True,
                "project": self.GOOGLE_CLOUD_PROJECT,
                "location": self.GOOGLE_CLOUD_LOCATION,
            }

        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "Required environment variable `GOOGLE_API_KEY` is not set. Set it, or switch to Vertex AI "
                "by setting GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT."
            )
        return {"api_key": self.GOOGLE_API_KEY.get_secret_value()}

    # --- Opik ---

    OPIK_ENABLED: bool = Field(default=False, description="Whether to use Opik for monitoring and logging.")
    OPIK_WORKSPACE: str | None = Field(default=None, description="Name of the Opik workspace containing the project.")
    OPIK_PROJECT_NAME: str = Field(default="brown", description="Name of the Opik project.")
    OPIK_API_KEY: SecretStr | None = Field(default=None, description="The API key for the Opik API.")

    # --- App Config ---

    CONFIG_FILE: Annotated[
        FilePath, Field(default="configs/course-gemini-flash.yaml", description="Path to the application configuration YAML file.")
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
