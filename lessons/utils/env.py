import os
import warnings
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv

# Variables that select between alternative credential sets. They are resolved first
# (from the environment, the .env file, or Colab Secrets) and never prompted for.
SELECTOR_ENV_VARS = ("GOOGLE_GENAI_USE_VERTEXAI", "WEB_SEARCH_PROVIDER")

# Optional configuration resolved the same way: used when provided and skipped otherwise,
# never prompted for because the course works without them.
OPTIONAL_ENV_VARS = ("GOOGLE_CLOUD_LOCATION", "GITHUB_TOKEN")

TRUTHY_VALUES = ("true", "1", "yes")
DEFAULT_VERTEX_LOCATION = "global"

ADC_MISSING_MESSAGE = (
    "Vertex AI is enabled (GOOGLE_GENAI_USE_VERTEXAI=true) but no Application Default "
    "Credentials were found.\n\n"
    "Run this once in your terminal, then restart the notebook kernel:\n\n"
    "    gcloud auth application-default login\n\n"
    "Alternatively, switch back to the Gemini Developer API by removing "
    "GOOGLE_GENAI_USE_VERTEXAI from your .env and setting GOOGLE_API_KEY instead."
)


def load(dotenv_path: Path | None = None, required_env_vars: list[str] | None = None) -> None:
    if dotenv_path is None:
        dotenv_path = Path().absolute().parent.parent / ".env"

    # Detect Google Colab environment and prepare access to Colab Secrets
    is_colab = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ)

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Environment variables loaded from `{dotenv_path}`")
    elif not is_colab:
        # Colab has no .env file by design: credentials come from Colab Secrets instead.
        warnings.warn(f"Environment file `{dotenv_path}` not found.")

    colab_user_data = None
    if is_colab:
        try:
            from google.colab import userdata as _colab_userdata

            colab_user_data = _colab_userdata
            print("Google Colab environment detected. Using Colab Secrets to load environment variables.")
        except Exception:
            colab_user_data = None

    _resolve_selector_env_vars(colab_user_data)

    if required_env_vars is not None:
        required_env_vars = _apply_credential_selectors(required_env_vars)

        for env_var in required_env_vars:
            if env_var not in os.environ or not os.environ.get(env_var):
                # Fallback: if on Colab, try to fetch from Colab Secrets first
                if colab_user_data is not None:
                    try:
                        secret_value = colab_user_data.get(env_var)
                    except Exception:
                        secret_value = None
                    if secret_value:
                        os.environ[env_var] = secret_value
                        continue
                # Final fallback: prompt user to input the variable
                manually_set_envvar(env_var)

    _setup_vertex_ai(is_colab)

    print("Environment variables loaded successfully.")


def _resolve_selector_env_vars(colab_user_data) -> None:
    """Load selector and optional variables from Colab Secrets when absent, without ever prompting."""
    for env_var in SELECTOR_ENV_VARS + OPTIONAL_ENV_VARS:
        if os.environ.get(env_var) or colab_user_data is None:
            continue
        try:
            secret_value = colab_user_data.get(env_var)
        except Exception:
            secret_value = None
        if secret_value:
            os.environ[env_var] = secret_value


def _is_vertex_ai_enabled() -> bool:
    """Whether Gemini calls should go through Vertex AI instead of the Gemini Developer API."""
    return (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() in TRUTHY_VALUES


def _setup_vertex_ai(is_colab: bool) -> None:
    """Make Vertex AI authentication work without extra manual steps in the notebook.

    Vertex AI uses Application Default Credentials rather than an API key, so there is
    nothing to prompt for with getpass. On Colab this triggers the interactive Google
    sign-in that creates those credentials; locally it verifies they already exist and,
    if not, fails immediately with the exact command to run instead of letting a cryptic
    DefaultCredentialsError surface later during the first model call.
    """
    if not _is_vertex_ai_enabled():
        return

    # Keep the location explicit so every client (notebooks, Nova, Brown) agrees on it.
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", DEFAULT_VERTEX_LOCATION)

    if is_colab:
        try:
            from google.colab import auth as colab_auth

            colab_auth.authenticate_user()
        except Exception as error:
            raise RuntimeError(
                "Vertex AI is enabled (GOOGLE_GENAI_USE_VERTEXAI=true) but Colab authentication failed. "
                "Re-run this cell and complete the Google sign-in prompt, or switch back to the Gemini "
                "Developer API by setting GOOGLE_API_KEY instead."
            ) from error
        print("Vertex AI mode: authenticated with Google Cloud through Colab.")
        return

    try:
        import google.auth

        google.auth.default()
    except Exception as error:
        raise RuntimeError(ADC_MISSING_MESSAGE) from error

    print("Vertex AI mode: using your local Application Default Credentials.")


def _apply_credential_selectors(required_env_vars: list[str]) -> list[str]:
    """Adjust the required credentials to match the selected Gemini and web search providers.

    With GOOGLE_GENAI_USE_VERTEXAI=true, Gemini authenticates through Application Default
    Credentials, so GOOGLE_CLOUD_PROJECT is required instead of GOOGLE_API_KEY. With
    WEB_SEARCH_PROVIDER=tavily, TAVILY_API_KEY is required instead of PPLX_API_KEY.
    """
    required = list(required_env_vars)

    if _is_vertex_ai_enabled() and "GOOGLE_API_KEY" in required:
        print("Vertex AI mode detected (GOOGLE_GENAI_USE_VERTEXAI=true): requiring GOOGLE_CLOUD_PROJECT instead of GOOGLE_API_KEY.")
        required.remove("GOOGLE_API_KEY")
        if "GOOGLE_CLOUD_PROJECT" not in required:
            required.append("GOOGLE_CLOUD_PROJECT")

    web_search_provider = (os.environ.get("WEB_SEARCH_PROVIDER") or "perplexity").strip().lower()
    if web_search_provider == "tavily" and "PPLX_API_KEY" in required:
        print("Tavily web search detected (WEB_SEARCH_PROVIDER=tavily): requiring TAVILY_API_KEY instead of PPLX_API_KEY.")
        required.remove("PPLX_API_KEY")
        if "TAVILY_API_KEY" not in required:
            required.append("TAVILY_API_KEY")

    return required


def manually_set_envvar(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Could not load `{var}` from environment file. Please enter it manually: ")
