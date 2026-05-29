import os
import warnings
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv


def load(dotenv_path: Path | None = None, required_env_vars: list[str] | None = None) -> None:
    if dotenv_path is None:
        dotenv_path = Path().absolute().parent.parent / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Environment variables loaded from `{dotenv_path}`")
    else:
        warnings.warn(f"Environment file `{dotenv_path}` not found.")

    # Detect Google Colab environment and prepare access to Colab Secrets
    is_colab = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ)
    colab_user_data = None
    if is_colab:
        try:
            from google.colab import userdata as _colab_userdata

            colab_user_data = _colab_userdata
            print("Google Colab environment detected. Using Colab Secrets to load environment variables.")
        except Exception:
            colab_user_data = None

    if required_env_vars is not None:
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

    print("Environment variables loaded successfully.")


def manually_set_envvar(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Could not load `{var}` from environment file. Please enter it manually: ")
