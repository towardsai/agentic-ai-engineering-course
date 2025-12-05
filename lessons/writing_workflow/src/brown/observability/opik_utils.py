import os

import opik
from loguru import logger

from brown.config import get_settings


def configure() -> None:
    settings = get_settings()
    if settings.OPIK_API_KEY and settings.OPIK_PROJECT_NAME:
        os.environ["OPIK_PROJECT_NAME"] = settings.OPIK_PROJECT_NAME

        try:
            opik.configure(
                api_key=settings.OPIK_API_KEY.get_secret_value(),
                workspace=settings.OPIK_WORKSPACE,
                use_local=False,
                force=True,
                automatic_approvals=True,
            )
            logger.info("Opik configured successfully!")
        except Exception:
            logger.warning(
                "Couldn't configure Opik. Most likely there is a problem with your OPIK_API_KEY or other OPIK_* \
                    environment variables."
            )
    else:
        logger.warning("OPIK_API_KEY and OPIK_PROJECT_NAME are not set. Set them to enable LLMOps with Opik.")


def get_dataset(name: str) -> opik.Dataset | None:
    """
    Get a dataset by name.

    Args:
        name: The name of the dataset to retrieve.

    Returns:
        opik.Dataset | None: The dataset if found, None if there was an error retrieving it.
    """

    client = opik.Opik()
    try:
        dataset = client.get_dataset(name=name)
    except Exception:
        dataset = None

    return dataset


def update_or_create_dataset(name: str, description: str, items: list[dict]) -> opik.Dataset:
    """
    Update an existing dataset or create a new one if it doesn't exist.

    Args:
        name: The name of the dataset to update or create.
        description: The description of the dataset.
        items: The items to insert into the dataset.

    Returns:
        opik.Dataset: The updated or created dataset.
    """

    client = opik.Opik()
    dataset = client.get_or_create_dataset(name=name, description=description)
    dataset.clear()

    dataset.insert(items)

    return dataset
