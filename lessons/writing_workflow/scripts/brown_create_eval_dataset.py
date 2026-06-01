#!/usr/bin/env python3
"""Script to create and manage evaluation datasets."""

import json
from pathlib import Path

import click
from loguru import logger

from brown.evals.dataset import EvalDataset
from brown.observability import upload_dataset


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("inputs/evals/dataset"),
    help="Directory containing the evaluation dataset metadata and data files.",
)
@click.option(
    "--name",
    type=str,
    show_default=True,
    help="Dataset name to register.",
)
@click.option(
    "--description",
    type=str,
    show_default=True,
    help="Dataset description.",
)
def main(
    input_dir: Path,
    name: str,
    description: str,
) -> None:
    """Create and manage evaluation datasets from markdown files.

    This script loads evaluation datasets from a directory structure containing
    metadata.json and markdown files, following the pattern:

    input_dir/
    ├── metadata.json
    └── data/
        └── {sample_directory}/
            ├── article_guideline.md
            ├── research.md
            ├── article_ground_truth.md
            └── article_generated.md

    Args:
        input_dir: Directory containing the evaluation dataset metadata and data files.
        name: Dataset name to register.
        description: Dataset description.
    """

    logger.info(f"Loading evaluation dataset from: {input_dir}")

    try:
        dataset = EvalDataset.load_dataset(input_dir, name=name, description=description)
        logger.success(f"Successfully loaded `{len(dataset.samples)}` evaluation sample(s).")

        logger.info(f"Uploading dataset to Opik: `{dataset.name}`")
        upload_dataset(dataset)
        logger.success(f"Successfully uploaded dataset to Opik: `{dataset.name}`")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise click.ClickException(f"Dataset loading failed: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        raise click.ClickException(f"Invalid metadata file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.ClickException(f"Dataset loading failed: {e}")


if __name__ == "__main__":
    main()
