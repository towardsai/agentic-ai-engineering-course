#!/usr/bin/env python3

"""Script to run evaluations on the Brown agent using evaluation datasets."""

from pathlib import Path
from typing import List, Literal

import click
from loguru import logger

from brown.evals.metrics import build_evaluation_metrics
from brown.evals.tasks import create_evaluation_task
from brown.models.config import ModelConfig
from brown.models.get_model import SupportedModels
from brown.observability.evaluation import evaluate


@click.command()
@click.option(
    "--dataset-name",
    type=str,
    required=True,
    help="Name of the evaluation dataset to use for evaluation.",
)
@click.option(
    "--metrics",
    type=str,
    multiple=True,
    default=["follows_gt"],
    help="Metrics to use for evaluation. Available metrics: 'follows_gt' (evaluates agains ground truth),"
    "'user_intent' (evaluates guideline adherence and research anchoring).",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="test",
    help="Dataset split to use. 'val' includes only 'Lesson 10: Memory', 'test' includes all other lessons.",
)
@click.option(
    "--cache-dir",
    type=click.Path(writable=True, path_type=Path),
    required=True,
    help="Cache directory for generated articles and evaluation results.",
)
@click.option(
    "--workers",
    type=int,
    default=2,
    help="Number of parallel workers to use for evaluation.",
)
@click.option(
    "--nb-samples",
    type=int,
    help="Number of samples to evaluate. If not provided, evaluates the entire dataset.",
)
@click.option(
    "--read-from-cache",
    is_flag=True,
    help="Read from cache.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode.",
)
def main(
    dataset_name: str,
    metrics: List[str],
    split: Literal["val", "test"],
    cache_dir: Path,
    workers: int,
    nb_samples: int | None,
    read_from_cache: bool,
    debug: bool,
) -> None:
    """Run evaluation over an Opik dataset with the Brown agent.

    Loads the specified evaluation dataset, generates articles with Brown for each
    sample, and scores results using the selected metrics.

    Args:
        dataset_name: Name of the Opik dataset to evaluate.
        metrics: Names of scoring metrics to use. Supported: "follows_gt", "user_intent".
        split: Dataset split to use. "val" includes only "Lesson 10: Memory", "test" includes all other lessons.
        cache_dir: Directory used to cache generated artifacts and results.
        workers: Number of parallel workers to use during evaluation.
        nb_samples: Number of samples to evaluate. If None, evaluates the full dataset.
        read_from_cache: If True, read generated outputs from cache instead of generating.
        debug: If True, enables additional debugging information in the workflow.
        metrics: Names of scoring metrics to use. Supported: "follows_gt", "user_intent".

    Raises:
        click.ClickException: If the evaluation run fails.

    Returns:
        None
    """

    split = split.lower()
    logger.info(f"Starting evaluation with dataset `{dataset_name}` on split `{split}`")

    cache_dir.mkdir(parents=True, exist_ok=True)

    evaluation_task = create_evaluation_task(
        cache_dir=cache_dir,
        read_from_cache=read_from_cache,
        clean_cache=False,
        debug=debug,
    )

    model = SupportedModels.GOOGLE_GEMINI_25_PRO
    model_config = ModelConfig(temperature=0.0, thinking_budget=int(1024 * 0.5), include_thoughts=False, max_retries=3)
    evaluation_metrics = build_evaluation_metrics(metrics, model, model_config)

    if split == "val":
        dataset_item_names = ["Lesson 10: Memory"]
    elif split == "test":
        dataset_item_names = [
            "Lesson 2: Workflows vs. Agents",
            "Lesson 3: Context Engineering",
            "Lesson 5: Workflow Patterns",
            "Lesson 6: Tools",
            "Lesson 8: ReAct Practice",
            "Lesson 9: Retrieval-Augmented Generation (RAG)",
            "Lesson 11: Multimodal Data",
        ]
    else:
        raise ValueError(f"Invalid split: {split}")

    try:
        evaluate(
            dataset_name=dataset_name,
            metrics=evaluation_metrics,
            evaluation_task=evaluation_task,
            llm_judge_config={"model": model, **model_config.model_dump()},
            workers=workers,
            nb_samples=nb_samples,
            dataset_item_names=dataset_item_names,
            split=split,
        )

        logger.success("Evaluation completed successfully!")
        logger.info(f"Results saved in: {cache_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
