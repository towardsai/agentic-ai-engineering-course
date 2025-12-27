from typing import Any, Callable, Literal

from loguru import logger
from opik import evaluation
from opik.evaluation.metrics import base_metric

from brown.config import get_settings
from brown.config_app import get_app_config as load_app_config

from .opik_utils import get_dataset

app_config = load_app_config()


def evaluate(
    dataset_name: str,
    metrics: list[base_metric.BaseMetric],
    evaluation_task: Callable,
    llm_judge_config: dict[str, Any],
    workers: int = 2,
    nb_samples: int | None = None,
    dataset_item_names: list[str] | None = None,
    split: Literal["val", "test"] = "test",
) -> None:
    """Run an Opik evaluation with the provided metrics and task.

    Fetches the Opik dataset, optionally filters items by name, and executes the
    provided evaluation task per dataset item, scoring with the given metrics.

    Args:
        dataset_name: Name of the Opik dataset to evaluate.
        metrics: List of metric instances used to score task outputs.
        evaluation_task: Callable that generates outputs for a single dataset item.
        llm_judge_config: Configuration passed to the scoring environment (e.g., model info).
        workers: Number of parallel task threads. Defaults to 2.
        nb_samples: Optional cap on the number of samples; if None, evaluates all.
        dataset_item_names: Optional subset of dataset item names to evaluate.
        split: Dataset split to use. "val" includes only "Lesson 10: Memory", "test" includes all other lessons.

    Raises:
        ValueError: If the dataset does not exist.
        AssertionError: If `OPIK_API_KEY` is missing.

    Returns:
        None
    """

    assert get_settings().OPIK_API_KEY, "OPIK_API_KEY is not set. We need it to track the experiment with Opik."

    dataset = get_dataset(dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' does not exist.")
    if dataset_item_names:
        all_dataset_items = dataset.get_items()
        dataset_item_ids = [item["id"] for item in all_dataset_items if item["name"] in dataset_item_names]
        logger.info("Successfuly filtered dataset based on the provided dataset item names.")
        logger.info(f"Evaluating {len(dataset_item_ids)}/{len(all_dataset_items)} dataset items.")
    else:
        dataset_item_ids = None

    logger.info("Starting evaluation...")

    llm_judge_config = {
        "split": split,
        "dataset_name": dataset.name,
        "llm_judge_config": llm_judge_config,
        "app_config": app_config.model_dump(mode="json"),
    }

    logger.info("Evaluation details:")
    logger.info(f"Dataset: {dataset.name}")
    logger.info(f"Metrics: {[m.__class__.__name__ for m in metrics]}")

    evaluation.evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_config=llm_judge_config,
        task_threads=workers,
        nb_samples=nb_samples,
        dataset_item_ids=dataset_item_ids,
    )
