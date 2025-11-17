from typing import TYPE_CHECKING

from loguru import logger

from . import opik_utils

if TYPE_CHECKING:
    from brown.evals import EvalDataset


def upload_dataset(evaluation_dataset: "EvalDataset") -> None:
    """
    Upload evaluation datasets to the Opik observability platform.

    Args:
        evaluation_dataset: The evaluation dataset containing samples with article guidelines,
            research content, ground truth articles and optionally generated articles to be uploaded.

    Returns:
        None
    """

    samples = evaluation_dataset.model_dump(mode="json")["samples"]
    eval_samples = [sample for sample in samples if not sample["is_few_shot_example"]]
    logger.info(f"Uploading `{len(eval_samples)}/{len(samples)}` evaluation samples to Opik.")
    training_samples = [sample for sample in samples if sample["is_few_shot_example"]]
    logger.info(f"The following `{len(training_samples)}/{len(samples)}` samples will be used for training or as few-shot examples:")
    for sample in training_samples:
        logger.info(f"- `{sample['name']}`")

    opik_utils.update_or_create_dataset(
        name=evaluation_dataset.name,
        description=evaluation_dataset.description,
        items=eval_samples,
    )
