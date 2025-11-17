"""Evaluation metrics for assessing article quality.

This module provides evaluation metrics for measuring article quality across
different dimensions like content accuracy, guideline adherence, and research anchoring.
"""

from opik.evaluation.metrics import base_metric

from brown.models import ModelConfig, SupportedModels

from .follows_gt import FollowsGTMetric
from .user_intent import UserIntentMetric

__all__ = ["UserIntentMetric", "FollowsGTMetric", "build_evaluation_metrics"]


def build_evaluation_metrics(
    metrics: list[str], model: SupportedModels, model_config: ModelConfig | None = None
) -> list[base_metric.BaseMetric]:
    """Get evaluation metrics based on the provided metric names.

    Args:
        metrics: List of metric names to use. Valid values are:
            - "user_intent": Evaluates if article follows guidelines and is anchored in research
            - "follows_gt": Evaluates article structure and content

    Returns:
        List of metric instances

    Raises:
        ValueError: If an unknown metric name is provided

    """
    metrics_mapping = {
        "user_intent": UserIntentMetric(model=model, model_config=model_config),
        "follows_gt": FollowsGTMetric(model=model, model_config=model_config),
    }

    try:
        return [metrics_mapping[metric] for metric in metrics]
    except KeyError as e:
        raise ValueError(f"Unknown metric name: {e}")
