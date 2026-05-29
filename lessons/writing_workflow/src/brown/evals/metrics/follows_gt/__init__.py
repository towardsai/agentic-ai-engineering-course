"""Follows ground truth evaluation metric.

This metric evaluates how well generated articles follow the structure and content
of expected ground truth articles across multiple dimensions.
"""

from .metric import FollowsGTMetricLLMJudge

__all__ = ["FollowsGTMetricLLMJudge"]
