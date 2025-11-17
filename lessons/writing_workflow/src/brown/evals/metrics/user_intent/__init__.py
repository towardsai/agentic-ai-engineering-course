"""User intent evaluation metric.

This metric evaluates how well generated articles follow provided guidelines
and are anchored in research across two dimensions.
"""

from .metric import UserIntentMetric

__all__ = [
    "UserIntentMetric",
]
