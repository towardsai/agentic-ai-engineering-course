"""Evaluation module for the Brown article generation system.

This module provides tools and datasets for evaluating the quality of generated articles
across multiple dimensions including content accuracy, guideline adherence, and research anchoring.
"""

from .dataset import EvalDataset

__all__ = ["EvalDataset"]
