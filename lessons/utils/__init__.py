"""
Course AI Agents Utils

A collection of utility functions for AI agents development.
"""

__version__ = "0.1.0"

from .env import load
from .pretty_print import Color, function_call, wrapped

__all__ = ["load", "wrapped", "function_call", "Color"]
