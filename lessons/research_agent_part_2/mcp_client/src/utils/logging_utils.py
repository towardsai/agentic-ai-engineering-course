"""Logging configuration utilities for the MCP Client."""

import logging

from ..settings import settings


def configure_logging():
    """Configure logging for the MCP client and external libraries.

    This function sets up logging levels for:
    - Root logger (main application)
    - External libraries (opik, httpx, etc.)
    """
    # Configure root logger
    logging.getLogger().setLevel(settings.log_level)

    # Configure logging for external libraries to respect our dependency log level
    logging.getLogger("opik").setLevel(settings.log_level_dependencies)
    logging.getLogger("httpx").setLevel(settings.log_level_dependencies)  # opik uses httpx for HTTP requests
    logging.getLogger("google").setLevel(settings.log_level_dependencies)  # Google GenAI library
    logging.getLogger("fastmcp").setLevel(settings.log_level_dependencies)  # FastMCP library
