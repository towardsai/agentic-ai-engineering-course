"""Memory usage resource implementation."""

import logging
import os
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)


async def get_memory_usage_resource() -> Dict[str, Any]:
    """
    Monitor memory usage of research operations.

    Returns:
        Dict with process and system memory usage information
    """
    try:
        process = psutil.Process(os.getpid())

        return {
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "process_memory_percent": process.memory_percent(),
            "system_memory": {
                "total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "used_percent": psutil.virtual_memory().percent,
            },
        }
    except ImportError:
        return {"error": "psutil not available, memory monitoring disabled"}
