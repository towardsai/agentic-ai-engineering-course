"""System status resource implementation."""

import logging
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def get_system_status_resource() -> Dict[str, Any]:
    """
    Get system status and health information.

    Returns:
        Dict with system status and environment information
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            key: value
            for key, value in os.environ.items()
            if not any(secret in key.lower() for secret in ["key", "token", "secret", "password"])
        },
    }
