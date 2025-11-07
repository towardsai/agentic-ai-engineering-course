"""MCP resources package."""

from .get_memory_usage_resource import get_memory_usage_resource
from .get_system_status_resource import get_system_status_resource

__all__ = [
    "get_system_status_resource",
    "get_memory_usage_resource",
]
