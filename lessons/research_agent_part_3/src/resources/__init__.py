"""MCP resources package."""

from .get_memory_usage_resource import get_memory_usage_resource
from .get_recent_article_guidelines_resource import get_recent_article_guidelines_resource
from .get_recent_local_files_resource import get_recent_local_files_resource
from .get_system_status_resource import get_system_status_resource

__all__ = [
    "get_system_status_resource",
    "get_memory_usage_resource",
    "get_recent_article_guidelines_resource",
    "get_recent_local_files_resource",
]
