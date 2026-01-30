"""MCP Resources registration for dynamic data context."""

from typing import Any, Dict

import opik
from fastmcp import FastMCP

from ..resources import (
    get_memory_usage_resource,
    get_recent_article_guidelines_resource,
    get_recent_local_files_resource,
    get_system_status_resource,
)


def register_mcp_resources(mcp: FastMCP) -> None:
    """Register all MCP resources with the server instance."""

    @mcp.resource("system://status")
    @opik.track(type="general", tags=["resource"])
    async def system_status() -> Dict[str, Any]:
        """Get system status and health information."""
        return await get_system_status_resource()

    @mcp.resource("system://memory")
    @opik.track(type="general", tags=["resource"])
    async def memory_usage() -> Dict[str, Any]:
        """Monitor memory usage of the server."""
        return await get_memory_usage_resource()

    @mcp.resource("articles://recent")
    @opik.track(type="general", tags=["resource"])
    async def recent_article_guidelines() -> Dict[str, Any]:
        """Get the last 5 created article guidelines for the current user."""
        return await get_recent_article_guidelines_resource()

    @mcp.resource("files://recent")
    @opik.track(type="general", tags=["resource"])
    async def recent_local_files() -> Dict[str, Any]:
        """Get the last 5 uploaded local files across all users."""
        return await get_recent_local_files_resource()
