"""MCP Resources registration for dynamic data context."""

from typing import Any, Dict

from fastmcp import FastMCP

from ..resources import (
    get_memory_usage_resource,
    get_system_status_resource,
)


def register_mcp_resources(mcp: FastMCP) -> None:
    """Register all MCP resources with the server instance."""

    @mcp.resource("system://status")
    async def system_status() -> Dict[str, Any]:
        """Get system status and health information."""
        return await get_system_status_resource()

    @mcp.resource("system://memory")
    async def memory_usage() -> Dict[str, Any]:
        """Monitor memory usage of the server."""
        return await get_memory_usage_resource()
