"""Main MCP server implementation with research tools."""

import os
from argparse import ArgumentParser

from fastmcp import FastMCP

from .config.settings import settings
from .routers.prompts import register_mcp_prompts
from .routers.resources import register_mcp_resources
from .routers.tools import register_mcp_tools
from .utils.logging_utils import configure_logging
from .utils.opik_utils import configure_opik

# Configure logging
configure_logging()

# Git must never prompt for credentials: the server runs without an interactive
# terminal, so a prompt (e.g. from gitingest's clone hitting a 401) blocks the
# tool call forever. With this set, git fails fast and the error is reported.
os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")

# ============================================================================
# MAIN EXECUTION AND EXPORT FUNCTIONS
# ============================================================================


def create_mcp_server() -> FastMCP:
    """
    Create and configure the MCP server instance.

    This function can be imported to get a configured MCP server
    for use with in-memory transport in clients.

    Returns:
        FastMCP: Configured MCP server instance
    """
    # Create the FastMCP server instance
    mcp = FastMCP(
        name=settings.server_name,
        version=settings.version,
    )

    # Register all MCP endpoints
    register_mcp_tools(mcp)
    register_mcp_resources(mcp)
    register_mcp_prompts(mcp)

    return mcp


if __name__ == "__main__":
    parser = ArgumentParser(description="MCP Server for Nova Research Agent")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="The transport protocol to use for the MCP server (stdio or streamable-http).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8001,
        help="Port number for HTTP transport (default: 8001)",
    )
    args = parser.parse_args()

    # Initialize Opik monitoring if configured
    if configure_opik():
        print(f"📊 Opik monitoring enabled for project: {settings.opik_project_name}")
    else:
        print("📊 Opik monitoring disabled (missing configuration)")

    mcp = create_mcp_server()

    # Run the server with the specified transport
    if args.transport == "streamable-http":
        mcp.run(transport=args.transport, port=args.port)
    else:
        mcp.run(transport=args.transport)
