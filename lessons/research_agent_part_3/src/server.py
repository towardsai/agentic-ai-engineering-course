"""Main MCP server implementation with research tools."""

from argparse import ArgumentParser

from fastmcp import FastMCP
from fastmcp.server.auth.providers.descope import DescopeProvider

from .config.settings import settings
from .routers.downloads import register_download_routes
from .routers.prompts import register_mcp_prompts
from .routers.resources import register_mcp_resources
from .routers.tools import register_mcp_tools
from .routers.ui import register_ui_routes
from .utils.logging_utils import configure_logging
from .utils.opik_utils import configure_opik

# Configure logging
configure_logging()

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

    # Configure Descope authentication if credentials are provided
    auth_provider = None
    if settings.descope_project_id and settings.server_base_url:
        auth_provider = DescopeProvider(
            project_id=settings.descope_project_id,
            descope_base_url=settings.descope_base_url,
            base_url=settings.server_base_url,
        )

    # Create the FastMCP server instance
    mcp = FastMCP(
        name=settings.server_name,
        version=settings.version,
        auth=auth_provider,
    )

    # Register all MCP endpoints
    register_mcp_tools(mcp)
    register_mcp_resources(mcp)
    register_mcp_prompts(mcp)

    # Register UI routes (custom HTTP endpoints, not MCP)
    register_ui_routes(mcp)

    # Register download routes (custom HTTP endpoints for file downloads)
    register_download_routes(mcp)

    return mcp


if __name__ == "__main__":
    parser = ArgumentParser(description="MCP Server for Nova Research Agent")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "http"],
        default="http",
        help="The transport protocol to use for the MCP server.",
    )
    args = parser.parse_args()

    # Initialize Opik monitoring if configured
    if configure_opik():
        print(f"📊 Opik monitoring enabled for project: {settings.opik_project_name}")
    else:
        print("📊 Opik monitoring disabled (missing configuration)")

    # Log authentication status
    if settings.descope_project_id and settings.server_base_url:
        print("🔐 Descope authentication enabled")
    else:
        print("🔐 Authentication disabled (missing configuration)")

    mcp = create_mcp_server()

    if args.transport == "http":
        # Use uvicorn with http_app() for proper OAuth discovery endpoint setup
        # This is required for Descope authentication to work correctly
        import uvicorn

        app = mcp.http_app(path="/")
        print(f"🚀 Starting HTTP server on http://{settings.server_host}:{settings.server_port}")
        print(f"   MCP endpoint: http://{settings.server_host}:{settings.server_port}/")
        uvicorn.run(app, host=settings.server_host, port=settings.server_port)
    else:
        # Use stdio transport for direct communication
        mcp.run(transport=args.transport)
