"""UI routes registration for web-based user interfaces."""

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from ..ui.upload_local_files_ui import (
    get_local_files_upload_page,
    get_upload_status,
    post_upload_local_file,
)
from ..ui.upload_ui import get_upload_page, post_upload_guideline


def register_ui_routes(mcp: FastMCP) -> None:
    """
    Register all UI routes with the FastMCP server.

    These are custom HTTP routes that serve web UIs, not MCP tools/resources/prompts.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.custom_route("/upload_article_guideline", methods=["GET"])
    async def upload_page(request: Request) -> HTMLResponse:
        """Serve the article guideline upload page."""
        return await get_upload_page(request)

    @mcp.custom_route("/upload_article_guideline", methods=["POST"])
    async def upload_guideline(request: Request) -> JSONResponse:
        """Handle article guideline file upload."""
        return await post_upload_guideline(request)

    @mcp.custom_route("/upload_local_files", methods=["GET"])
    async def local_files_upload_page(request: Request) -> HTMLResponse:
        """Serve the local files upload page."""
        return await get_local_files_upload_page(request)

    @mcp.custom_route("/upload_local_files/status", methods=["GET"])
    async def local_files_status(request: Request) -> JSONResponse:
        """Get upload status for local files."""
        return await get_upload_status(request)

    @mcp.custom_route("/upload_local_files", methods=["POST"])
    async def upload_local_file(request: Request) -> JSONResponse:
        """Handle local file upload."""
        return await post_upload_local_file(request)
