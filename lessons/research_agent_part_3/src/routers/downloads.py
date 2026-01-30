"""Download routes for serving database content as files."""

import logging
import uuid

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..db.models import Article
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)


def register_download_routes(mcp: FastMCP) -> None:
    """
    Register all download routes with the FastMCP server.

    These are custom HTTP routes that serve database content as downloadable files.

    Args:
        mcp: The FastMCP server instance
    """

    @mcp.custom_route("/download_research", methods=["GET"])
    async def download_research(request: Request) -> Response:
        """
        Download the research.md file for a given article guideline.

        Query Parameters:
            article_guideline_id: UUID of the article guideline

        Returns:
            Response with research.md content as downloadable file, or error JSON
        """
        try:
            # Get article_guideline_id from query parameters
            article_guideline_id = request.query_params.get("article_guideline_id")

            if not article_guideline_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing required parameter: article_guideline_id"},
                )

            # Validate UUID format
            try:
                article_uuid = uuid.UUID(article_guideline_id)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid article_guideline_id format: {article_guideline_id}"},
                )

            # Query database for article
            session_factory = await get_async_session_factory()
            async with session_factory() as session:
                article = await session.get(Article, article_uuid)

                if not article:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Article with ID '{article_guideline_id}' not found"},
                    )

                # Check if research content exists
                if not article.research:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": f"No research content found for article '{article_guideline_id}'. "
                            "Please run the create_research_file tool first."
                        },
                    )

                # Return research content as downloadable file
                return Response(
                    content=article.research,
                    media_type="text/markdown; charset=utf-8",
                    headers={
                        "Content-Disposition": 'attachment; filename="research.md"',
                    },
                )

        except Exception as e:
            logger.error(f"Error downloading research file: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"},
            )
