"""MCP Prompts registration for adaptive conversation starters."""

from fastmcp import FastMCP

# Import prompts with a different name to avoid naming collision
from ..prompts.research_instructions_prompt import full_research_instructions_prompt as _get_research_instructions


def register_mcp_prompts(mcp: FastMCP) -> None:
    """Register all MCP prompts with the server instance."""

    @mcp.prompt()
    async def full_research_instructions_prompt() -> str:
        """Complete Nova research agent workflow instructions."""
        return await _get_research_instructions()
