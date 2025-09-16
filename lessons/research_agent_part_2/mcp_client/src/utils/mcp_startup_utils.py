"""MCP startup utilities."""

from typing import List

from fastmcp import Client


async def get_capabilities_from_mcp_client(client: Client) -> tuple[List, List, List]:
    """Get available capabilities."""
    async with client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

    return tools, resources, prompts


def print_startup_info(tools: List, resources: List, prompts: List):
    """Print startup information about available capabilities."""
    print(f"🛠️  Available tools: {len(tools)}")
    print(f"📚 Available resources: {len(resources)}")
    print(f"💬 Available prompts: {len(prompts)}")
    print()
    print(
        "Available Commands: /tools, /resources, /prompts, /prompt/<name>, "
        "/resource/<uri>, /model-thinking-switch, /quit"
    )
    print()
