"""
Nova MCP Client - Interactive MCP client with configurable transport options.

Usage:
    # In-memory transport (default - MCP server runs in same process)
    uv run python -m src.client

    # Stdio transport (MCP server runs as external process)
    uv run python -m src.client --transport stdio

Transport Options:
    - in-memory: MCP server runs in the same process (faster, easier debugging)
    - stdio: MCP server runs as external process with explicit configuration:
        * Transport: stdio
        * Command: uv --directory <server_path> run mcp-server --transport stdio
        * Better isolation and process separation
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from fastmcp import Client

from .settings import settings
from .utils.handle_message_utils import handle_user_message
from .utils.logging_utils import configure_logging
from .utils.mcp_startup_utils import get_capabilities_from_mcp_client, print_startup_info
from .utils.opik_handler import configure_opik
from .utils.parse_message_utils import parse_user_input

# Configure logging
configure_logging()


async def main():
    """Main function to demonstrate FastMCP client with configurable transport."""
    parser = argparse.ArgumentParser(description="Nova MCP Client")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["in-memory", "stdio"],
        default="in-memory",
        help="Transport method: 'in-memory' (default) or 'stdio' for external MCP server",
    )
    args = parser.parse_args()

    try:
        # Initialize Opik if configured
        if configure_opik():
            logging.info("📊 Opik monitoring enabled")
        else:
            logging.info("📊 Opik monitoring disabled (missing configuration)")

        # Initialize MCP client based on transport mode
        if args.transport == "in-memory":
            # Add the project root to Python path to enable importing mcp_server
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            from mcp_server.src.server import create_mcp_server

            logging.info("🚀 Starting MCP client with in-memory transport...")
            mcp_server = create_mcp_server()
            mcp_client = Client(mcp_server)

        elif args.transport == "stdio":
            # Define server configuration explicitly
            config = {
                "mcpServers": {
                    "research-agent": {
                        "transport": "stdio",
                        "command": "uv",
                        "args": [
                            "--directory",
                            str(settings.server_main_path),
                            "run",
                            "-m",
                            "src.server",
                            "--transport",
                            "stdio",
                        ],
                    }
                }
            }

            logging.info("🚀 Starting MCP client with stdio transport...")
            mcp_client = Client(config)

        # Print startup information about MCP server
        tools, resources, prompts = await get_capabilities_from_mcp_client(mcp_client)
        print_startup_info(tools, resources, prompts)

        # Initialize conversation history
        conversation_history = []

        # Initialize thinking state (enabled by default)
        thinking_enabled = True

        # Main conversation loop
        async with mcp_client:
            while True:
                try:
                    # Get user input
                    user_input = input("👤 You: ").strip()
                    if not user_input:
                        continue

                    # Parse the user input to determine what type it is
                    parsed_input = parse_user_input(user_input)

                    # Handle the user message and determine if we should continue
                    should_continue, thinking_enabled = await handle_user_message(
                        parsed_input=parsed_input,
                        tools=tools,
                        resources=resources,
                        prompts=prompts,
                        conversation_history=conversation_history,
                        mcp_client=mcp_client,
                        thinking_enabled=thinking_enabled,
                    )

                    if not should_continue:
                        break

                except KeyboardInterrupt:
                    print()
                    logging.info("👋 Interrupted by user. Goodbye!")
                    break
                except Exception as e:
                    print()
                    logging.error(f"Error: {e}")
                    logging.info("Continuing conversation...\n")

    except Exception as e:
        logging.error(f"Failed to initialize MCP client: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
