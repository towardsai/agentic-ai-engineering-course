"""
Multi-Server MCP Client - Interactive MCP client connecting to multiple servers.

This client connects to both Nova (research agent) and Brown (writing workflow)
MCP servers using a single FastMCP Client instance with multi-server configuration.
All tools, resources, and prompts are accessible without prefixes.

Usage:
    uv run python -m src.client
    uv run python -m src.client --config ../mcp_composed_server_config.json
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from fastmcp import Client

from .settings import settings
from .utils.handle_message_utils import handle_user_message
from .utils.logging_utils import configure_logging
from .utils.opik_handler import configure_opik
from .utils.parse_message_utils import parse_user_input
from .utils.print_utils import print_header

# Configure logging
configure_logging()


def print_welcome_message(server_name: str, tools: list, resources: list, prompts: list):
    """Print a welcome message for a server showing its capabilities count."""
    print_header(server_name.replace("-", " ").title())
    print(f"  - {len(tools)} tools available")
    print(f"  - {len(resources)} resources available")
    print(f"  - {len(prompts)} prompts available")
    print()


async def main():
    """Main function for the multi-server MCP client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Server MCP Client")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to MCP servers config file (default: mcp_servers_config.json)",
    )
    args = parser.parse_args()

    # Determine which config file to use
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Make it relative to the current working directory
            config_path = Path.cwd() / config_path
    else:
        config_path = settings.mcp_config_path

    try:
        # Initialize Opik if configured
        if configure_opik():
            logging.info("📊 Opik monitoring enabled")
        else:
            logging.info("📊 Opik monitoring disabled (missing configuration)")

        # Load configuration from JSON file
        logging.info(f"Loading MCP server configuration from: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

        server_names = list(config["mcpServers"].keys())
        logging.info(f"Found {len(server_names)} MCP servers in configuration: {', '.join(server_names)}")

        # Create a single client with multi-server configuration
        logging.info("Connecting to MCP servers...")
        client = Client(config)

        # Connect and fetch capabilities
        async with client:
            logging.info("Fetching capabilities from all servers...")
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            logging.info(f"Total capabilities: {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")

            # Print welcome message showing all capabilities
            print()
            print_welcome_message("All Servers", tools, resources, prompts)

            # Print available commands
            print(
                "Available Commands: /tools, /resources, /prompts, /prompt/<name>?arg=value, /resource/<uri>, /model-thinking-switch, /quit"
            )
            print()

            # Initialize conversation history
            conversation_history = []

            # Initialize thinking state (enabled by default)
            thinking_enabled = True

            # Main conversation loop
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
                        mcp_client=client,
                        thinking_enabled=thinking_enabled,
                        server_names=server_names,
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

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        logging.error("Please ensure the MCP configuration file exists")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        return
    except Exception as e:
        logging.error(f"Failed to initialize MCP client: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
