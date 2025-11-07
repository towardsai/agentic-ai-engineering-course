"""Command handling utilities."""

import logging
from typing import List

from fastmcp import Client

from .print_utils import Color, print_header, print_item
from .types import InputType, ProcessedInput


def handle_thinking_toggle(thinking_enabled: bool) -> bool:
    """Handle /model-thinking-switch command by toggling thinking state.

    Returns:
        bool: The new thinking state (opposite of input)
    """
    new_state = not thinking_enabled
    if new_state:
        logging.info("🤔 Thinking mode ENABLED")
        logging.info("LLM thoughts will be displayed during responses")
    else:
        logging.info("🤐 Thinking mode DISABLED")
        logging.info("LLM will act without thinking ahead")

    return new_state


def handle_command(processed_input: ProcessedInput, tools: List, resources: List, prompts: List):
    """Handle informational commands.

    This function only handles informational commands (COMMAND_INFO_* types).
    """
    if processed_input.input_type == InputType.COMMAND_INFO_TOOLS:
        print_header("🛠️  Available Tools")
        for i, tool in enumerate(tools, 1):
            print_item(tool.name, tool.description, i, Color.BRIGHT_WHITE, Color.YELLOW)

    elif processed_input.input_type == InputType.COMMAND_INFO_RESOURCES:
        print_header("📚 Available Resources")
        for i, resource in enumerate(resources, 1):
            print_item(resource.uri, resource.description, i, Color.BRIGHT_WHITE, Color.YELLOW)

    elif processed_input.input_type == InputType.COMMAND_INFO_PROMPTS:
        print_header("💬 Available Prompts")
        for i, prompt in enumerate(prompts, 1):
            print_item(prompt.name, prompt.description, i, Color.BRIGHT_WHITE, Color.YELLOW)


async def handle_prompt_command(prompt_name: str, prompts: List, client: Client) -> str | None:
    """Handle /prompt/<prompt-name> command by retrieving prompt content.

    Returns:
        types.Content: The prompt content as a user message, or None if failed.
    """
    # Find the prompt by name
    matching_prompts = [p for p in prompts if p.name == prompt_name]
    if not matching_prompts:
        logging.error(f"Prompt '{prompt_name}' not found. Available prompts:")
        print()
        for i, prompt in enumerate(prompts, 1):
            print_item(prompt.name, prompt.description, i, Color.BRIGHT_WHITE, Color.MAGENTA)
        return None

    prompt = matching_prompts[0]

    try:
        # Get the prompt from the MCP server
        prompt_result = await client.get_prompt(prompt.name)
        prompt_content = str(prompt_result)

        # Return the prompt content
        return prompt_content

    except Exception as e:
        logging.error(f"Failed to retrieve prompt '{prompt_name}': {e}")
        logging.info(f"Error details: {type(e).__name__}: {str(e)}")
        return None


async def handle_resource_command(resource_uri: str, resources: List, client: Client) -> None:
    """Handle /resource/<resource-uri> command by retrieving resource content.

    This function prints the resource content directly and does not return a value
    (unlike handle_prompt_command which returns content for LLM processing).
    """
    # Find the resource by URI
    matching_resources = [r for r in resources if str(r.uri) == resource_uri]
    if not matching_resources:
        logging.error(f"Resource '{resource_uri}' not found. Available resources:")
        print()
        for i, resource in enumerate(resources, 1):
            print_item(str(resource.uri), resource.description, i, Color.BRIGHT_WHITE, Color.CYAN)
        return

    resource = matching_resources[0]

    try:
        # Read the resource from the MCP server
        resource_result = await client.read_resource(resource_uri)

        print_header(f"📖 Resource Content: {resource_uri}")
        print(resource_result[0].text)
        print()

    except Exception as e:
        logging.error(f"Failed to read resource '{resource_uri}': {e}")
        logging.info(f"Error details: {type(e).__name__}: {str(e)}")
