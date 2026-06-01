"""Command handling utilities."""

import logging
from typing import List

from fastmcp import Client

from .print_utils import Color, print_header, print_item
from .types import InputType, ProcessedInput

logger = logging.getLogger(__name__)


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


def handle_command(processed_input: ProcessedInput, tools: List, resources: List, prompts: List, server_names: List[str]):
    """Handle informational commands by listing all capabilities.

    Args:
        processed_input: The parsed user input
        tools: List of all tools from all servers
        resources: List of all resources from all servers
        prompts: List of all prompts from all servers
        server_names: List of server names from the configuration (unused, kept for compatibility)
    """
    if processed_input.input_type == InputType.COMMAND_INFO_TOOLS:
        print_header(f"Tools ({len(tools)})")
        for i, tool in enumerate(tools, 1):
            print_item(tool.name, tool.description, i, Color.BRIGHT_WHITE, Color.YELLOW)

    elif processed_input.input_type == InputType.COMMAND_INFO_RESOURCES:
        print_header(f"Resources ({len(resources)})")
        for i, resource in enumerate(resources, 1):
            print_item(str(resource.uri), resource.description, i, Color.BRIGHT_WHITE, Color.YELLOW)

    elif processed_input.input_type == InputType.COMMAND_INFO_PROMPTS:
        print_header(f"Prompts ({len(prompts)})")
        for i, prompt in enumerate(prompts, 1):
            print_item(prompt.name, prompt.description, i, Color.BRIGHT_WHITE, Color.YELLOW)


def _format_prompt_arguments(prompt) -> str:
    """Format prompt arguments for display.

    Args:
        prompt: The prompt object with arguments attribute

    Returns:
        Formatted string showing required and optional arguments
    """
    if not hasattr(prompt, "arguments") or not prompt.arguments:
        return ""

    required = []
    optional = []
    for arg in prompt.arguments:
        if arg.required:
            required.append(arg.name)
        else:
            optional.append(arg.name)

    parts = []
    if required:
        parts.append(f"Required: {', '.join(required)}")
    if optional:
        parts.append(f"Optional: {', '.join(optional)}")

    return " | ".join(parts) if parts else ""


async def handle_prompt_command(prompt_name: str, prompt_arguments: dict[str, str], prompts: List, client: Client) -> str | None:
    """Handle /prompt/<prompt-name>?arg=value command by retrieving prompt content.

    Args:
        prompt_name: Name of the prompt to retrieve
        prompt_arguments: Dictionary of arguments to pass to the prompt
        prompts: List of available prompts
        client: MCP client instance

    Returns:
        The prompt content as a user message, or None if failed.
    """
    # Find the prompt by name
    matching_prompts = [p for p in prompts if p.name == prompt_name]
    if not matching_prompts:
        logging.error(f"Prompt '{prompt_name}' not found. Available prompts:")
        print()
        for i, prompt in enumerate(prompts, 1):
            args_info = _format_prompt_arguments(prompt)
            description = prompt.description or ""
            if args_info:
                description = f"{description}\n   Arguments: {args_info}" if description else f"Arguments: {args_info}"
            print_item(prompt.name, description, i, Color.BRIGHT_WHITE, Color.MAGENTA)
        return None

    prompt = matching_prompts[0]

    # Check for required arguments
    if hasattr(prompt, "arguments") and prompt.arguments:
        required_args = [arg.name for arg in prompt.arguments if arg.required]
        missing_args = [arg for arg in required_args if arg not in prompt_arguments]
        if missing_args:
            logging.error(f"Missing required arguments for prompt '{prompt_name}': {', '.join(missing_args)}")
            print()
            print(f"Usage: /prompt/{prompt_name}?{'&'.join(f'{arg}=<value>' for arg in required_args)}")
            print()
            print("Arguments:")
            for arg in prompt.arguments:
                req_marker = "(required)" if arg.required else "(optional)"
                arg_desc = arg.description or "No description"
                print(f"  - {arg.name} {req_marker}: {arg_desc}")
            return None

    try:
        # Get the prompt from the MCP server with arguments
        prompt_result = await client.get_prompt(prompt.name, arguments=prompt_arguments)
        prompt_content = str(prompt_result)

        # Return the prompt content
        return prompt_content

    except Exception as e:
        logging.error(f"Failed to retrieve prompt '{prompt_name}': {e}")
        logging.info(f"Error details: {type(e).__name__}: {str(e)}")

        # If error mentions missing arguments, show usage help
        if "Missing required arguments" in str(e) and hasattr(prompt, "arguments") and prompt.arguments:
            print()
            required_args = [arg.name for arg in prompt.arguments if arg.required]
            if required_args:
                print(f"Usage: /prompt/{prompt_name}?{'&'.join(f'{arg}=<value>' for arg in required_args)}")
                print()
                print("Arguments:")
                for arg in prompt.arguments:
                    req_marker = "(required)" if arg.required else "(optional)"
                    arg_desc = arg.description or "No description"
                    print(f"  - {arg.name} {req_marker}: {arg_desc}")

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
