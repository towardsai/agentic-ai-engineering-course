"""Message handling utilities."""

import logging
from typing import List

from fastmcp import Client
from google.genai import types

from .command_utils import handle_command, handle_prompt_command, handle_resource_command, handle_thinking_toggle
from .handle_agent_loop_utils import handle_agent_loop
from .print_utils import Color, Style, print_colored
from .types import InputType, ProcessedInput


async def handle_user_message(
    parsed_input: ProcessedInput,
    tools: List,
    resources: List,
    prompts: List,
    conversation_history: List[types.Content],
    mcp_client: Client,
    thinking_enabled: bool,
) -> tuple[bool, bool]:
    """Handle user message based on its parsed input type.

    Returns a tuple of (should_continue, thinking_enabled) where:
    - should_continue: True if the conversation should continue, False if should quit
    - thinking_enabled: Current state of thinking (can be toggled by commands)

    This function processes the user input according to its type and handles the conversation flow.
    """
    # Execute handlers based on input type
    if parsed_input.input_type == InputType.TERMINATE:
        # Handle termination command
        logging.info("👋 Terminating application...")
        return False, thinking_enabled

    elif parsed_input.input_type in [
        InputType.COMMAND_INFO_TOOLS,
        InputType.COMMAND_INFO_RESOURCES,
        InputType.COMMAND_INFO_PROMPTS,
    ]:
        # Handle informational commands (like /tools, /resources, /prompts)
        handle_command(parsed_input, tools, resources, prompts)
        return True, thinking_enabled

    elif parsed_input.input_type == InputType.COMMAND_PROMPT:
        # Handle prompt loading command
        prompt_content = await handle_prompt_command(parsed_input.prompt_name, prompts, mcp_client)
        if prompt_content is not None:
            prompt_message = types.Content(role="user", parts=[types.Part(text=prompt_content)])
            conversation_history.append(prompt_message)
            await handle_agent_loop(conversation_history, tools, mcp_client, thinking_enabled)
        return True, thinking_enabled

    elif parsed_input.input_type == InputType.COMMAND_RESOURCE:
        # Handle resource reading command
        await handle_resource_command(parsed_input.resource_uri, resources, mcp_client)
        return True, thinking_enabled

    elif parsed_input.input_type == InputType.COMMAND_MODEL_THINKING_SWITCH:
        # Handle thinking toggle command
        thinking_enabled = handle_thinking_toggle(thinking_enabled)
        return True, thinking_enabled

    elif parsed_input.input_type == InputType.COMMAND_UNKNOWN:
        # Handle unknown commands starting with "/"
        print_colored(f"❌ Unknown command: '{parsed_input.user_message}'", Color.BRIGHT_RED, Style.BOLD)
        print(
            "Available commands: /tools, /resources, /prompts, /prompt/<name>, "
            "/resource/<uri>, /model-thinking-switch, /quit"
        )
        return True, thinking_enabled

    elif parsed_input.input_type == InputType.NORMAL_MESSAGE:
        # Handle normal user message by adding it to conversation
        user_message = types.Content(role="user", parts=[types.Part(text=parsed_input.user_message)])
        conversation_history.append(user_message)
        await handle_agent_loop(conversation_history, tools, mcp_client, thinking_enabled)
        return True, thinking_enabled

    return True, thinking_enabled
