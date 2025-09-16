"""Type definitions for client utilities."""

from enum import Enum
from typing import Optional


class InputType(Enum):
    """Types of user input that can be processed."""

    NORMAL_MESSAGE = "normal_message"
    COMMAND_INFO_TOOLS = "command_info_tools"  # Show available tools
    COMMAND_INFO_RESOURCES = "command_info_resources"  # Show available resources
    COMMAND_INFO_PROMPTS = "command_info_prompts"  # Show available prompts
    COMMAND_PROMPT = "command_prompt"  # Commands that load a prompt
    COMMAND_RESOURCE = "command_resource"  # Commands that read a resource
    COMMAND_MODEL_THINKING_SWITCH = "command_model_thinking_switch"  # Toggle thinking mode
    COMMAND_UNKNOWN = "command_unknown"  # Unknown commands starting with "/"
    TERMINATE = "terminate"  # Commands that should terminate the app


class ProcessedInput:
    """Result of processing user input."""

    def __init__(
        self,
        input_type: InputType,
        should_continue: bool = True,
        prompt_name: Optional[str] = None,
        resource_uri: Optional[str] = None,
        user_message: Optional[str] = None,
    ):
        self.input_type = input_type
        self.should_continue = should_continue
        self.prompt_name = prompt_name
        self.resource_uri = resource_uri
        self.user_message = user_message
