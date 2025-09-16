"""Message parsing utilities."""

from .types import InputType, ProcessedInput


def parse_user_input(user_input: str) -> ProcessedInput:
    """Parse user input and return information about what type of input it is.

    This function only analyzes the input and returns metadata about how it should be handled.
    It does NOT execute any actions or modify any state.
    """

    user_input_lower = user_input.strip().lower()

    if user_input_lower.startswith("/prompt/"):
        # This is a prompt loading command
        prompt_name = user_input_lower[8:]  # Remove "/prompt/" prefix
        return ProcessedInput(
            input_type=InputType.COMMAND_PROMPT, should_continue=True, prompt_name=prompt_name, user_message=user_input
        )

    if user_input_lower.startswith("/resource/"):
        # This is a resource reading command
        resource_uri = user_input[10:]  # Remove "/resource/" prefix, keep original case
        return ProcessedInput(
            input_type=InputType.COMMAND_RESOURCE,
            should_continue=True,
            resource_uri=resource_uri,
            user_message=user_input,
        )

    if user_input_lower == "/quit":
        # This is a termination command
        return ProcessedInput(input_type=InputType.TERMINATE, should_continue=False, user_message=user_input)

    # Check if this is a command starting with "/"
    if user_input_lower.startswith("/"):
        # This is a specific informational command
        if user_input_lower == "/tools":
            input_type = InputType.COMMAND_INFO_TOOLS
        elif user_input_lower == "/resources":
            input_type = InputType.COMMAND_INFO_RESOURCES
        elif user_input_lower == "/prompts":
            input_type = InputType.COMMAND_INFO_PROMPTS
        elif user_input_lower == "/model-thinking-switch":
            input_type = InputType.COMMAND_MODEL_THINKING_SWITCH
        else:
            # Unknown command starting with "/", treat as invalid command
            input_type = InputType.COMMAND_UNKNOWN
    else:
        # Not a command, treat as normal message
        input_type = InputType.NORMAL_MESSAGE

    # Handle normal user messages
    if input_type == InputType.NORMAL_MESSAGE:
        return ProcessedInput(input_type=InputType.NORMAL_MESSAGE, should_continue=True, user_message=user_input)

    return ProcessedInput(input_type=input_type, should_continue=True, user_message=user_input)
