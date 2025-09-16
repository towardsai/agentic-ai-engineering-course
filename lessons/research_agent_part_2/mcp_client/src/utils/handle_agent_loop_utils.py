"""Agent loop handling utilities."""

import json
from typing import List

from fastmcp import Client
from google.genai import types

from ..settings import settings
from .llm_utils import (
    LLMClient,
    build_llm_config_with_tools,
    extract_final_answer,
    extract_first_function_call,
    extract_thought_summary,
)
from .print_utils import Color, Style, print_colored


async def execute_tool(name: str, args: dict, client: Client):
    """Execute a tool and return the result."""
    print()
    print_colored(f"⚡ Executing tool '{name}' via MCP server...", Color.CYAN)
    try:
        tool_result = await client.call_tool(name, args)
        print_colored("✅ Tool execution successful!", Color.CYAN)
        return tool_result
    except Exception as e:
        error_msg = f"Tool '{name}' execution failed: {e}"
        print_colored(f"❌ {error_msg}", Color.BRIGHT_RED)
        raise Exception(error_msg)


async def handle_agent_loop(
    conversation_history: List[types.Content],
    tools: List,
    client: Client,
    thinking_enabled: bool,
):
    """Handle the agent loop for tool execution."""
    # Initialize LLM client
    llm_config = build_llm_config_with_tools(tools, thinking_enabled)
    llm_client = LLMClient(settings.model_id, llm_config)

    while True:
        print()
        # Call LLM with current conversation history
        response = await llm_client.generate_content(conversation_history)

        # Extract and display thoughts as separate message (only if enabled)
        if thinking_enabled:
            thoughts = extract_thought_summary(response)
            if thoughts:
                print_colored("🤔 LLM's Thoughts:", Color.BRIGHT_MAGENTA, Style.BOLD)
                print_colored(thoughts, Color.MAGENTA)
                print()

        # Check for function calls
        function_call_info = extract_first_function_call(response)
        if function_call_info:
            name, args = function_call_info

            # Check if this is a tool call
            is_tool = any(tool.name == name for tool in tools)

            if is_tool:
                print()
                print_colored("🔧 Function Call (Tool):", Color.CYAN, Style.BOLD)
                print_colored("  Tool: ", Color.CYAN, end="")
                print_colored(name, Color.CYAN, Style.BOLD)
                print_colored("  Arguments: ", Color.CYAN, end="")
                print_colored(json.dumps(args, indent=2), Color.CYAN)

                # Execute the tool via MCP server
                try:
                    tool_result = await execute_tool(name, args, client)
                    # Add tool result to conversation history
                    tool_response = f"Tool '{name}' executed successfully. Result: {tool_result}"
                    conversation_history.append(types.Content(role="user", parts=[types.Part(text=tool_response)]))
                except Exception as e:
                    # Error already printed by execute_tool, just add to conversation
                    conversation_history.append(types.Content(role="user", parts=[types.Part(text=str(e))]))

            else:
                error_msg = f"Unknown function call: '{name}'. Not found in tools."
                print_colored(f"❌ {error_msg}", Color.BRIGHT_RED)
                conversation_history.append(types.Content(role="user", parts=[types.Part(text=error_msg)]))

        else:
            # Extract final text response - this ends the ReAct loop
            final_text = extract_final_answer(response)
            if final_text:
                print_colored("💬 LLM Response: ", Color.WHITE, end="")
                print(final_text)

                # Add LLM's final response to conversation history
                conversation_history.append(response.candidates[0].content)
            else:
                print_colored("💬 No response generated", Color.BRIGHT_RED)

            print()
            break  # Exit the agent loop
