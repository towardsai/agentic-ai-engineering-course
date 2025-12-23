"""Brown MCP Server for article generation and editing.

This module provides a FastMCP server that exposes Brown's article generation
and editing capabilities through Model Context Protocol (MCP) tools and prompts.
The server allows external applications to generate articles from research data
and edit existing articles based on user feedback.

Example:
    The server can be run directly:
        python -m brown.mcp.server

    Or used as an MCP server in other applications that support MCP protocol.
"""

import argparse
import uuid
from pathlib import Path

from fastmcp import Context, FastMCP
from langgraph.pregel.main import RunnableConfig
from loguru import logger

from brown.builders import build_loaders, build_short_term_memory
from brown.config_app import get_app_config as load_app_config
from brown.observability import tracing
from brown.workflows import (
    build_edit_article_workflow,
    build_edit_selected_text_workflow,
    build_generate_article_workflow,
)
from brown.workflows.types import WorkflowProgress

app_config = load_app_config()

logger.info("Initializing Brown MCP Server...")
mcp = FastMCP("Brown MCP Server")
logger.info("Brown MCP Server initialized successfully")


async def parse_message(chunk_data: dict, ctx: Context, prefix: str = "") -> None:
    """Parse and report workflow streaming messages to the MCP client.

    This function handles different types of streaming data from Brown workflows
    and converts them into appropriate MCP context messages and progress reports.

    Args:
        chunk_data: The streaming data from the workflow, can be a string message
                   or a dictionary containing progress information.
        ctx: MCP context for sending messages and progress updates to the client.
        prefix: Optional prefix to add to all messages for context identification.

    Raises:
        ValueError: If chunk_data is not a supported type (str or dict).
    """
    if prefix:
        prefix = f"{prefix}: "

    if isinstance(chunk_data, str):
        await ctx.info(f"{prefix}{chunk_data}")
    elif isinstance(chunk_data, dict):
        message = WorkflowProgress(**chunk_data)
        await ctx.info(f"{prefix}{message.progress}%: {message.message}")
        await ctx.report_progress(progress=message.progress, total=100, message=f"{prefix}{message.message}")
    else:
        raise ValueError(f"Unsupported chunk data type: {type(chunk_data)}")


@mcp.prompt
def generate_article_prompt(dir_path: Path) -> str:
    """Retrieve a prompt that will trigger the article generation workflow using Brown.

    Args:
        dir_path: Path to the directory containing article resources (research,
            guidelines, etc.).

    Returns:
        A formatted prompt string that will trigger the "generate_article" tool of the Brown MCP Server,
        which will take care of everything.
    """
    return f"""
Using Brown hosted as an MCP server, generate an article using all the necessary resources from 
the following directory: `{dir_path}`. Don't check if any expected files are missing, just trigger 
the "generate_article" tool of the Brown MCP Server, which will take care of everything.
"""


@mcp.tool
async def generate_article(dir_path: Path, ctx: Context) -> str:
    """Generate an article from scratch using Brown's article generation workflow.

    This tool orchestrates the complete article generation process by leveraging
    the Brown MCP Server workflow. It processes research data, guidelines, and other
    resources from the specified directory to create a comprehensive article.

    Args:
        dir_path: Path to the directory containing article resources including:
            - Research data and sources
            - Article guidelines and style requirements
            - Any additional context files
        ctx: MCP context for streaming progress updates and user communication.

    Returns:
        A string containing a success confirmation message after the article
        has been generated and saved to the specified directory.

    Example:
        >>> result = await generate_article(Path("/path/to/article/resources"), ctx)
        >>> print(result)
        Article generation completed successfully!

        # The generated article will be saved as article.md in the specified directory,
        # with multiple review iterations applied as article_000.md, article_001.md, etc.
    """

    async with build_short_term_memory(app_config) as checkpointer:
        generate_article_workflow = build_generate_article_workflow(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        tracer = tracing.build_handler(thread_id, tags=["generate"])
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            },
            "callbacks": [tracer],
        }

        async for chunk in generate_article_workflow.astream({"dir_path": dir_path}, config=config, stream_mode=["custom", "values"]):
            _, chunk_data = chunk
            await parse_message(chunk_data, ctx)

    return "Article generation completed successfully!"


@mcp.prompt
def edit_article_prompt(human_feedback: str = "") -> str:
    """Retrieve a prompt that will trigger the article editing workflow using Brown.

    Args:
        dir_path: Path to the directory containing article resources (article.md,
            guidelines, research, etc.).
        human_feedback: User's feedback or instructions for editing the entire article.

    Returns:
        A formatted prompt string that will trigger the "edit_article" tool of the Brown MCP Server,
        which will take care of everything.
    """
    return f"""
Using Brown hosted as an MCP server, edit an entire article based on human feedback 
and other expected requirements. Don't check if any expected files are missing, 
just trigger the "edit_article" tool of the Brown MCP Server, which will take care 
of everything.

Human feedback:
<human_feedback>
{human_feedback}
</human_feedback>

If the <human_feedback> is empty, you will infer it from the previous messages. If there are no other messages 
to infer from, use an empty string. Don't ever fill it in with things such as "Please provide more details" or
fill it in with generic stuff.
"""


@mcp.tool
async def edit_article(
    article_path: str,
    human_feedback: str,
    ctx: Context,
) -> str:
    """Edit an entire article based on human feedback and expected requirements
    using Brown's article editing workflow.

    This tool orchestrates the complete article editing process by leveraging
    the Brown MCP Server workflow. It processes human feedback and article context
    to provide comprehensive content modifications at the article level.

    Args:
        article_path: Path to the article that has to be edited.
        human_feedback: User's feedback or instructions for editing the entire article.
        ctx: MCP context for streaming progress updates and user communication.

    Returns:
        The fully edited article content in markdown format plus instructions on what to do with the edited article.

    Example:
        >>> result = await edit_article(
        ...     "/path/to/article/resources",
        ...     "Make the introduction more engaging and add more technical depth to section 2",
        ...     ctx
        ... )
        >>> print(result)
        Here is the edited article:
        <article>

        Here is what you have to do with the edited article:
        <instructions>
    """

    async with build_short_term_memory(app_config) as checkpointer:
        edit_article_workflow = build_edit_article_workflow(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        tracer = tracing.build_handler(thread_id, tags=["edit"])
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "callbacks": [tracer]}

        dir_path = Path(article_path).parent
        await ctx.info(f"Editing article from file {article_path}")
        await ctx.info(f"Using directory `{dir_path}` as context")

        final_result = None
        async for chunk in edit_article_workflow.astream(
            {
                "dir_path": Path(dir_path),
                "human_feedback": human_feedback,
            },
            config=config,
            stream_mode=["custom", "values"],
        ):
            chunk_type, chunk_data = chunk
            await parse_message(chunk_data, ctx)

            if chunk_type == "values":
                final_result = chunk_data

    return final_result or "Article editing completed successfully!"


@mcp.prompt
def edit_selected_text_prompt(human_feedback: str = "") -> str:
    """Retrieve a prompt that will trigger the selected text editing workflow using Brown.

    Args:
        dir_path: Path to the directory containing article resources (article.md,
            guidelines, research, etc.).
        selected_text: The specific text selected from the article that needs to be edited.
        first_line_number: Line number from the original file where the selected text starts.
        last_line_number: Line number from the original file where the selected text ends.
        human_feedback: User's feedback or instructions for the edit.

    Returns:
        A formatted prompt string that will trigger the "edit_selected_text" tool of the Brown MCP Server,
        which will take care of everything.
    """
    return f"""
Using Brown hosted as an MCP server, edit the selected text from the article based on human feedback and 
other expected requirements. Don't check if any expected files are missing, just trigger the "edit_selected_text" 
tool of the Brown MCP Server, which will take care of everything.

Human feedback:
<human_feedback>
{human_feedback}
</human_feedback>

If the <human_feedback> is empty, you will infer it from the previous messages. If there are no other messages 
to infer from, use an empty string. Don't ever fill it in with things such as "Please provide more details" or
fill it in with generic stuff.
"""


@mcp.tool
async def edit_selected_text(
    article_path: str,
    human_feedback: str,
    selected_text: str,
    first_line_number: int,
    last_line_number: int,
    ctx: Context,
) -> str:
    """Edit a selected section of an article based on human feedback and expected requirements
    using Brown's selected text editing workflow.

    This tool orchestrates the complete selected text editing process by leveraging
    the Brown MCP Server workflow. It processes the selected text, human feedback, and
    article context to provide precise content modifications.

    Args:
        article_path: Path to the article containing the selected text that has to be edited.
        human_feedback: User's feedback or instructions for the edit.
        selected_text: The specific text selected from the article that needs to be edited, as
            a one-on-one copy of the specific text that needs to be edited.
        first_line_number: Line number from the original file where the selected text starts.
        last_line_number: Line number from the original file where the selected text ends.
        ctx: MCP context for streaming progress updates and user communication.

    Returns:
        The fully edited selected text in markdown format plus instructions on what to do with
        the edited selected text.

    Example:
        >>> result = await edit_selected_text(
        ...     "/path/to/article/resources",
        ...     "Make this more concise",
        ...     "This is a very long sentence that could be shorter.",
        ...     11,
        ...     11,
        ...     ctx
        ... )
        >>> print(result)
        Here is the edited selected text:
        <selected_text>

        Here is what you have to do with the edited selected text:
        <instructions>
    """

    async with build_short_term_memory(app_config) as checkpointer:
        edit_selected_text_workflow = build_edit_selected_text_workflow(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        tracer = tracing.build_handler(thread_id, tags=["edit_selected_text"])
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "callbacks": [tracer]}

        dir_path = Path(article_path).parent
        await ctx.info(f"Editing selected text from file {article_path}")
        await ctx.info(f"Using directory `{dir_path}` as context")

        final_result = None
        async for chunk in edit_selected_text_workflow.astream(
            {
                "dir_path": Path(dir_path),
                "human_feedback": human_feedback,
                "selected_text": selected_text,
                "number_line_before_selected_text": first_line_number,
                "number_line_after_selected_text": last_line_number,
            },
            config=config,
            stream_mode=["custom", "values"],
        ):
            chunk_type, chunk_data = chunk
            await parse_message(chunk_data, ctx)

            if chunk_type == "values":
                final_result = chunk_data

    return final_result or "Selected text editing completed successfully!"


@mcp.resource("resource://config/app", mime_type="application/json")
def get_app_config() -> dict:
    """Get the application configuration for Brown Agent as an MCP resource.

    This resource provides access to the complete Brown Agent configuration,
    including model settings, file paths, and workflow parameters. The configuration
    is loaded from YAML files and converted to a JSON-serializable format.

    Returns:
        dict: The application configuration as a JSON-serializable dictionary containing:
            - Model configurations for each workflow node (write_article, review_article, etc.)
            - File paths for profiles, examples, and context files
            - Number of review iterations and workflow settings
            - Temperature and other model parameters
            - Tool configurations and model assignments
    """

    return app_config.model_dump(mode="json")


@mcp.resource("resource://profiles/character")
def get_character_profile() -> str:
    """Get the character profile resource for Brown Agent as an MCP resource.

    This resource provides access to the character-specific writing persona
    that the Brown Agent uses for consistent article generation and editing.
    The character profile defines the voice, perspective, and personal style
    that should be reflected in the generated content.

    The profile is loaded using the same builders pattern as the workflows,
    ensuring consistency across the Brown Agent system.

    Returns:
        str: The character profile content in markdown format, loaded from
             the configured character profile file.

    Raises:
        ValueError: If the character profile cannot be loaded or is not found.

    Example:
        The character profile typically includes information about:
        - Writing voice and tone preferences
        - Personal background and expertise areas
        - Preferred terminology and expressions
        - Communication style and approach
    """
    return __get_profile("character")


def __get_profile(profile_name: str) -> str:
    """Internal helper function to load and return a specific article profile.

    This private function handles the actual loading of profile content from the
    filesystem using the Brown Agent's profile loading infrastructure. It uses
    the same builders pattern as the workflows to ensure consistency.

    Available profile types:
    - article: Core article writing guidelines and style preferences
    - structure: Article structure and organization guidelines
    - mechanics: Writing mechanics, grammar, and formatting rules
    - terminology: Domain-specific terminology and vocabulary preferences
    - tonality: Tone, voice, and style guidelines
    - character: Character-specific writing persona and voice

    Args:
        profile_name: The profile type to retrieve. Must be one of the supported
                     profile types (article, character, mechanics, structure, terminology, tonality).

    Returns:
        str: The requested profile content in markdown format.

    Raises:
        ValueError: If the profile_name is not supported or the profile cannot be loaded.

    Note:
        This is a private function intended for internal use by MCP resource
        functions. It uses the builders pattern to load profiles consistently
        with the workflows.

    Example:
        >>> __get_profile("article")
        "# Article Profile\n\nThis profile defines..."

        >>> __get_profile("character")
        "# Character Profile\n\nPaul Iusztin is..."
    """

    loaders = build_loaders(app_config)
    profiles_loader = loaders["profiles"]
    profiles = profiles_loader.load()

    try:
        profile_obj = getattr(profiles, profile_name)
        return profile_obj.content
    except Exception:
        raise ValueError(f"Unsupported profile type '{profile_name}'. Must be one of: {', '.join(profiles.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brown MCP Server")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (stdio or streamable-http)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8002,
        help="Port number for HTTP transport (default: 8002)",
    )
    args = parser.parse_args()
    
    # Run the server with the specified transport
    if args.transport == "streamable-http":
        mcp.run(transport=args.transport, port=args.port)
    else:
        mcp.run(transport=args.transport)
