"""Brown MCP CLI - Command-line interface for Brown article workflows.

This script provides a CLI interface to Brown's article generation and editing
capabilities through the MCP (Model Context Protocol) server. It supports three
main operations: generating articles, editing entire articles, and editing
selected text sections.

Usage:
    python scripts/brown_mcp_cli.py generate-article --dir-path /path/to/article
    python scripts/brown_mcp_cli.py edit-article --dir-path /path/to/article --human-feedback "Improve the introduction"
    python scripts/brown_mcp_cli.py edit-text --dir-path /path/to/article --human-feedback "Make this clearer"
--first-line 10 --last-line 20
"""

import asyncio
import re
from functools import wraps
from pathlib import Path

import click
from fastmcp import Client
from loguru import logger

from brown.mcp.server import mcp

# Create client using in-memory transport with the Brown MCP Server
client = Client(mcp)


def async_command(f):
    """Decorator to run an async click command."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
def cli():
    """Brown MCP CLI - Command-line interface for Brown article workflows."""
    pass


@cli.command()
@click.option(
    "--dir-path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the directory containing article resources"
)
@async_command
async def generate_article(dir_path: Path) -> None:
    """Generate an article from scratch using Brown's article generation workflow.

    This command processes research data, guidelines, and other resources from
    the specified directory to create a comprehensive article.
    """
    logger.info(f"Starting article generation for directory: {dir_path}")

    try:
        async with client:
            result = await client.call_tool("generate_article", {"dir_path": str(dir_path)})
            logger.success("Article generation completed successfully")
            logger.info("Generated article content:")
            if hasattr(result, "content") and isinstance(result.content, list):
                # Handle list of TextContent objects
                result_content = "".join(item.text for item in result.content if hasattr(item, "text"))
            else:
                result_content = result.content if hasattr(result, "content") else str(result)
            print(result_content)
    except Exception as e:
        logger.error(f"Article generation failed: {e}")
        raise click.ClickException(f"Article generation failed: {e}")


@cli.command()
@click.option(
    "--dir-path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the directory containing article resources"
)
@click.option("--human-feedback", type=str, required=True, help="Human feedback for editing the article")
@async_command
async def edit_article(dir_path: Path, human_feedback: str) -> None:
    """Edit an entire article based on human feedback.

    This command processes human feedback and article context to provide
    comprehensive content modifications at the article level.
    """
    logger.info(f"Starting article editing for directory: {dir_path}")
    logger.info(f"Human feedback: {human_feedback}")

    try:
        async with client:
            result = await client.call_tool(
                "edit_article",
                {
                    "dir_path": str(dir_path),
                    "human_feedback": human_feedback,
                },
            )

            # Parse the result to extract the edited article content
            if hasattr(result, "content") and isinstance(result.content, list):
                # Handle list of TextContent objects
                result_content = "".join(item.text for item in result.content if hasattr(item, "text"))
            else:
                result_content = result.content if hasattr(result, "content") else str(result)

            article_content = _parse_edit_article_result(result_content)

            # Save the edited article to the directory
            article_path = dir_path / "article.md"
            article_path.write_text(article_content, encoding="utf-8")

            logger.success("Article editing completed successfully")
            logger.info(f"Edited article saved to: {article_path}")
            print(f"Edited article saved to: {article_path}")

    except Exception as e:
        logger.error(f"Article editing failed: {e}")
        raise click.ClickException(f"Article editing failed: {e}")


@cli.command()
@click.option(
    "--dir-path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the directory containing article resources"
)
@click.option("--human-feedback", type=str, required=True, help="Human feedback for editing the selected text")
@click.option("--first-line", type=int, required=True, help="First line number of the selected text")
@click.option("--last-line", type=int, required=True, help="Last line number of the selected text")
@async_command
async def edit_selected_text(dir_path: Path, human_feedback: str, first_line: int, last_line: int) -> None:
    """Edit a selected section of an article based on user feedback.

    This command processes the selected text, user feedback, and article context
    to provide precise content modifications.
    """
    logger.info(f"Starting selected text editing for directory: {dir_path}")
    logger.info(f"Human feedback: {human_feedback}")
    logger.info(f"Selected lines: {first_line}-{last_line}")

    try:
        # Read the current article to extract selected text
        article_path = dir_path / "article.md"
        if not article_path.exists():
            raise click.ClickException(f"Article file not found: {article_path}")

        article_lines = article_path.read_text(encoding="utf-8").splitlines()
        selected_text = "\n".join(article_lines[first_line - 1 : last_line])

        logger.info(f"Extracted selected text ({len(selected_text)} characters)")

        async with client:
            result = await client.call_tool(
                "edit_selected_text",
                {
                    "dir_path": str(dir_path),
                    "human_feedback": human_feedback,
                    "selected_text": selected_text,
                    "first_line_number": first_line,
                    "last_line_number": last_line,
                },
            )

            # Parse the result to extract the edited text
            if hasattr(result, "content") and isinstance(result.content, list):
                # Handle list of TextContent objects
                result_content = "".join(item.text for item in result.content if hasattr(item, "text"))
            else:
                result_content = result.content if hasattr(result, "content") else str(result)
            edited_content, new_first_line, new_last_line = _parse_edit_text_result(result_content)

            # Apply changes back to the article
            updated_article_lines = article_lines.copy()
            updated_article_lines[new_first_line - 1 : new_last_line] = edited_content.splitlines()

            # Save the updated article
            updated_content = "\n".join(updated_article_lines)
            article_path.write_text(updated_content, encoding="utf-8")

            logger.success("Selected text editing completed successfully")
            logger.info(f"Updated article saved to: {article_path}")
            print(f"Updated article saved to: {article_path}")

    except Exception as e:
        logger.error(f"Selected text editing failed: {e}")
        raise click.ClickException(f"Selected text editing failed: {e}")


def _parse_edit_article_result(result: str) -> str:
    """Parse the edit article result to extract the article content.

    Args:
        result: The result string from the MCP tool

    Returns:
        The extracted article content
    """
    # Look for the article content after "Here is the edited article:"
    pattern = r"Here is the edited article:\s*\n(.*?)(?:\n\nHere is what you have to do|$)"
    match = re.search(pattern, result, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # Fallback: try to find content between markdown sections
        lines = result.split("\n")
        in_content = False
        content_lines = []

        for line in lines:
            if "Here is the edited article:" in line:
                in_content = True
                continue
            elif in_content and ("Here is what you have to do" in line or line.strip() == ""):
                break
            elif in_content:
                content_lines.append(line)

        if content_lines:
            return "\n".join(content_lines).strip()
        else:
            # If no pattern matches, return the full result
            logger.warning("Could not parse article content from result, using full result")
            return result


def _parse_edit_text_result(result: str) -> tuple[str, int, int]:
    """Parse the edit text result to extract the edited content and line numbers.

    Args:
        result: The result string from the MCP tool

    Returns:
        Tuple of (edited_content, first_line_number, last_line_number)
    """
    # Look for XML-like structure in the result
    content_pattern = r"<content>(.*?)</content>"
    first_line_pattern = r"<first_line_number>(\d+)</first_line_number>"
    last_line_pattern = r"<last_line_number>(\d+)</last_line_number>"

    content_match = re.search(content_pattern, result, re.DOTALL)
    first_line_match = re.search(first_line_pattern, result)
    last_line_match = re.search(last_line_pattern, result)

    if content_match and first_line_match and last_line_match:
        edited_content = content_match.group(1).strip()
        first_line = int(first_line_match.group(1))
        last_line = int(last_line_match.group(1))
        return edited_content, first_line, last_line
    else:
        # Fallback: try to extract from "Here is the edited selected text:"
        pattern = r"Here is the edited selected text:\s*\n(.*?)(?:\n\nHere is what you have to do|$)"
        match = re.search(pattern, result, re.DOTALL)

        if match:
            content = match.group(1).strip()
            # Use original line numbers as fallback
            return content, 1, 1
        else:
            logger.warning("Could not parse edited text from result, using full result")
            return result, 1, 1


if __name__ == "__main__":
    cli()
