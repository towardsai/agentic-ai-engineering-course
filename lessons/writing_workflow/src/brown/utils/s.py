import re
from typing import Any


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def normalize_any_to_str(output: Any) -> str:
    """Normalize LLM output to a string, handling str, list, dict, and fallback cases.

    Args:
        output: The raw output from the LLM (could be str, list, dict, etc.)

    Returns:
        str: The normalized string output.
    """

    if isinstance(output, str):
        return output

    if isinstance(output, list) and len(output) > 0:
        first_item = output[0]
        if isinstance(first_item, str):
            return "\n".join(output)
        else:
            raise ValueError(f"Expected a list of strings, got a list of `{type(first_item)}`")

    if isinstance(output, dict):
        # Try to extract a likely text field, fallback to str
        for key in ("content",):
            if key in output and isinstance(output[key], str):
                return output[key]

        raise ValueError(f"Expected a dict with a `content` key, got a dict of `{output.keys()}`")

    raise ValueError(f"Expected a str, list or dict, got a `{type(output)}`")


def clean_markdown_links(text: str) -> str:
    """
    Cleans markdown style links from a string.

    This function finds all markdown links, including image links, and replaces
    them with just the URL part of the link. It also handles cases of
    malformed links as described.

    Args:
        text: The input string containing markdown links.

    Returns:
        The cleaned string with markdown links replaced by their URLs.
    """

    # Regex for standard markdown links ![text](url) or [text](url)
    # It captures the URL part.
    pattern = r"!?!?\[([^\]]*)\]\(([^)]+)\)"

    # Replace standard markdown links with the URL
    cleaned_text = re.sub(pattern, r" \2 ", text)

    return cleaned_text
