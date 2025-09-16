"""Print utilities for colored terminal output."""

from enum import Enum
from typing import Optional


class Color(Enum):
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class Style(Enum):
    """Text style codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"


def print_colored(text: str, color: Color, style: Optional[Style] = None, end: str = "\n") -> None:
    """Print text with color and optional style.

    Args:
        text: The text to print
        color: The color to apply
        style: Optional text style to apply
        end: String to append after the text (default: newline)
    """
    if style:
        print(f"{style.value}{color.value}{text}{Color.RESET.value}", end=end)
    else:
        print(f"{color.value}{text}{Color.RESET.value}", end=end)


def print_header(
    text: str, delimiter_char: str = "=", color: Color = Color.BRIGHT_CYAN, style: Optional[Style] = Style.BOLD
) -> None:
    """Print a header with delimiters.

    Args:
        text: The header text
        delimiter_char: Character to use for delimiters
        color: Color for the header text
        style: Style for the header text
    """
    delimiter = delimiter_char * 60
    print_colored(delimiter, Color.BRIGHT_WHITE)
    print_colored(text, color, style)
    print_colored(delimiter, Color.BRIGHT_WHITE)
    print()


def print_item(
    name: str,
    description: str,
    index: Optional[int] = None,
    name_color: Color = Color.BRIGHT_WHITE,
    desc_color: Color = Color.YELLOW,
) -> None:
    """Print an item with name and description.

    Args:
        name: The item name
        description: The item description
        index: Optional index number for the item
        name_color: Color for the item name
        desc_color: Color for the item description
    """
    if index is not None:
        print_colored(f"{index}. ", Color.BRIGHT_GREEN, end="")
    print_colored(name, name_color, end="")
    print_colored(f"\n   {description}", desc_color)
    print()
