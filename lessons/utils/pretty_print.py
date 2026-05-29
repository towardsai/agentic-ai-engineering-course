import json
from typing import Iterable

from google.genai import types


class Color:
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    PURPLE = "\033[35m"
    TEAL = "\033[36m"
    BROWN = "\033[38;5;94m"
    RESET = "\033[0m"


def wrapped(
    text: Iterable[str] | dict | str,
    title: str = "",
    indent: int = 2,
    width: int = 100,
    header_color: str = Color.YELLOW,
) -> None:
    header, separator = __get_header_and_footer(title, width, header_color)

    if isinstance(text, str):
        text = [text]
    elif isinstance(text, dict):
        text = [json.dumps(text, indent=indent)]

    print(header)
    for line in text:
        print(f"{' ' * indent}{line}")
        print(separator)


def function_call(
    function_call: types.FunctionCall | None,
    only_name: bool = False,
    title: str = "",
    indent: int = 2,
    width: int = 100,
    header_color: str = Color.YELLOW,
    label_color: str = Color.ORANGE,
) -> None:
    header, footer = __get_header_and_footer(title, width, header_color)

    print(header)
    if function_call:
        print(f"{' ' * indent}{label_color}Function Name:{Color.RESET} `{function_call.name}")
        if only_name is False:
            print(
                f"{' ' * indent}{label_color}Function Arguments:{Color.RESET} `{json.dumps(function_call.args, indent=indent)}`"  # noqa: E501
            )
    else:
        print(f"{' ' * indent}{label_color}Function Call is missing{Color.RESET}")
    print(footer)


def __get_header_and_footer(title: str, width: int, color: str = Color.YELLOW) -> tuple[str, str]:
    header = (
        f"{color}{'-' * ((width - len(title)) // 2 - 1)} {title} {'-' * ((width - len(title)) // 2 - 1)}{Color.RESET}"  # noqa: E501
        if title
        else f"{color}{'-' * width}{Color.RESET}"  # noqa: E501
    )
    footer = f"{color}{'-' * width}{Color.RESET}"

    return header, footer
