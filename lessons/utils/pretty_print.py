import json
from typing import Iterable

from google.genai import types


class Color:
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"


def wrapped(text: Iterable[str] | dict | str, title: str = "", indent: int = 2, width: int = 100) -> None:
    header, separator = __get_header_and_footer(title, width)

    if isinstance(text, str):
        text = [text]
    elif isinstance(text, dict):
        text = [json.dumps(text, indent=indent)]

    print(header)
    for line in text:
        print(f"{' ' * indent}{line}")
        print(separator)


def function_call(
    function_call: types.FunctionCall, only_name: bool = False, title: str = "", indent: int = 2, width: int = 100
) -> None:
    header, footer = __get_header_and_footer(title, width)

    print(header)
    print(f"{' ' * indent}{Color.ORANGE}Function Name:{Color.RESET} `{function_call.name}")
    if only_name is False:
        print(
            f"{' ' * indent}{Color.ORANGE}Function Arguments:{Color.RESET} `{json.dumps(function_call.args, indent=indent)}`"
        )
    print(footer)


def __get_header_and_footer(title: str, width: int) -> tuple[str, str]:
    header = (
        f"{Color.YELLOW}{'-' * ((width - len(title)) // 2 - 1)} {title} {'-' * ((width - len(title)) // 2 - 1)}{Color.RESET}"  # noqa: E501
        if title
        else f"{Color.YELLOW}{'-' * width}{Color.RESET}"  # noqa: E501
    )
    footer = f"{Color.YELLOW}{'-' * width}{Color.RESET}"

    return header, footer
