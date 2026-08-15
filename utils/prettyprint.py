"""A module for pretty-printing (both data structure, and terminal coloring).

@author Wouter Vermeulen
@date 2026-01-18
"""

from colorama import Fore, Style
from sys import __stdout__


COLORS = {
    "green": Fore.GREEN,
    "red": Fore.RED,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
}


def color_text(text: str, color: str = "green") -> str:
    """Returns the input text wrapped in ANSI escape codes for colored terminal output.

    :param text: The text to color.
    :param color: The color name. Supported colors are listed in the ``COLORS`` dictionary.

    :return: The colored text with ANSI escape codes.
    """
    if color not in COLORS:
        raise ValueError(f"Color '{color}' not supported. Supported colors: {list(COLORS.keys())}")

    return f"{COLORS[color]}{text}{Style.RESET_ALL}"


def tprint(*values, sep: str | None = " ", end: str | None = "\n", flush: bool = False, color: str | None = None):
    """Print function that outputs to terminal, even with abaqus cae nogui."""
    try:
        if color is not None:
            print(color_text(sep.join(values), color=color), end=end, flush=flush, file=__stdout__)
        else:
            print(*values, sep=sep, end=end, flush=flush, file=__stdout__)
    except ImportError:
        print(*values, sep=sep, end=end, flush=flush)
