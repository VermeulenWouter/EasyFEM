"""Matplotlib styling configuration."""

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.prettyprint import color_text, tprint


# Check palettes with https://projects.susielu.com/viz-palette to ensure good visibility!
COLOR_CYCLE = ["#92bcb5", "#584982", "#b29ff4", "#9f0b64", "#d45fea", "#85dc4d"]


class FigSizes(Enum):
    """Figure size presets for matplotlib figures."""
    DEFAULT = {
        "figure.figsize": (5, 3.5),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6
    }
    HALF_WIDTH = {
        "figure.figsize": (3.075, 2.75),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6
    }
    DOUBLE = {
        "figure.figsize": (6.5, 3.5),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6
    }


class FigDestinations(Enum):
    """Intended destinations for figures."""
    LATEX = {
        "style": {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{libertine}"
        },
        "output": {
            "filetype": "pdf"
        }
    }
    """For LaTeX, use a LaTeX font and output figure to PDF"""

    POWERPOINT = {
        "style": {
            "font.family":  "DejaVu Sans"
        },
        "output": {
            "filetype": "png"
        }
    }
    """For PowerPoint, use a Microsoft font and output figure to PNG"""


REPORT = FigDestinations.LATEX
"""Alias for report figures"""
PRESENTATION = FigDestinations.POWERPOINT
"""Alias for presentation figures"""


def _preconfigure_matplotlib() -> None:
    """Apply consistent global matplotlib settings."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE, linestyle=["-", "--", "-.", ":", "-", "--"]),
        "axes.grid": True
    })


class Figure(plt.Figure):
    """Custom Figure class with preset size and destination settings."""
    def __init__(self, figsize: dict, destination: FigDestinations = FigDestinations.LATEX,
                 *args, **kwargs) -> None:
        """Create a matplotlib figure with preset size and destination settings.

        :param figsize: Preset figure size from FigSizes enum
        :param destination: Intended destination for the figure from FigDestinations enum

        :return: Matplotlib Figure object
        """
        mpl.rcParams.update(figsize)  # Size & textsize
        mpl.rcParams.update(destination.value["style"])  # Font

        if kwargs.get("tight_layout", None) is None and kwargs.get("constrained_layout", None) is None:
            kwargs["constrained_layout"] = True

        super().__init__(*args, **kwargs)
        self.__destination = destination

    def savefig(self, fname: Path, *args, legend_loc: str = "best", no_grid: bool = False, no_legend: bool = False, legend_kwargs=None, **kwargs) -> None:
        """Save the figure with consistent settings based on destination."""
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)

        dpi = kwargs.pop("dpi", 300)
        ftype = self.__destination.value["output"]["filetype"]

        if len(self.axes) == 1:
            if not no_grid:
                plt.grid(True)
            if not no_legend:
                if legend_kwargs:
                    plt.legend(**legend_kwargs)
                else:
                    plt.legend(loc=legend_loc)
        else:
            if not no_grid:
                plt.grid(True)
            if not no_legend:
                for ax in self.axes:
                    if legend_kwargs:
                        ax.legend(**legend_kwargs)
                    else:
                        ax.legend(loc=legend_loc)

        super().savefig(f"{fname}.{ftype}", *args, dpi=dpi, **kwargs)
        plt.close()

        tprint(color_text(f"Figure saved to {fname}{ftype}"))

    def double_axis(self, sharex: bool = True, **kwargs) -> tuple:
        """Create a secondary y-axis sharing the same x-axis.

        :param sharex: Whether the secondary axis should share the x-axis with the primary axis

        :return: Tuple of (primary_axis, secondary_axis)
        """
        ax1 = self.add_subplot(111)
        ax2 = ax1.twinx()  # if sharex else self.add_subplot(111, sharex=ax1)

        return ax1, ax2


def subplots(figsize: FigSizes = FigSizes.DEFAULT, destination: FigDestinations = FigDestinations.LATEX, x_axes: int = 1, figsize_specific = None, *args, **kwargs):
    """Create a figure and subplots with preset settings based on destination."""

    if figsize_specific:
        figsize = figsize.value.copy()
        figsize.update({"figure.figsize": figsize_specific})
    else:
        figsize = figsize.value
    fig, ax = plt.subplots(FigureClass=Figure, figsize=figsize, destination=destination, *args, **kwargs)

    if x_axes == 2:
        ax2 = ax.twinx()
        return fig, ax, ax2

    if x_axes > 2:
        raise ValueError("Only up to 2 x-axes supported")

    return fig, ax
