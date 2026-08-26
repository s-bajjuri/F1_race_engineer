from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

BACKGROUND = "#050607"
PANEL = "#0D1014"
PANEL_ALT = "#12171D"
GRID = "#2A333D"
TEXT = "#F4F7FA"
MUTED = "#9AA7B4"
ACCENT = "#4CA8FF"
CYAN = "#36D6E7"
GREEN = "#35D07F"
RED = "#FF5664"
AMBER = "#FFB347"
PURPLE = "#B68CFF"
WHITE = "#FFFFFF"


def style_figure(fig: Figure) -> None:
    fig.patch.set_facecolor(BACKGROUND)


def style_axis(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title is not None:
        ax.set_title(
            title,
            loc="left",
            fontsize=11,
            fontweight="medium",
            color=TEXT,
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, color=GRID, alpha=0.58, linewidth=0.7)


def style_legend(ax: Axes, *, columns: int = 1) -> None:
    legend = ax.legend(
        loc="upper left",
        ncols=columns,
        frameon=False,
        fontsize=8.5,
    )
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(MUTED)
