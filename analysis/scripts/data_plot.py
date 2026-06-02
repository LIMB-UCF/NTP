"""
Taxel-voltage composite plotting for processed texture data.

Creates a 2x2 figure showing nine taxel voltage traces for one
texture and trial number at two speeds and two forces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import default_processed_root


FILE_PATTERN = "Data[Texture({texture})][Speed({speed})][Force({force})][Trial({trial})].csv"

SPEED_WINDOWS = {
    1200: (18.3, 20.9),
    2400: (17.7, 19.0),
    3600: (17.5, 18.5),
    4800: (17.4, 18.2),
    6000: (17.4, 18.0),
}

WINDOW_DURATIONS = {
    speed: end - start
    for speed, (start, end) in SPEED_WINDOWS.items()
}

TAXEL_COLORS = [
    "#EF4444", "#F97316", "#F59E0B",
    "#84CC16", "#22C55E", "#14B8A6",
    "#06B6D4", "#3B82F6", "#A855F7",
]

FIGURE_SIZE = (3.8, 3.2)
COLUMN_GAP_SPACE = 0.1
ROW_GAP_SPACE = 0.06

GLOBAL_LABEL_FONT_SIZE = 9
TICK_LABEL_SIZE = 7
LEGEND_FONT_SIZE = 7
SUBPLOT_LABEL_SIZE = 8
PLACEHOLDER_FONT_SIZE = 8

TAXEL_LINEWIDTH = 1.0
TAXEL_ALPHA = 0.9

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

DEFAULT_TEXTURE = "002"
DEFAULT_TRIAL = "69"
DEFAULT_FORCES = ["0.0", "1.5"]
DEFAULT_SPEEDS = ["1200", "2400"]


def default_output_dir() -> Path:
    """Return the default output directory for this plot."""

    return Path(__file__).resolve().parent / "data_plot"


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a processed taxel CSV file."""

    return pd.read_csv(path, header=None)


def add_panel_label(ax, label: str) -> None:
    """Add a panel label to an axis."""

    ax.text(
        0.98,
        0.98,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=SUBPLOT_LABEL_SIZE,
        zorder=10,
    )


def plot_single_panel(
    ax,
    df: pd.DataFrame,
    *,
    is_left_col: bool,
    is_bottom_row: bool,
    t_max_ms: int,
) -> None:
    """Plot taxel voltage traces in a single axis."""

    time_seconds = df.iloc[:, 10].values
    time_ms = (time_seconds - time_seconds[0]) * 1000.0
    taxel_data = df.iloc[:, :9].values

    for taxel_index in range(9):
        ax.plot(
            time_ms,
            taxel_data[:, taxel_index],
            color=TAXEL_COLORS[taxel_index],
            linewidth=TAXEL_LINEWIDTH,
            alpha=TAXEL_ALPHA,
        )

    ax.set_xlim(0, t_max_ms)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    if is_bottom_row:
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    if not is_left_col:
        ax.set_yticklabels([])

    ax.grid(False)


def axes_block_bbox(fig, axes_list) -> tuple[float, float, float, float]:
    """Return the figure-coordinate bounding box for a group of axes."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in axes_list
    ]

    return (
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def create_composite_plot(
    *,
    texture: str,
    trial: str,
    speeds: list[str],
    forces: list[str],
    data_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Create and save the taxel-voltage composite figure."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    speeds_i = [int(speed) for speed in speeds]
    left_duration = WINDOW_DURATIONS[speeds_i[0]]
    right_duration = WINDOW_DURATIONS[speeds_i[1]]

    fig = plt.figure(figsize=FIGURE_SIZE)

    outer = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.0, 0.22],
        wspace=0.02,
    )

    plot_area = outer[0].subgridspec(
        2,
        2,
        width_ratios=[left_duration, right_duration],
        wspace=COLUMN_GAP_SPACE,
        hspace=ROW_GAP_SPACE,
    )

    ax_legend = fig.add_subplot(outer[1])
    ax_legend.axis("off")

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]
    plot_axes = []

    for row, force in enumerate(forces):
        for col, speed in enumerate(speeds):
            speed_i = int(speed)
            t_max_ms = int(WINDOW_DURATIONS[speed_i] * 1000.0)

            filename = FILE_PATTERN.format(
                texture=texture,
                speed=speed,
                force=force,
                trial=trial,
            )
            path = data_dir / f"Texture{texture}" / filename

            ax = fig.add_subplot(plot_area[row, col])
            plot_axes.append(ax)
            add_panel_label(ax, panel_labels[row][col])

            is_bottom = row == len(forces) - 1
            is_left = col == 0

            if not path.is_file():
                ax.text(
                    0.5,
                    0.5,
                    "Data Not Found",
                    ha="center",
                    va="center",
                    fontsize=PLACEHOLDER_FONT_SIZE,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_single_panel(
                ax,
                read_csv(path),
                is_left_col=is_left,
                is_bottom_row=is_bottom,
                t_max_ms=t_max_ms,
            )

    handles = [
        Line2D([0], [0], color=TAXEL_COLORS[taxel_index], lw=2)
        for taxel_index in range(9)
    ]
    labels = [f"T{taxel_index + 1}" for taxel_index in range(9)]

    ax_legend.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
    )

    fig.subplots_adjust(bottom=0.18, left=0.12, right=0.98, top=0.98)

    left, bottom, right, top = axes_block_bbox(fig, plot_axes)
    x_center = (left + right) / 2.0
    y_center = (bottom + top) / 2.0

    fig.text(
        x_center,
        0.10,
        "Time (ms)",
        ha="center",
        va="top",
        fontsize=GLOBAL_LABEL_FONT_SIZE,
    )
    fig.text(
        left - 0.04,
        y_center,
        "Taxel Value (V)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=GLOBAL_LABEL_FONT_SIZE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_plot.pdf"

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Parse arguments and save the taxel-voltage composite figure."""

    parser = argparse.ArgumentParser(description="Create a taxel-voltage composite plot.")
    parser.add_argument("--texture", default=DEFAULT_TEXTURE)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--speeds", nargs=2, default=DEFAULT_SPEEDS)
    parser.add_argument("--forces", nargs=2, default=DEFAULT_FORCES)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    create_composite_plot(
        texture=args.texture,
        trial=args.trial,
        speeds=args.speeds,
        forces=args.forces,
        data_dir=default_processed_root(__file__) if args.data_dir is None else args.data_dir,
        output_dir=default_output_dir() if args.output_dir is None else args.output_dir,
    )


if __name__ == "__main__":
    main()