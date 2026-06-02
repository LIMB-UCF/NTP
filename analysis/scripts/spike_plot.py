"""
Spike-raster composite plotting for spike-train data.

Creates a 2x2 figure showing nine-taxel spike rasters for one
texture and trial number at two speeds and two forces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import default_spike_root


FILE_PATTERN = "SpikeTrains[Texture({texture})][Speed({speed})][Force({force})][Trial({trial})].npz"

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
SUBPLOT_LABEL_SIZE = 8
PLACEHOLDER_FONT_SIZE = 8

SPIKE_LINEWIDTH = 0.8

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

DEFAULT_TEXTURE = "002"
DEFAULT_TRIAL = "69"
DEFAULT_FORCES = ["0.0", "1.5"]
DEFAULT_SPEEDS = ["1200", "2400"]

XLABEL_Y_FIG = 0.10
YLABEL_X_OFF = 0.04


def default_output_dir() -> Path:
    """Return the default output directory for this plot."""

    return Path(__file__).resolve().parent / "spike_plot"


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


def load_spikes(path: str | Path) -> np.ndarray:
    """Load spike data as a binary array with shape [T, 9]."""

    data = np.load(path)
    spikes = data["spikes"][:, :9]

    return (spikes > 0).astype(np.uint8)


def plot_single_spike_panel(
    ax,
    spikes: np.ndarray,
    *,
    is_left_col: bool,
    is_bottom_row: bool,
    speed: str,
) -> None:
    """Plot a nine-taxel spike raster in a single axis."""

    time_bins = spikes.shape[0]
    duration_s = WINDOW_DURATIONS[int(speed)]
    t_max_ms = duration_s * 1000.0
    t_ms = np.linspace(0.0, t_max_ms, time_bins, endpoint=False)

    for taxel_index in range(9):
        spike_times = t_ms[spikes[:, taxel_index].astype(bool)]

        if spike_times.size == 0:
            continue

        y = taxel_index + 1

        ax.vlines(
            spike_times,
            y - 0.35,
            y + 0.35,
            color=TAXEL_COLORS[taxel_index],
            linewidth=SPIKE_LINEWIDTH,
        )

    ax.set_xlim(0, t_max_ms)
    ax.set_ylim(9.5, 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    if is_bottom_row:
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    else:
        ax.set_xticklabels([])

    if is_left_col:
        ax.set_yticks(np.arange(1, 10))
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    else:
        ax.set_yticklabels([])

    ax.grid(False)
    ax.set_facecolor("white")


def create_composite_spike_plot(
    *,
    texture: str,
    trial: str,
    speeds: list[str],
    forces: list[str],
    data_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Create and save the spike-raster composite figure."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    speeds_i = [int(speed) for speed in speeds]
    left_duration = WINDOW_DURATIONS[speeds_i[0]]
    right_duration = WINDOW_DURATIONS[speeds_i[1]]

    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("white")

    grid = gridspec.GridSpec(
        2,
        2,
        width_ratios=[left_duration, right_duration],
        wspace=COLUMN_GAP_SPACE,
        hspace=ROW_GAP_SPACE,
    )

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]
    plot_axes = []

    for row, force in enumerate(forces):
        for col, speed in enumerate(speeds):
            filename = FILE_PATTERN.format(
                texture=texture,
                speed=speed,
                force=force,
                trial=trial,
            )
            path = data_dir / f"Texture{texture}" / filename

            ax = fig.add_subplot(grid[row, col])
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
                ax.set_facecolor("white")
                continue

            plot_single_spike_panel(
                ax,
                load_spikes(path),
                is_left_col=is_left,
                is_bottom_row=is_bottom,
                speed=speed,
            )

    fig.subplots_adjust(bottom=0.18, left=0.12, right=0.98, top=0.98)

    left, bottom, right, top = axes_block_bbox(fig, plot_axes)
    x_center = (left + right) / 2.0
    y_center = (bottom + top) / 2.0

    fig.text(
        x_center,
        XLABEL_Y_FIG,
        "Time (ms)",
        ha="center",
        va="top",
        fontsize=GLOBAL_LABEL_FONT_SIZE,
    )
    fig.text(
        left - YLABEL_X_OFF,
        y_center,
        "Taxel Index",
        ha="center",
        va="center",
        rotation=90,
        fontsize=GLOBAL_LABEL_FONT_SIZE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spike_plot.pdf"

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Parse arguments and save the spike-raster composite figure."""

    parser = argparse.ArgumentParser(description="Create a spike-raster composite plot.")
    parser.add_argument("--texture", default=DEFAULT_TEXTURE)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--forces", nargs=2, default=DEFAULT_FORCES)
    parser.add_argument("--speeds", nargs=2, default=DEFAULT_SPEEDS)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    create_composite_spike_plot(
        texture=args.texture,
        trial=args.trial,
        speeds=args.speeds,
        forces=args.forces,
        data_dir=default_spike_root(__file__) if args.data_dir is None else args.data_dir,
        output_dir=default_output_dir() if args.output_dir is None else args.output_dir,
    )


if __name__ == "__main__":
    main()