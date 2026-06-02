"""
Speed-force accuracy heatmap plotting for the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import FORCES, SPEEDS_MM_MIN, default_spike_root, default_split_path, load_test_dataset
from inference import (
    MODEL_TYPES,
    WINDOW_SIZE,
    STRIDE,
    NUM_STEPS,
    find_latest_weights_by_model,
    load_models,
    predict_all_models,
)


FIGURE_SIZE = (4.8, 2.6)

GLOBAL_LABEL_FONT_SIZE = 9
TICK_LABEL_SIZE = 7
CELL_FONT_SIZE = 6
TITLE_FONT_SIZE = 9
SUBPLOT_TITLE_FONT_SIZE = 9

GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

HEATMAP_VMIN = 0.90
HEATMAP_VMAX = 1.00

SPEED_LABELS_MM_PER_S = [str(int(speed / 60)) for speed in SPEEDS_MM_MIN]


def accuracy_grid(dataset, result) -> np.ndarray:
    """Return a force-by-speed accuracy grid for one model result."""

    correct = np.zeros((len(FORCES), len(SPEEDS_MM_MIN)), dtype=np.float64)
    total = np.zeros((len(FORCES), len(SPEEDS_MM_MIN)), dtype=np.float64)

    force_to_row = {force: i for i, force in enumerate(FORCES)}
    speed_to_col = {speed: i for i, speed in enumerate(SPEEDS_MM_MIN)}

    for local_i, sample_index in enumerate(result.kept_indices):
        sample = dataset.samples[int(sample_index)]

        row = force_to_row[float(sample.force)]
        col = speed_to_col[int(sample.speed_mm_min)]

        total[row, col] += 1.0
        correct[row, col] += float(result.y_true[local_i] == result.y_pred[local_i])

    grid = np.full_like(correct, np.nan, dtype=np.float64)
    np.divide(correct, total, out=grid, where=total > 0)

    return grid


def plot_heatmaps(
    grids_by_model: dict[str, np.ndarray],
    output_path: Path,
    main_title: str = "",
) -> None:
    """Save a 2×2 speed-force heatmap figure."""

    fig = plt.figure(figsize=FIGURE_SIZE)

    if main_title:
        fig.suptitle(main_title, fontsize=TITLE_FONT_SIZE, y=0.98)

    fig.supxlabel("Speed (mm/s)", fontsize=GLOBAL_LABEL_FONT_SIZE, y=0.02)
    fig.supylabel("Force (Δmm)", fontsize=GLOBAL_LABEL_FONT_SIZE, x=0.02, rotation=90)

    gs = gridspec.GridSpec(2, 2, wspace=0.08, hspace=0.18)

    panel_order = ["SNN", "tSNN", "FS-SNN", "FS-tSNN"]
    image = None

    for index, model_type in enumerate(panel_order):
        row = index // 2
        col = index % 2
        ax = fig.add_subplot(gs[row, col])

        grid = grids_by_model[model_type]

        image = ax.imshow(
            grid,
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
        )

        ax.set_title(model_type, fontsize=SUBPLOT_TITLE_FONT_SIZE, pad=0, y=1.02)

        ax.set_xticks(np.arange(len(SPEEDS_MM_MIN)))
        ax.set_yticks(np.arange(len(FORCES)))

        if row == 1:
            ax.set_xticklabels(SPEED_LABELS_MM_PER_S, fontsize=TICK_LABEL_SIZE)
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_yticklabels([f"{force:.1f}" for force in FORCES], fontsize=TICK_LABEL_SIZE)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=0)

        ax.set_xticks(np.arange(-0.5, len(SPEEDS_MM_MIN), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FORCES), 1), minor=True)
        ax.grid(which="minor", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                value = grid[y, x]
                text = "—" if np.isnan(value) else f"{value:.4g}"
                ax.text(x, y, text, ha="center", va="center", fontsize=CELL_FONT_SIZE)

    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cb_ticks = np.arange(HEATMAP_VMIN, HEATMAP_VMAX + 0.001, 0.01)
    colorbar = fig.colorbar(image, cax=cax, ticks=cb_ticks)
    colorbar.ax.tick_params(labelsize=CELL_FONT_SIZE)
    colorbar.set_label("Accuracy", fontsize=GLOBAL_LABEL_FONT_SIZE)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Run test-set inference and save speed-force heatmaps."""

    parser = argparse.ArgumentParser(description="Plot test-set speed-force heatmaps.")
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    classifiers_dir = project_root / "analysis" / "classifiers"

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)
    split_npz = Path(args.split).resolve() if args.split else default_split_path(__file__)

    output_dir = script_dir / "heatmaps"
    output_path = output_dir / "heatmaps.pdf"
    cache_path = make_cache_path(output_dir, "heatmaps_cache.npz")

    dataset = load_test_dataset(split_npz, spike_root)
    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)

    params = common_cache_params(
        plot_name="heatmaps",
        dataset_name=dataset.name,
        spike_root=spike_root,
        model_weights=weights,
        split_npz=split_npz,
        extra={
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "num_steps": NUM_STEPS,
            "fs_policy": "skip",
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is None:
        models = load_models(weights)
        results = predict_all_models(models, dataset, weights_by_model=weights, device=args.device)
        grids_by_model = {
            model_type: accuracy_grid(dataset, results[model_type])
            for model_type in MODEL_TYPES
        }

        save_cache(
            cache_path,
            params=params,
            **{f"grid_{model_type}": grids_by_model[model_type] for model_type in MODEL_TYPES},
        )
    else:
        grids_by_model = {
            model_type: np.asarray(cache[f"grid_{model_type}"], dtype=np.float64)
            for model_type in MODEL_TYPES
        }
        print(f"Loaded cache: {cache_path}")

    plot_heatmaps(grids_by_model, output_path, main_title=args.title)


if __name__ == "__main__":
    main()