"""
Top confusion-pair plotting for the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import (
    CLASS_TO_TEXTURE,
    TEXTURE_LABELS,
    default_spike_root,
    default_split_path,
    load_test_dataset,
    texture_to_group,
)
from inference import (
    MODEL_TYPES,
    WINDOW_SIZE,
    STRIDE,
    NUM_STEPS,
    confusion_matrix_from_result,
    find_latest_weights_by_model,
    load_models,
    predict_all_models,
)


FIGURE_SIZE = (7.6, 6.4)

GLOBAL_LABEL_FONT_SIZE = 11
TICK_LABEL_SIZE = 8
PANEL_LABEL_SIZE = 10
BAR_LABEL_SIZE = 7
COUNT_TEXT_SIZE = 8

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

WSPACE = 0.06
HSPACE = 0.06
SUPLABEL_PAD_Y = 0.035
SUPLABEL_PAD_X = 0.040

YMIN_ERRORS = 8
Y_BUFFER_FRAC = 0.25

GROUP_PALETTES = {
    "Flat": ["#8E44AD"],
    "Ridges": ["#C7E9C0", "#A1D99B", "#74C476", "#31A354", "#006D2C"],
    "Waves": ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"],
    "Bumps": ["#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"],
    "Spikes": ["#FEE391", "#FEC44F", "#FE9929", "#D95F0E", "#993404"],
}

DENSITY_SUFFIX_ORDER = [3, 2, 1, 4, 5]
DENSITY_SUFFIX_TO_RANK = {suffix: i for i, suffix in enumerate(DENSITY_SUFFIX_ORDER)}


def texture_density_rank(texture: str) -> int:
    """Return the density rank used for texture color shading."""

    return DENSITY_SUFFIX_TO_RANK.get(int(texture[-1]), 2)


def texture_to_color(texture: str) -> tuple[float, float, float]:
    """Return the RGB color assigned to a texture."""

    group = texture_to_group(texture)

    if texture == "000":
        return mcolors.to_rgb(GROUP_PALETTES["Flat"][0])

    return mcolors.to_rgb(GROUP_PALETTES[group][texture_density_rank(texture)])


def class_to_texture(class_index: int) -> str:
    """Return the texture label for a class index."""

    return CLASS_TO_TEXTURE[int(class_index)]


def rgb_luminance(rgb: tuple[float, float, float]) -> float:
    """Return relative luminance for an RGB tuple."""

    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ceil_to_nearest_5(value: float) -> int:
    """Round a value up to the nearest multiple of five."""

    return int(np.ceil(value / 5.0) * 5.0)


def add_panel_label(ax, label: str) -> None:
    """Add a panel label inside an axis."""

    ax.text(
        0.98,
        0.98,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        zorder=10,
    )


def top_k_confusion_pairs(cm: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top-k off-diagonal confusion pairs by error count."""

    trues: list[int] = []
    preds: list[int] = []
    errors: list[int] = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue

            error_count = int(cm[i, j])

            if error_count > 0:
                trues.append(i)
                preds.append(j)
                errors.append(error_count)

    if not errors:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty

    order = np.argsort(-np.asarray(errors, dtype=np.int64))[:k]

    return (
        np.asarray([trues[i] for i in order], dtype=np.int64),
        np.asarray([preds[i] for i in order], dtype=np.int64),
        np.asarray([errors[i] for i in order], dtype=np.int64),
    )


def plot_confusion_pairs(
    top_pairs_by_model: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    top_k: int,
) -> None:
    """Save a 2×2 top confusion-pair bar chart."""

    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.supxlabel("Descending Confusion Pair", fontsize=GLOBAL_LABEL_FONT_SIZE, y=SUPLABEL_PAD_Y)
    fig.supylabel("Number of Errors", fontsize=GLOBAL_LABEL_FONT_SIZE, x=SUPLABEL_PAD_X)

    gs = gridspec.GridSpec(2, 2, wspace=WSPACE, hspace=HSPACE)

    panel_grid = [["SNN", "tSNN"], ["FS-SNN", "FS-tSNN"]]
    ylims: list[int] = []

    for row in panel_grid:
        row_max = 0

        for model_type in row:
            errors = top_pairs_by_model[model_type][2]
            if errors.size:
                row_max = max(row_max, int(np.max(errors)))

        base = max(row_max, YMIN_ERRORS)
        ylims.append(max(ceil_to_nearest_5(base * (1.0 + Y_BUFFER_FRAC)), YMIN_ERRORS))

    ylims = [8, 16]

    for r in range(2):
        for c in range(2):
            model_type = panel_grid[r][c]
            ax = fig.add_subplot(gs[r, c])
            add_panel_label(ax, model_type)

            trues, preds, errors = top_pairs_by_model[model_type]

            x = np.arange(1, top_k + 1, dtype=np.int64)
            heights = np.zeros((top_k,), dtype=np.float64)
            true_labels = [""] * top_k
            pred_labels = [""] * top_k

            n_show = int(min(top_k, errors.size))

            for i in range(n_show):
                heights[i] = float(errors[i])
                true_labels[i] = class_to_texture(int(trues[i]))
                pred_labels[i] = class_to_texture(int(preds[i]))

            facecolors = []
            edgecolors = []

            for i in range(top_k):
                if i < n_show:
                    facecolors.append(texture_to_color(true_labels[i]))
                    edgecolors.append(texture_to_color(pred_labels[i]))
                else:
                    facecolors.append((0.8, 0.8, 0.8))
                    edgecolors.append((0.8, 0.8, 0.8))

            bars = ax.bar(
                x,
                heights,
                width=0.75,
                color=facecolors,
                edgecolor=edgecolors,
                linewidth=2.2,
                align="center",
            )

            for i, bar in enumerate(bars):
                height = float(bar.get_height())

                if height <= 0:
                    continue

                text_color = "white" if rgb_luminance(facecolors[i]) < 0.45 else "black"

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height * 0.55,
                    f"{int(round(height))}",
                    ha="center",
                    va="center",
                    fontsize=COUNT_TEXT_SIZE,
                    color=text_color,
                    clip_on=True,
                )

                if i < n_show:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.6,
                        f"{true_labels[i]}→{pred_labels[i]}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=BAR_LABEL_SIZE,
                        color="black",
                        clip_on=False,
                    )

            ax.set_ylim(0.0, float(ylims[r]))
            ax.set_xticks(x)

            if r == 1:
                ax.set_xticklabels([str(i) for i in x], fontsize=TICK_LABEL_SIZE)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

            if c == 1:
                ax.set_yticklabels([])

            ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Run test-set inference and save the confusion-pair plot."""

    parser = argparse.ArgumentParser(description="Plot top test-set confusion pairs.")
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    classifiers_dir = project_root / "analysis" / "classifiers"

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)
    split_npz = Path(args.split).resolve() if args.split else default_split_path(__file__)

    output_dir = script_dir / "confusion"
    output_path = output_dir / "confusion.pdf"
    cache_path = make_cache_path(output_dir, "confusion_cache.npz")

    dataset = load_test_dataset(split_npz, spike_root)
    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)

    params = common_cache_params(
        plot_name="confusion",
        dataset_name=dataset.name,
        spike_root=spike_root,
        model_weights=weights,
        split_npz=split_npz,
        extra={
            "top_k": args.top_k,
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
        cms = {
            model_type: confusion_matrix_from_result(results[model_type])
            for model_type in MODEL_TYPES
        }

        save_cache(
            cache_path,
            params=params,
            **{f"cm_{model_type}": cms[model_type] for model_type in MODEL_TYPES},
        )
    else:
        cms = {
            model_type: np.asarray(cache[f"cm_{model_type}"], dtype=np.int64)
            for model_type in MODEL_TYPES
        }
        print(f"Loaded cache: {cache_path}")

    top_pairs = {
        model_type: top_k_confusion_pairs(cms[model_type], args.top_k)
        for model_type in MODEL_TYPES
    }

    plot_confusion_pairs(top_pairs, output_path, args.top_k)


if __name__ == "__main__":
    main()