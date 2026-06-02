"""
Latency-accuracy plotting for the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import default_spike_root, default_split_path, load_spikes_npz, load_test_dataset
from inference import (
    MODEL_TYPES,
    WINDOW_SIZE,
    STRIDE,
    NUM_STEPS,
    accuracy_from_result,
    find_latest_weights_by_model,
    load_models,
    make_deadlines,
    predict_dataset,
)


FIGURE_SIZE = (4.8, 2.6)

GLOBAL_LABEL_FONT_SIZE = 9
TICK_LABEL_SIZE = 7
LEGEND_FONT_SIZE = 7

CURVE_LINEWIDTH = 1.0
GRID_LINEWIDTH = 0.6
GRID_ALPHA = 0.35

FS_MEAN_DEADLINE_MS = 676
FS_MEAN_LINEWIDTH = 1.0

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02


def style_map() -> dict[str, dict[str, object]]:
    """Return line styles for latency curves."""

    return {
        "SNN": {"color": (0.70, 0.05, 0.05), "label": "SNN"},
        "FS-SNN": {"color": (0.05, 0.55, 0.05), "label": "FS-SNN"},
        "tSNN": {"color": (0.05, 0.20, 0.65), "label": "tSNN"},
        "FS-tSNN": {"color": (0.85, 0.45, 0.05), "label": "FS-tSNN"},
    }


def plot_latency(
    deadlines_ms: list[int],
    acc_by_model: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save a latency-accuracy curve plot."""

    styles = style_map()

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111)

    for model_type in MODEL_TYPES:
        style = styles[model_type]
        ax.plot(
            deadlines_ms,
            acc_by_model[model_type],
            linewidth=CURVE_LINEWIDTH,
            color=style["color"],
            label=style["label"],
        )

    ax.axvline(
        FS_MEAN_DEADLINE_MS,
        linewidth=FS_MEAN_LINEWIDTH,
        linestyle="--",
        color="black",
        label="FS Mean",
    )

    ax.set_xlabel("Decision Deadline (ms)", fontsize=GLOBAL_LABEL_FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=GLOBAL_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved: {output_path}")


def max_time_steps(dataset) -> int:
    """Return the maximum spike-train length in a dataset."""

    return max(load_spikes_npz(sample.path).shape[0] for sample in dataset.samples)


def main() -> None:
    """Run deadline-clipped inference and save the latency plot."""

    parser = argparse.ArgumentParser(description="Plot test-set latency accuracy.")
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    classifiers_dir = project_root / "analysis" / "classifiers"

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)
    split_npz = Path(args.split).resolve() if args.split else default_split_path(__file__)

    output_dir = script_dir / "latency"
    output_path = output_dir / "latency.pdf"
    cache_path = make_cache_path(output_dir, "latency_cache.npz")

    dataset = load_test_dataset(split_npz, spike_root)
    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)

    deadlines_ms = make_deadlines(max_time_steps(dataset), window_size=WINDOW_SIZE, stride=STRIDE)

    params = common_cache_params(
        plot_name="latency",
        dataset_name=dataset.name,
        spike_root=spike_root,
        model_weights=weights,
        split_npz=split_npz,
        extra={
            "deadlines_ms": deadlines_ms,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "num_steps": NUM_STEPS,
            "fs_policy": "skip",
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is None:
        models = load_models(weights)
        acc_by_model: dict[str, np.ndarray] = {}

        for model_type in MODEL_TYPES:
            curve: list[float] = []

            for deadline in deadlines_ms:
                result = predict_dataset(
                    models[model_type],
                    model_type,
                    dataset,
                    weights_path=weights[model_type],
                    device=args.device,
                    deadline=deadline,
                )
                curve.append(accuracy_from_result(result))

            acc_by_model[model_type] = np.asarray(curve, dtype=np.float64)

        save_cache(
            cache_path,
            params=params,
            deadlines_ms=np.asarray(deadlines_ms, dtype=np.int64),
            **{f"acc_{model_type}": acc_by_model[model_type] for model_type in MODEL_TYPES},
        )
    else:
        deadlines_ms = [int(x) for x in np.asarray(cache["deadlines_ms"]).tolist()]
        acc_by_model = {
            model_type: np.asarray(cache[f"acc_{model_type}"], dtype=np.float64)
            for model_type in MODEL_TYPES
        }
        print(f"Loaded cache: {cache_path}")

    plot_latency(deadlines_ms, acc_by_model, output_path)


if __name__ == "__main__":
    main()