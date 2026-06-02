"""
Speed- and force-colored t-SNE plotting over the full spike-train dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize as l2normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import FORCES, SPEEDS_MM_S, default_spike_root, load_all_dataset, subsample_per_texture
from inference import (
    WINDOW_SIZE,
    STRIDE,
    NUM_STEPS,
    find_latest_weights_by_model,
    load_model,
    predict_dataset,
)


MODEL_TYPE = "tSNN"

FIGURE_SIZE = (4.8, 2.2)

GLOBAL_LABEL_FONT_SIZE = 9
TICK_LABEL_SIZE = 7
LEGEND_FONT_SIZE = 7

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

SPEED_COLORS = {
    20.0: "#1F77B4",
    40.0: "#2CA02C",
    60.0: "#FF7F0E",
    80.0: "#D62728",
    100.0: "#9467BD",
}

FORCE_COLORS = {
    0.0: "#0072B2",
    0.5: "#009E73",
    1.0: "#D55E00",
    1.5: "#CC79A7",
}


def labels_to_ids(values: np.ndarray, levels: list[float]) -> np.ndarray:
    """Map numeric labels to categorical integer ids."""

    ids = np.zeros(values.shape[0], dtype=np.int64)

    for i, value in enumerate(values):
        ids[i] = int(np.argmin([abs(float(value) - level) for level in levels]))

    return ids


def build_discrete_cmap_and_norm(
    *,
    levels: list[float],
    color_map: dict[float, str],
    name: str,
) -> tuple[mpl.colors.ListedColormap, mpl.colors.BoundaryNorm]:
    """Return a discrete colormap and normalization for categorical labels."""

    colors = [mpl.colors.to_rgb(color_map[level]) for level in levels]
    cmap = mpl.colors.ListedColormap(colors, name=name)
    boundaries = np.arange(-0.5, len(levels) + 0.5, 1.0)
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    return cmap, norm


def build_discrete_legend_handles(
    *,
    levels: list[float],
    color_map: dict[float, str],
    label_formatter,
    markersize: float,
) -> list[mlines.Line2D]:
    """Return legend handles for categorical scatter colors."""

    return [
        mlines.Line2D(
            [],
            [],
            color=color_map[level],
            marker="o",
            linestyle="None",
            markersize=markersize,
            label=label_formatter(level),
        )
        for level in levels
    ]


def preprocess_features(features: np.ndarray, metric: str) -> np.ndarray:
    """Scale or normalize features before t-SNE."""

    if metric == "cosine":
        return l2normalize(features, norm="l2", axis=1, copy=True)

    return StandardScaler().fit_transform(features)


def run_tsne(
    features: np.ndarray,
    *,
    seed: int,
    metric: str,
    perplexity: float,
    max_iter: int,
) -> np.ndarray:
    """Return two-dimensional t-SNE coordinates."""

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric=metric,
        max_iter=max_iter,
        verbose=0,
    )

    return tsne.fit_transform(features).astype(np.float32)


def plot_speed_force_tsne(
    coords: np.ndarray,
    speeds_mm_s: np.ndarray,
    forces: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    """Save a two-panel t-SNE plot colored by speed and force."""

    fig = plt.figure(figsize=FIGURE_SIZE)

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.24],
        wspace=0.04,
    )

    ax_speed = fig.add_subplot(gs[0, 0])
    ax_force = fig.add_subplot(gs[0, 1], sharex=ax_speed, sharey=ax_speed)
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    fig.supxlabel("t-SNE Dimension 1", fontsize=GLOBAL_LABEL_FONT_SIZE, y=0.04)
    fig.supylabel("t-SNE Dimension 2", fontsize=GLOBAL_LABEL_FONT_SIZE, x=0.04)

    n_samples = coords.shape[0]

    if n_samples > 20000:
        marker_size = 2
    elif n_samples > 5000:
        marker_size = 4
    else:
        marker_size = 8

    speed_ids = labels_to_ids(speeds_mm_s, SPEEDS_MM_S)
    force_ids = labels_to_ids(forces, FORCES)

    speed_cmap, speed_norm = build_discrete_cmap_and_norm(
        levels=SPEEDS_MM_S,
        color_map=SPEED_COLORS,
        name="SpeedCategories",
    )

    force_cmap, force_norm = build_discrete_cmap_and_norm(
        levels=FORCES,
        color_map=FORCE_COLORS,
        name="ForceCategories",
    )

    ax_speed.scatter(
        coords[:, 0],
        coords[:, 1],
        c=speed_ids,
        cmap=speed_cmap,
        norm=speed_norm,
        s=marker_size,
        alpha=1.0,
        linewidths=0.0,
        edgecolors="none",
    )

    ax_speed.text(
        0.98,
        0.98,
        "Speed",
        transform=ax_speed.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        zorder=10,
    )

    ax_speed.set_xticks([])
    ax_speed.set_yticks([])
    ax_speed.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    ax_force.scatter(
        coords[:, 0],
        coords[:, 1],
        c=force_ids,
        cmap=force_cmap,
        norm=force_norm,
        s=marker_size,
        alpha=1.0,
        linewidths=0.0,
        edgecolors="none",
    )

    ax_force.text(
        0.98,
        0.98,
        "Force",
        transform=ax_force.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        zorder=10,
    )

    ax_force.set_xticks([])
    ax_force.set_yticks([])
    ax_force.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    speed_handles = build_discrete_legend_handles(
        levels=SPEEDS_MM_S,
        color_map=SPEED_COLORS,
        label_formatter=lambda value: f"{value:g}",
        markersize=5.0,
    )

    force_handles = build_discrete_legend_handles(
        levels=FORCES,
        color_map=FORCE_COLORS,
        label_formatter=lambda value: f"{value:.1f}",
        markersize=5.0,
    )

    speed_legend = ax_leg.legend(
        handles=speed_handles,
        title="Speed (mm/s)",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    ax_leg.add_artist(speed_legend)

    force_legend = ax_leg.legend(
        handles=force_handles,
        title="Force (Δmm)",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    speed_bbox = speed_legend.get_window_extent(renderer=renderer)
    force_bbox = force_legend.get_window_extent(renderer=renderer)

    inv = ax_leg.transAxes.inverted()

    speed_bbox_axes = mpl.transforms.Bbox(inv.transform(speed_bbox))
    force_bbox_axes = mpl.transforms.Bbox(inv.transform(force_bbox))

    gap = (1.0 - speed_bbox_axes.height - force_bbox_axes.height) / 3.0

    if gap < 0:
        print("Warning: legends are too tall to fit with equal spacing in ax_leg.")

    speed_y = 1.0 - gap
    force_y = 1.0 - (2.0 * gap) - speed_bbox_axes.height

    speed_legend.set_bbox_to_anchor((0.0, speed_y), transform=ax_leg.transAxes)
    force_legend.set_bbox_to_anchor((0.0, force_y), transform=ax_leg.transAxes)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD_INCHES)

    if show:
        plt.show()

    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Run full-dataset tSNN t-SNE and save the speed-force plot."""

    parser = argparse.ArgumentParser(description="Plot full-dataset speed/force tSNE embeddings.")
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--samples_per_texture", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    classifiers_dir = project_root / "analysis" / "classifiers"

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)

    output_dir = script_dir / "speed-force_tsne"
    output_path = output_dir / "speed-force_tsne.pdf"
    cache_path = make_cache_path(output_dir, "speed-force_tsne_cache.npz")

    dataset = load_all_dataset(spike_root)
    dataset = subsample_per_texture(dataset, args.samples_per_texture, seed=args.seed)

    weights = find_latest_weights_by_model(classifiers_dir, [MODEL_TYPE])

    params = common_cache_params(
        plot_name="speed-force_tsne",
        dataset_name=dataset.name,
        spike_root=spike_root,
        model_weights=weights,
        extra={
            "samples_per_texture": args.samples_per_texture,
            "seed": args.seed,
            "metric": args.metric,
            "perplexity": args.perplexity,
            "max_iter": args.max_iter,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "num_steps": NUM_STEPS,
            "model_type": MODEL_TYPE,
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is None:
        model = load_model(weights[MODEL_TYPE], MODEL_TYPE)

        result = predict_dataset(
            model,
            MODEL_TYPE,
            dataset,
            weights_path=weights[MODEL_TYPE],
            device=args.device,
        )

        features = preprocess_features(result.scores, args.metric)

        coords = run_tsne(
            features,
            seed=args.seed,
            metric=args.metric,
            perplexity=args.perplexity,
            max_iter=args.max_iter,
        )

        speeds_mm_s = np.asarray(
            [dataset.samples[int(index)].speed_mm_s for index in result.kept_indices],
            dtype=float,
        )
        forces = np.asarray(
            [dataset.samples[int(index)].force for index in result.kept_indices],
            dtype=float,
        )

        save_cache(
            cache_path,
            params=params,
            coords=coords,
            speeds_mm_s=speeds_mm_s,
            forces=forces,
        )
    else:
        coords = np.asarray(cache["coords"], dtype=np.float32)
        speeds_mm_s = np.asarray(cache["speeds_mm_s"], dtype=float)
        forces = np.asarray(cache["forces"], dtype=float)
        print(f"Loaded cache: {cache_path}")

    plot_speed_force_tsne(coords, speeds_mm_s, forces, output_path, show=args.show)


if __name__ == "__main__":
    main()