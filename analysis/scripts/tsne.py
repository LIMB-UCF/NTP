"""
Texture-colored t-SNE plotting over the full spike-train dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize as l2normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import default_spike_root, load_all_dataset, subsample_per_texture, texture_to_group
from inference import (
    MODEL_TYPES,
    WINDOW_SIZE,
    STRIDE,
    NUM_STEPS,
    find_latest_weights_by_model,
    load_models,
    predict_dataset,
)


FIGURE_SIZE = (7.6, 6.4)

GLOBAL_LABEL_FONT_SIZE = 11
TICK_LABEL_SIZE = 8
PANEL_LABEL_SIZE = 9

SAVE_DPI = 300
SAVE_BBOX = "tight"
SAVE_PAD_INCHES = 0.02

GROUPS_IN_ORDER = ["Flat", "Ridges", "Waves", "Bumps", "Spikes"]

GROUP_PALETTES = {
    "Flat": ["#8E44AD"],
    "Ridges": ["#C7E9C0", "#A1D99B", "#74C476", "#31A354", "#006D2C"],
    "Waves": ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"],
    "Bumps": ["#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"],
    "Spikes": ["#FEE391", "#FEC44F", "#FE9929", "#D95F0E", "#993404"],
}

DENSITY_SUFFIX_ORDER = [3, 2, 1, 4, 5]
DENSITY_SUFFIX_TO_RANK = {suffix: i for i, suffix in enumerate(DENSITY_SUFFIX_ORDER)}

GROUP_OFFSETS = {
    "Flat": 0,
    "Ridges": 1,
    "Waves": 6,
    "Bumps": 11,
    "Spikes": 16,
}

TOTAL_BINS = 21


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


def texture_density_rank(texture: str) -> int:
    """Return the density rank used for texture color shading."""

    return DENSITY_SUFFIX_TO_RANK.get(int(texture[-1]), 2)


def texture_to_color_id(texture: str) -> int:
    """Return the discrete color-bin index for a texture."""

    if texture == "000":
        return 0

    group = texture_to_group(texture)
    return GROUP_OFFSETS[group] + texture_density_rank(texture)


def build_discrete_cmap_and_norm() -> tuple[mpl.colors.ListedColormap, mpl.colors.BoundaryNorm]:
    """Return the texture-group colormap and normalization."""

    colors = [mpl.colors.to_rgb(GROUP_PALETTES["Flat"][0])]

    for group in ["Ridges", "Waves", "Bumps", "Spikes"]:
        for rank in range(5):
            colors.append(mpl.colors.to_rgb(GROUP_PALETTES[group][rank]))

    cmap = mpl.colors.ListedColormap(colors, name="TextureGroupsDensityDistinct")
    boundaries = np.arange(-0.5, TOTAL_BINS + 0.5, 1.0)
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    return cmap, norm


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


def plot_tsne(
    coords_by_model: dict[str, np.ndarray],
    textures_by_model: dict[str, list[str]],
    output_path: Path,
    show: bool,
) -> None:
    """Save a 2×2 texture-colored t-SNE figure."""

    fig = plt.figure(figsize=FIGURE_SIZE)

    fig.supxlabel("t-SNE Dimension 1", fontsize=GLOBAL_LABEL_FONT_SIZE, y=0.055)
    fig.supylabel("t-SNE Dimension 2", fontsize=GLOBAL_LABEL_FONT_SIZE, x=0.055)

    gs = gridspec.GridSpec(2, 2, wspace=0.03, hspace=0.03)

    panel_grid = [["SNN", "tSNN"], ["FS-SNN", "FS-tSNN"]]
    cmap, norm = build_discrete_cmap_and_norm()

    for r in range(2):
        for c in range(2):
            model_type = panel_grid[r][c]
            ax = fig.add_subplot(gs[r, c])
            add_panel_label(ax, model_type)

            coords = coords_by_model[model_type]
            textures = textures_by_model[model_type]
            color_ids = np.asarray([texture_to_color_id(texture) for texture in textures], dtype=np.int64)

            n_samples = len(textures)
            if n_samples > 20000:
                marker_size = 2
            elif n_samples > 5000:
                marker_size = 4
            else:
                marker_size = 8

            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_ids,
                cmap=cmap,
                norm=norm,
                s=marker_size,
                alpha=1.0,
                linewidths=0.0,
            )

            ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
            ax.set_xticks([])
            ax.set_yticks([])

    cax = fig.add_axes([0.92, 0.16, 0.02, 0.74])
    boundaries = np.arange(-0.5, TOTAL_BINS + 0.5, 1.0)

    colorbar = ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
        boundaries=boundaries,
        ticks=[0, 3, 8, 13, 18],
        spacing="proportional",
        orientation="vertical",
    )

    colorbar.set_ticklabels(GROUPS_IN_ORDER)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    for y in [0.5, 5.5, 10.5, 15.5]:
        cax.hlines(y, 0, 1, colors="k", linewidth=0.4, alpha=0.35)

    fig.subplots_adjust(left=0.085, right=0.90, top=0.96, bottom=0.09)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD_INCHES)

    if show:
        plt.show()

    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Run full-dataset model-output t-SNE and save the texture-colored plot."""

    parser = argparse.ArgumentParser(description="Plot full-dataset texture-colored t-SNE embeddings.")
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

    output_dir = script_dir / "tsne"
    output_path = output_dir / "tsne.pdf"
    cache_path = make_cache_path(output_dir, "tsne_cache.npz")

    dataset = load_all_dataset(spike_root)
    dataset = subsample_per_texture(dataset, args.samples_per_texture, seed=args.seed)

    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)

    params = common_cache_params(
        plot_name="tsne",
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
            "fs_policy": "skip",
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is None:
        models = load_models(weights)
        coords_by_model: dict[str, np.ndarray] = {}
        textures_by_model: dict[str, list[str]] = {}

        for model_type in MODEL_TYPES:
            result = predict_dataset(
                models[model_type],
                model_type,
                dataset,
                weights_path=weights[model_type],
                device=args.device,
            )

            features = preprocess_features(result.scores, args.metric)
            coords_by_model[model_type] = run_tsne(
                features,
                seed=args.seed,
                metric=args.metric,
                perplexity=args.perplexity,
                max_iter=args.max_iter,
            )
            textures_by_model[model_type] = [
                dataset.samples[int(index)].texture
                for index in result.kept_indices
            ]

        save_cache(
            cache_path,
            params=params,
            **{
                f"coords_{model_type}": coords_by_model[model_type]
                for model_type in MODEL_TYPES
            },
            **{
                f"textures_{model_type}": np.asarray(textures_by_model[model_type], dtype=str)
                for model_type in MODEL_TYPES
            },
        )
    else:
        coords_by_model = {
            model_type: np.asarray(cache[f"coords_{model_type}"], dtype=np.float32)
            for model_type in MODEL_TYPES
        }
        textures_by_model = {
            model_type: [str(texture) for texture in np.asarray(cache[f"textures_{model_type}"]).tolist()]
            for model_type in MODEL_TYPES
        }
        print(f"Loaded cache: {cache_path}")

    plot_tsne(coords_by_model, textures_by_model, output_path, show=args.show)


if __name__ == "__main__":
    main()