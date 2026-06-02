"""
Evaluate all models on the first-spike-truncated testing set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import (
    CLASS_TO_TEXTURE,
    Dataset,
    default_spike_root,
    default_split_path,
    first_spike_cutoff,
    load_spikes_npz,
    load_test_dataset,
    texture_to_group,
)
from inference import (
    DEFAULT_BATCH_SIZE,
    MODEL_TYPES,
    STRIDE,
    WINDOW_SIZE,
    find_latest_weights_by_model,
    get_device,
    load_models,
    predict_prepared_batch,
)


CACHE_NAME = "first_spike_eval_cache.npz"


def class_to_group(class_index: int) -> str:
    """Return texture-group name for a class index."""

    texture = CLASS_TO_TEXTURE[int(class_index)]
    return texture_to_group(texture)


def collect_first_spike_test_set(
    dataset: Dataset,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Return first-spike-truncated test samples and aligned labels.

    Samples where any channel never spikes are skipped.
    """

    spikes_fs: list[np.ndarray] = []
    y_true: list[int] = []
    kept_indices: list[int] = []
    sample_paths: list[str] = []

    for index, sample in enumerate(dataset.samples):
        spikes_tc = load_spikes_npz(sample.path)
        cutoff = first_spike_cutoff(spikes_tc)

        if cutoff is None:
            continue

        spikes_fs.append(spikes_tc[:cutoff, :])
        y_true.append(sample.y)
        kept_indices.append(index)
        sample_paths.append(str(sample.path))

    if not spikes_fs:
        raise RuntimeError("No first-spike-eligible test samples were found.")

    return (
        spikes_fs,
        np.asarray(y_true, dtype=np.int64),
        np.asarray(kept_indices, dtype=np.int64),
        sample_paths,
    )


@torch.inference_mode()
def evaluate_model_on_first_spike_data(
    model: nn.Module,
    model_type: str,
    spikes_fs: list[np.ndarray],
    y_true: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> dict[str, object]:
    """Evaluate one model on already first-spike-truncated samples."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model.eval()
    model.to(device)

    scores: list[np.ndarray] = []
    model_skipped = 0

    for batch_start in range(0, len(spikes_fs), batch_size):
        batch_spikes = spikes_fs[batch_start:batch_start + batch_size]

        batch_scores = predict_prepared_batch(
            model,
            model_type,
            batch_spikes,
            device=device,
            deadline=None,
            window_size=window_size,
            stride=stride,
        )

        for sample_scores in batch_scores:
            if sample_scores is None:
                model_skipped += 1
                continue

            scores.append(sample_scores)

    if not scores:
        raise RuntimeError(f"{model_type} produced no usable predictions.")

    scores_array = np.stack(scores, axis=0)
    y_eval = y_true[:scores_array.shape[0]]
    y_pred = np.argmax(scores_array, axis=1).astype(np.int64)

    accuracy = float(np.mean(y_pred == y_eval))

    true_groups = np.asarray([class_to_group(label) for label in y_eval], dtype=str)
    pred_groups = np.asarray([class_to_group(label) for label in y_pred], dtype=str)
    type_accuracy = float(np.mean(true_groups == pred_groups))

    return {
        "model": model_type,
        "n_used": int(y_eval.size),
        "model_skipped": int(model_skipped),
        "accuracy": accuracy,
        "type_accuracy": type_accuracy,
    }


def evaluate_all_models_on_first_spike_data(
    models: dict[str, nn.Module],
    spikes_fs: list[np.ndarray],
    y_true: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[dict[str, object]]:
    """Evaluate all configured model types on the same first-spike data."""

    rows: list[dict[str, object]] = []

    for model_type in MODEL_TYPES:
        rows.append(
            evaluate_model_on_first_spike_data(
                models[model_type],
                model_type,
                spikes_fs,
                y_true,
                device=device,
                batch_size=batch_size,
                window_size=window_size,
                stride=stride,
            )
        )

    return rows


def add_context_to_rows(
    rows: list[dict[str, object]],
    *,
    n_original: int,
    n_first_spike_used: int,
    n_first_spike_skipped: int,
    weights_by_model: dict[str, Path],
) -> list[dict[str, object]]:
    """Attach shared test-set and checkpoint metadata to result rows."""

    enriched: list[dict[str, object]] = []

    for row in rows:
        model_type = str(row["model"])

        enriched.append(
            {
                "model": model_type,
                "n_original": int(n_original),
                "n_first_spike_used": int(n_first_spike_used),
                "n_first_spike_skipped": int(n_first_spike_skipped),
                "model_skipped": int(row["model_skipped"]),
                "accuracy": float(row["accuracy"]),
                "type_accuracy": float(row["type_accuracy"]),
                "weights_path": str(Path(weights_by_model[model_type]).resolve()),
            }
        )

    return enriched


def rows_to_cache_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Convert result rows into arrays for npz storage."""

    return {
        "model": np.asarray([row["model"] for row in rows], dtype=str),
        "n_original": np.asarray([row["n_original"] for row in rows], dtype=np.int64),
        "n_first_spike_used": np.asarray(
            [row["n_first_spike_used"] for row in rows],
            dtype=np.int64,
        ),
        "n_first_spike_skipped": np.asarray(
            [row["n_first_spike_skipped"] for row in rows],
            dtype=np.int64,
        ),
        "model_skipped": np.asarray([row["model_skipped"] for row in rows], dtype=np.int64),
        "accuracy": np.asarray([row["accuracy"] for row in rows], dtype=np.float64),
        "type_accuracy": np.asarray([row["type_accuracy"] for row in rows], dtype=np.float64),
        "weights_path": np.asarray([row["weights_path"] for row in rows], dtype=str),
    }


def cache_arrays_to_rows(cache: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Convert cached result arrays back into row dictionaries."""

    required_keys = [
        "model",
        "n_original",
        "n_first_spike_used",
        "n_first_spike_skipped",
        "model_skipped",
        "accuracy",
        "type_accuracy",
        "weights_path",
    ]

    missing = [key for key in required_keys if key not in cache]

    if missing:
        raise KeyError(f"First-spike evaluation cache is missing required keys: {missing}")

    rows: list[dict[str, object]] = []
    n_rows = len(cache["model"])

    for index in range(n_rows):
        rows.append(
            {
                "model": str(cache["model"][index]),
                "n_original": int(cache["n_original"][index]),
                "n_first_spike_used": int(cache["n_first_spike_used"][index]),
                "n_first_spike_skipped": int(cache["n_first_spike_skipped"][index]),
                "model_skipped": int(cache["model_skipped"][index]),
                "accuracy": float(cache["accuracy"][index]),
                "type_accuracy": float(cache["type_accuracy"][index]),
                "weights_path": str(cache["weights_path"][index]),
            }
        )

    return rows


def format_percent(value: float) -> str:
    """Format a fraction as a percentage string."""

    if np.isnan(value):
        return "nan"

    return f"{100.0 * value:.2f}"


def print_results(rows: list[dict[str, object]], *, cache_path: Path) -> None:
    """Print first-spike evaluation results as a fixed-width table."""

    print(f"Cache: {cache_path}")

    if rows:
        n_original = int(rows[0]["n_original"])
        n_used = int(rows[0]["n_first_spike_used"])
        n_skipped = int(rows[0]["n_first_spike_skipped"])
        retained_pct = 100.0 * n_used / n_original if n_original else np.nan

        print()
        print("First-spike test subset")
        print(f"  Original test samples:      {n_original}")
        print(f"  First-spike samples used:   {n_used}")
        print(f"  First-spike samples skipped:{n_skipped:>4}")
        print(f"  Retained:                   {retained_pct:.2f}%")

    headers = [
        "Model",
        "Accuracy %",
        "Type Accuracy %",
        "N Used",
        "FS Skipped",
        "Model Skipped",
    ]

    table_rows = []

    for row in rows:
        table_rows.append(
            [
                str(row["model"]),
                format_percent(float(row["accuracy"])),
                format_percent(float(row["type_accuracy"])),
                str(row["n_first_spike_used"]),
                str(row["n_first_spike_skipped"]),
                str(row["model_skipped"]),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]

    print()
    header_line = "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))
    divider_line = "  ".join("-" * widths[col] for col in range(len(headers)))

    print(header_line)
    print(divider_line)

    for row in table_rows:
        print("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))

    print()
    print("Weights")
    for row in rows:
        print(f"  {row['model']}: {row['weights_path']}")


def main() -> None:
    """Run first-spike test-set evaluation and cache the result table."""

    parser = argparse.ArgumentParser(
        description="Evaluate all models on first-spike-truncated test data."
    )
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir.parent
    classifiers_dir = analysis_dir / "classifiers"

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)
    split_npz = Path(args.split).resolve() if args.split else default_split_path(__file__)

    output_dir = script_dir / "first_spike_eval"
    cache_path = make_cache_path(output_dir, CACHE_NAME)

    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)

    params = common_cache_params(
        plot_name="first_spike_eval",
        dataset_name="test",
        spike_root=spike_root,
        model_weights=weights,
        split_npz=split_npz,
        extra={
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "batch_size": args.batch_size,
            "fs_policy": "evaluate_all_models_on_common_first_spike_subset",
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is not None:
        rows = cache_arrays_to_rows(cache)
        print(f"Loaded cache: {cache_path}")
        print_results(rows, cache_path=cache_path)
        return

    device = get_device(args.device)
    dataset = load_test_dataset(split_npz, spike_root)

    spikes_fs, y_true, kept_indices, _sample_paths = collect_first_spike_test_set(dataset)

    n_original = len(dataset.samples)
    n_first_spike_used = int(len(spikes_fs))
    n_first_spike_skipped = int(n_original - n_first_spike_used)

    print(f"Using device: {device}")
    print(f"Loaded test split: {n_original} samples")
    print(f"First-spike retained samples: {n_first_spike_used}")
    print(f"First-spike skipped samples: {n_first_spike_skipped}")

    models = load_models(weights)

    rows = evaluate_all_models_on_first_spike_data(
        models,
        spikes_fs,
        y_true,
        device=device,
        batch_size=args.batch_size,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    rows = add_context_to_rows(
        rows,
        n_original=n_original,
        n_first_spike_used=n_first_spike_used,
        n_first_spike_skipped=n_first_spike_skipped,
        weights_by_model=weights,
    )

    save_cache(
        cache_path,
        params=params,
        **rows_to_cache_arrays(rows),
    )

    print(f"Saved: {cache_path}")
    print_results(rows, cache_path=cache_path)


if __name__ == "__main__":
    main()