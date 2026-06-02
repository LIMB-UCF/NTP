"""
First-spike truncation statistics for spike-train datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import common_cache_params, load_cache, make_cache_path, save_cache
from data import (
    FORCES,
    INPUT_SIZE,
    SPEEDS_MM_MIN,
    Dataset,
    default_spike_root,
    default_split_path,
    first_spike_cutoff,
    load_all_dataset,
    load_spikes_npz,
    load_test_dataset,
)


CACHE_NAME = "first_spike_stats_cache.npz"


def load_dataset(
    *,
    dataset_name: str,
    spike_root: Path,
    split_npz: Optional[Path],
) -> Dataset:
    """Load either all spike-train samples or the test split."""

    if dataset_name == "test":
        if split_npz is None:
            raise ValueError("split_npz is required when dataset_name='test'.")

        return load_test_dataset(split_npz, spike_root)

    if dataset_name == "all":
        return load_all_dataset(spike_root)

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def describe(values: np.ndarray) -> dict[str, float]:
    """Return descriptive statistics for a numeric vector."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }

    percentiles = np.percentile(values, [5, 25, 75, 95])

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p95": float(percentiles[3]),
    }


def collect_first_spike_data(dataset: Dataset) -> dict[str, np.ndarray]:
    """Compute per-sample first-spike truncation data."""

    sample_paths: list[str] = []
    textures: list[str] = []
    speeds: list[int] = []
    forces: list[float] = []
    trials: list[int] = []

    t_before: list[float] = []
    t_after: list[float] = []
    reduction_pct: list[float] = []
    eligible: list[bool] = []
    truncated: list[bool] = []

    skipped_load_error = 0

    for sample in dataset.samples:
        try:
            spikes_tc = load_spikes_npz(sample.path)
        except Exception as error:
            skipped_load_error += 1
            print(f"Warning: could not load {sample.path}: {error}", file=sys.stderr)
            continue

        if spikes_tc.ndim != 2 or spikes_tc.shape[0] == 0:
            skipped_load_error += 1
            print(f"Warning: invalid spike array shape for {sample.path}", file=sys.stderr)
            continue

        before = int(spikes_tc.shape[0])
        cutoff = first_spike_cutoff(spikes_tc)

        sample_paths.append(str(sample.path))
        textures.append(sample.texture)
        speeds.append(-1 if sample.speed_mm_min is None else int(sample.speed_mm_min))
        forces.append(np.nan if sample.force is None else float(sample.force))
        trials.append(-1 if sample.trial is None else int(sample.trial))

        t_before.append(float(before))

        if cutoff is None:
            t_after.append(np.nan)
            reduction_pct.append(np.nan)
            eligible.append(False)
            truncated.append(False)
            continue

        after = int(min(cutoff, before))
        t_after.append(float(after))
        reduction_pct.append(100.0 * (1.0 - float(after) / float(before)))
        eligible.append(True)
        truncated.append(after < before)

    if skipped_load_error:
        print(f"Warning: {skipped_load_error} files were skipped because of load/shape errors.")

    return {
        "sample_path": np.asarray(sample_paths, dtype=str),
        "texture": np.asarray(textures, dtype=str),
        "speed": np.asarray(speeds, dtype=np.int64),
        "force": np.asarray(forces, dtype=np.float64),
        "trial": np.asarray(trials, dtype=np.int64),
        "t_before": np.asarray(t_before, dtype=np.float64),
        "t_after": np.asarray(t_after, dtype=np.float64),
        "reduction_pct": np.asarray(reduction_pct, dtype=np.float64),
        "eligible": np.asarray(eligible, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
    }


def summarize_group(
    *,
    group: str,
    label: str,
    mask: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, object]:
    """Return one summary row for a subset of samples."""

    t_before = data["t_before"][mask]
    t_after = data["t_after"][mask]
    reduction_pct = data["reduction_pct"][mask]
    eligible = data["eligible"][mask]
    truncated = data["truncated"][mask]

    n_total = int(mask.sum())
    n_eligible = int(eligible.sum())
    n_skipped = int(n_total - n_eligible)
    n_truncated = int(truncated.sum())

    before_stats = describe(t_before)
    after_stats = describe(t_after[eligible])
    reduction_stats = describe(reduction_pct[eligible])

    retention_pct = np.nan if n_total == 0 else 100.0 * float(n_eligible) / float(n_total)
    skipped_pct = np.nan if n_total == 0 else 100.0 * float(n_skipped) / float(n_total)
    truncated_pct = np.nan if n_eligible == 0 else 100.0 * float(n_truncated) / float(n_eligible)

    return {
        "group": group,
        "label": label,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_skipped": n_skipped,
        "n_truncated": n_truncated,
        "retention_pct": retention_pct,
        "skipped_pct": skipped_pct,
        "truncated_pct": truncated_pct,
        "t_before_mean": before_stats["mean"],
        "t_before_median": before_stats["median"],
        "t_before_std": before_stats["std"],
        "t_before_min": before_stats["min"],
        "t_before_max": before_stats["max"],
        "t_after_mean": after_stats["mean"],
        "t_after_median": after_stats["median"],
        "t_after_std": after_stats["std"],
        "t_after_min": after_stats["min"],
        "t_after_max": after_stats["max"],
        "t_after_p5": after_stats["p5"],
        "t_after_p25": after_stats["p25"],
        "t_after_p75": after_stats["p75"],
        "t_after_p95": after_stats["p95"],
        "reduction_pct_mean": reduction_stats["mean"],
        "reduction_pct_median": reduction_stats["median"],
        "reduction_pct_std": reduction_stats["std"],
        "reduction_pct_p5": reduction_stats["p5"],
        "reduction_pct_p25": reduction_stats["p25"],
        "reduction_pct_p75": reduction_stats["p75"],
        "reduction_pct_p95": reduction_stats["p95"],
    }


def build_summary_rows(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Build overall, speed, force, and force-speed summary rows."""

    rows: list[dict[str, object]] = []
    n_samples = data["t_before"].shape[0]
    all_mask = np.ones((n_samples,), dtype=bool)

    rows.append(
        summarize_group(
            group="Overall",
            label="All",
            mask=all_mask,
            data=data,
        )
    )

    for speed in SPEEDS_MM_MIN:
        rows.append(
            summarize_group(
                group="Speed",
                label=str(speed),
                mask=data["speed"] == int(speed),
                data=data,
            )
        )

    for force in FORCES:
        rows.append(
            summarize_group(
                group="Force",
                label=f"{force:g}",
                mask=np.isclose(data["force"], float(force), equal_nan=False),
                data=data,
            )
        )

    for force in FORCES:
        for speed in SPEEDS_MM_MIN:
            condition_mask = (
                np.isclose(data["force"], float(force), equal_nan=False)
                & (data["speed"] == int(speed))
            )
            rows.append(
                summarize_group(
                    group="Condition",
                    label=f"force={force:g}, speed={speed}",
                    mask=condition_mask,
                    data=data,
                )
            )

    return rows


def rows_to_cache_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Convert summary rows into arrays for npz storage."""

    keys = list(rows[0].keys())

    arrays: dict[str, np.ndarray] = {}

    for key in keys:
        values = [row[key] for row in rows]

        if key in {"group", "label"}:
            arrays[key] = np.asarray(values, dtype=str)
        elif key.startswith("n_"):
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)

    return arrays


def cache_arrays_to_rows(cache: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Convert cached summary arrays back into row dictionaries."""

    required_keys = [
        "group",
        "label",
        "n_total",
        "n_eligible",
        "n_skipped",
        "n_truncated",
        "retention_pct",
        "skipped_pct",
        "truncated_pct",
        "t_before_mean",
        "t_before_median",
        "t_before_std",
        "t_before_min",
        "t_before_max",
        "t_after_mean",
        "t_after_median",
        "t_after_std",
        "t_after_min",
        "t_after_max",
        "t_after_p5",
        "t_after_p25",
        "t_after_p75",
        "t_after_p95",
        "reduction_pct_mean",
        "reduction_pct_median",
        "reduction_pct_std",
        "reduction_pct_p5",
        "reduction_pct_p25",
        "reduction_pct_p75",
        "reduction_pct_p95",
    ]

    missing = [key for key in required_keys if key not in cache]

    if missing:
        raise KeyError(f"First-spike stats cache is missing required keys: {missing}")

    n_rows = len(cache["group"])
    rows: list[dict[str, object]] = []

    for index in range(n_rows):
        row: dict[str, object] = {}

        for key in required_keys:
            if key in {"group", "label"}:
                row[key] = str(cache[key][index])
            elif key.startswith("n_"):
                row[key] = int(cache[key][index])
            else:
                row[key] = float(cache[key][index])

        rows.append(row)

    return rows


def format_float(value: float, *, precision: int = 2) -> str:
    """Format a float for table display."""

    if np.isnan(value):
        return "nan"

    return f"{value:.{precision}f}"


def print_table(rows: list[dict[str, object]], *, group: str) -> None:
    """Print selected rows as a fixed-width summary table."""

    selected = [row for row in rows if row["group"] == group]

    if not selected:
        return

    headers = [
        "Group",
        "Label",
        "n",
        "Eligible",
        "Skipped",
        "Skipped %",
        "T_before mean",
        "T_before median",
        "T_after mean",
        "T_after median",
        "Reduction mean %",
        "Reduction median %",
    ]

    table_rows = []

    for row in selected:
        table_rows.append(
            [
                str(row["group"]),
                str(row["label"]),
                str(row["n_total"]),
                str(row["n_eligible"]),
                str(row["n_skipped"]),
                format_float(float(row["skipped_pct"])),
                format_float(float(row["t_before_mean"])),
                format_float(float(row["t_before_median"])),
                format_float(float(row["t_after_mean"])),
                format_float(float(row["t_after_median"])),
                format_float(float(row["reduction_pct_mean"])),
                format_float(float(row["reduction_pct_median"])),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]

    header_line = "  ".join(
        headers[col].ljust(widths[col])
        for col in range(len(headers))
    )
    divider_line = "  ".join("-" * widths[col] for col in range(len(headers)))

    print(header_line)
    print(divider_line)

    for row in table_rows:
        print(
            "  ".join(
                row[col].ljust(widths[col])
                for col in range(len(headers))
            )
        )


def print_report(rows: list[dict[str, object]], *, cache_path: Path) -> None:
    """Print first-spike statistics."""

    print(f"Cache: {cache_path}")
    print(f"Input channels used: {INPUT_SIZE}")

    print()
    print("Overall")
    print_table(rows, group="Overall")

    print()
    print("By Speed")
    print_table(rows, group="Speed")

    print()
    print("By Force")
    print_table(rows, group="Force")

    print()
    print("By Force-Speed Condition")
    print_table(rows, group="Condition")


def main() -> None:
    """Compute, cache, and print first-spike truncation statistics."""

    parser = argparse.ArgumentParser(
        description="Compute first-spike truncation statistics for spike trains."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "test"],
        default="all",
        help="Dataset to analyze: all spike trains or the test split.",
    )
    parser.add_argument("--spike_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    spike_root = Path(args.spike_dir).resolve() if args.spike_dir else default_spike_root(__file__)
    split_npz = Path(args.split).resolve() if args.split else default_split_path(__file__)

    output_dir = script_dir / "first_spike_stats"
    cache_path = make_cache_path(output_dir, CACHE_NAME)

    params = common_cache_params(
        plot_name="first_spike_stats",
        dataset_name=args.dataset,
        spike_root=spike_root,
        split_npz=split_npz if args.dataset == "test" else None,
        extra={
            "input_size": INPUT_SIZE,
            "fs_policy": "skip_when_any_channel_never_spikes",
        },
    )

    cache = None if args.force_recompute else load_cache(cache_path, expected_params=params)

    if cache is not None:
        rows = cache_arrays_to_rows(cache)
        print(f"Loaded cache: {cache_path}")
        print_report(rows, cache_path=cache_path)
        return

    dataset = load_dataset(
        dataset_name=args.dataset,
        spike_root=spike_root,
        split_npz=split_npz,
    )

    data = collect_first_spike_data(dataset)
    rows = build_summary_rows(data)

    save_cache(
        cache_path,
        params=params,
        **rows_to_cache_arrays(rows),
    )

    print(f"Saved: {cache_path}")
    print_report(rows, cache_path=cache_path)


if __name__ == "__main__":
    main()