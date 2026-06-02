"""
ANOVA tests for force-speed classification accuracy on the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import f as f_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import load_cache, make_cache_path, save_cache
from data import (
    FORCES,
    SPEEDS_MM_MIN,
    default_spike_root,
    default_split_path,
    load_test_dataset,
)
from inference import (
    DEFAULT_BATCH_SIZE,
    MODEL_TYPES,
    STRIDE,
    WINDOW_SIZE,
    find_latest_weights_by_model,
    load_models,
    predict_all_models,
)



ANOVA_CACHE_NAME = "anova_cache.npz"


def design_intercept(n_rows: int) -> np.ndarray:
    """Return an intercept-only design matrix."""

    return np.ones((n_rows, 1), dtype=np.float64)


def categorical_dummies(values: np.ndarray, levels: list[object]) -> np.ndarray:
    """Return treatment-coded dummy columns, dropping the first level."""

    values = np.asarray(values)
    columns: list[np.ndarray] = []

    for level in levels[1:]:
        columns.append((values == level).astype(np.float64))

    if not columns:
        return np.zeros((values.shape[0], 0), dtype=np.float64)

    return np.column_stack(columns)


def interaction_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return all pairwise products between two dummy-coded matrices."""

    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.zeros((left.shape[0], 0), dtype=np.float64)

    columns: list[np.ndarray] = []

    for left_col in range(left.shape[1]):
        for right_col in range(right.shape[1]):
            columns.append(left[:, left_col] * right[:, right_col])

    return np.column_stack(columns)


def residual_sum_of_squares(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    """Return residual sum of squares and residual degrees of freedom."""

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients

    rss = float(np.sum(residuals ** 2))
    rank = int(np.linalg.matrix_rank(x))
    df_residual = int(y.shape[0] - rank)

    return rss, df_residual


def partial_f_test(
    y: np.ndarray,
    reduced_x: np.ndarray,
    full_x: np.ndarray,
) -> tuple[int, int, float, float, float, float, float]:
    """Return df, SS, MS, F, p, and partial eta-squared for a nested F-test."""

    reduced_rss, reduced_df_error = residual_sum_of_squares(y, reduced_x)
    full_rss, full_df_error = residual_sum_of_squares(y, full_x)

    df_effect = int(reduced_df_error - full_df_error)
    df_error = int(full_df_error)

    ss_effect = max(float(reduced_rss - full_rss), 0.0)
    ss_error = float(full_rss)

    if df_effect <= 0 or df_error <= 0:
        ms_effect = np.nan
        ms_error = np.nan
        f_value = np.nan
        p_value = np.nan
        partial_eta_squared = np.nan
    else:
        ms_effect = ss_effect / float(df_effect)
        ms_error = ss_error / float(df_error)

        if ms_error <= 0.0:
            f_value = np.nan
            p_value = np.nan
        else:
            f_value = ms_effect / ms_error
            p_value = (
                np.nan
                if f_distribution is None
                else float(f_distribution.sf(f_value, df_effect, df_error))
            )

        denominator = ss_effect + ss_error
        partial_eta_squared = np.nan if denominator <= 0.0 else ss_effect / denominator

    return (
        df_effect,
        df_error,
        ss_effect,
        ss_error,
        ms_effect,
        ms_error,
        f_value,
        p_value,
        partial_eta_squared,
    )


def kept_trial_data(dataset, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correctness, force, and speed arrays for retained samples."""

    correct: list[float] = []
    forces: list[float] = []
    speeds: list[int] = []

    for local_index, dataset_index in enumerate(result.kept_indices):
        sample = dataset.samples[int(dataset_index)]

        if sample.force is None or sample.speed_mm_min is None:
            continue

        correct.append(float(result.y_true[local_index] == result.y_pred[local_index]))
        forces.append(float(sample.force))
        speeds.append(int(sample.speed_mm_min))

    return (
        np.asarray(correct, dtype=np.float64),
        np.asarray(forces, dtype=np.float64),
        np.asarray(speeds, dtype=np.int64),
    )


def anova_rows_for_model(model_type: str, dataset, result) -> list[dict[str, object]]:
    """Return ANOVA result rows for one model."""

    y, force_values, speed_values = kept_trial_data(dataset, result)

    if y.size == 0:
        raise ValueError(f"No retained samples available for {model_type}.")

    force_levels = list(FORCES)
    speed_levels = list(SPEEDS_MM_MIN)
    condition_levels = [
        f"{force:g}_{speed}"
        for force in force_levels
        for speed in speed_levels
    ]

    condition_values = np.asarray(
        [
            f"{force:g}_{speed}"
            for force, speed in zip(force_values, speed_values)
        ],
        dtype=str,
    )

    intercept = design_intercept(y.shape[0])

    force_x = categorical_dummies(force_values, force_levels)
    speed_x = categorical_dummies(speed_values, speed_levels)
    condition_x = categorical_dummies(condition_values, condition_levels)
    interaction_x = interaction_columns(force_x, speed_x)

    intercept_force = np.column_stack([intercept, force_x])
    intercept_speed = np.column_stack([intercept, speed_x])
    intercept_force_speed = np.column_stack([intercept, force_x, speed_x])
    intercept_force_speed_interaction = np.column_stack(
        [intercept, force_x, speed_x, interaction_x]
    )
    intercept_condition = np.column_stack([intercept, condition_x])

    tests = [
        (
            "Two-way",
            "Force",
            intercept_speed,
            intercept_force_speed,
        ),
        (
            "Two-way",
            "Speed",
            intercept_force,
            intercept_force_speed,
        ),
        (
            "Two-way",
            "Force by Speed",
            intercept_force_speed,
            intercept_force_speed_interaction,
        ),
        (
            "One-way",
            "Force-Speed Condition",
            intercept,
            intercept_condition,
        ),
    ]

    rows: list[dict[str, object]] = []

    for analysis_type, effect, reduced_x, full_x in tests:
        (
            df_effect,
            df_error,
            ss_effect,
            ss_error,
            ms_effect,
            ms_error,
            f_value,
            p_value,
            partial_eta_squared,
        ) = partial_f_test(y, reduced_x, full_x)

        rows.append(
            {
                "model": model_type,
                "analysis": analysis_type,
                "effect": effect,
                "n": int(y.shape[0]),
                "df_effect": df_effect,
                "df_error": df_error,
                "ss_effect": ss_effect,
                "ss_error": ss_error,
                "ms_effect": ms_effect,
                "ms_error": ms_error,
                "f_value": f_value,
                "p_value": p_value,
                "partial_eta_squared": partial_eta_squared,
            }
        )

    return rows


def rows_to_cache_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Convert ANOVA table rows into arrays for npz storage."""

    return {
        "model": np.asarray([row["model"] for row in rows], dtype=str),
        "analysis": np.asarray([row["analysis"] for row in rows], dtype=str),
        "effect": np.asarray([row["effect"] for row in rows], dtype=str),
        "n": np.asarray([row["n"] for row in rows], dtype=np.int64),
        "df_effect": np.asarray([row["df_effect"] for row in rows], dtype=np.int64),
        "df_error": np.asarray([row["df_error"] for row in rows], dtype=np.int64),
        "ss_effect": np.asarray([row["ss_effect"] for row in rows], dtype=np.float64),
        "ss_error": np.asarray([row["ss_error"] for row in rows], dtype=np.float64),
        "ms_effect": np.asarray([row["ms_effect"] for row in rows], dtype=np.float64),
        "ms_error": np.asarray([row["ms_error"] for row in rows], dtype=np.float64),
        "f_value": np.asarray([row["f_value"] for row in rows], dtype=np.float64),
        "p_value": np.asarray([row["p_value"] for row in rows], dtype=np.float64),
        "partial_eta_squared": np.asarray(
            [row["partial_eta_squared"] for row in rows],
            dtype=np.float64,
        ),
    }


def cache_arrays_to_rows(cache: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Convert cached ANOVA arrays back into row dictionaries."""

    required_keys = [
        "model",
        "analysis",
        "effect",
        "n",
        "df_effect",
        "df_error",
        "ss_effect",
        "ss_error",
        "ms_effect",
        "ms_error",
        "f_value",
        "p_value",
        "partial_eta_squared",
    ]

    missing = [key for key in required_keys if key not in cache]

    if missing:
        raise KeyError(f"ANOVA cache is missing required keys: {missing}")

    n_rows = len(cache["model"])
    rows: list[dict[str, object]] = []

    for index in range(n_rows):
        rows.append(
            {
                "model": str(cache["model"][index]),
                "analysis": str(cache["analysis"][index]),
                "effect": str(cache["effect"][index]),
                "n": int(cache["n"][index]),
                "df_effect": int(cache["df_effect"][index]),
                "df_error": int(cache["df_error"][index]),
                "ss_effect": float(cache["ss_effect"][index]),
                "ss_error": float(cache["ss_error"][index]),
                "ms_effect": float(cache["ms_effect"][index]),
                "ms_error": float(cache["ms_error"][index]),
                "f_value": float(cache["f_value"][index]),
                "p_value": float(cache["p_value"][index]),
                "partial_eta_squared": float(cache["partial_eta_squared"][index]),
            }
        )

    return rows


def format_float(value: float, *, precision: int = 6) -> str:
    """Format a float for table display."""

    if np.isnan(value):
        return "nan"

    if abs(value) >= 1e4 or 0.0 < abs(value) < 1e-4:
        return f"{value:.{precision}e}"

    return f"{value:.{precision}f}"


def print_anova_table(rows: list[dict[str, object]]) -> None:
    """Print ANOVA results as a fixed-width table."""

    headers = [
        "Model",
        "Analysis",
        "Effect",
        "n",
        "df",
        "SS",
        "MS",
        "F",
        "p",
        "partial_eta2",
    ]

    table_rows = []

    for row in rows:
        table_rows.append(
            [
                str(row["model"]),
                str(row["analysis"]),
                str(row["effect"]),
                str(row["n"]),
                f'{int(row["df_effect"])}, {int(row["df_error"])}',
                format_float(float(row["ss_effect"])),
                format_float(float(row["ms_effect"])),
                format_float(float(row["f_value"])),
                format_float(float(row["p_value"])),
                format_float(float(row["partial_eta_squared"])),
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


def main() -> None:
    """Run inference, compute ANOVA results, cache the table, and print it."""

    parser = argparse.ArgumentParser(
        description="Run ANOVA tests for model accuracy across force-speed conditions."
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

    output_dir = script_dir / "anova"
    cache_path = make_cache_path(output_dir, ANOVA_CACHE_NAME)

    cache = None if args.force_recompute else load_cache(cache_path)

    if cache is not None:
        rows = cache_arrays_to_rows(cache)
        print(f"Loaded cache: {cache_path}")
        print_anova_table(rows)
        return

    dataset = load_test_dataset(split_npz, spike_root)
    weights = find_latest_weights_by_model(classifiers_dir, MODEL_TYPES)
    models = load_models(weights)

    results = predict_all_models(
        models,
        dataset,
        weights_by_model=weights,
        device=args.device,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        batch_size=args.batch_size,
    )

    rows: list[dict[str, object]] = []

    for model_type in MODEL_TYPES:
        rows.extend(anova_rows_for_model(model_type, dataset, results[model_type]))

    save_cache(
        cache_path,
        **rows_to_cache_arrays(rows),
    )

    print(f"Saved: {cache_path}")
    print_anova_table(rows)


if __name__ == "__main__":
    main()