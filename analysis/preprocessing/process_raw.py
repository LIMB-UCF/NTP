"""
Raw taxel-data preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEXTURE_LABELS = [
    "000", "001", "002", "003", "004", "005",
    "011", "012", "013", "014", "015",
    "021", "022", "023", "024", "025",
    "031", "032", "033", "034", "035",
]

SPEED_WINDOWS = {
    1200: (18.3, 20.9),
    2400: (17.7, 19.0),
    3600: (17.5, 18.5),
    4800: (17.4, 18.2),
    6000: (17.4, 18.0),
}

OUTLIER_THRESH = 250_000
WINDOW_KEEP_START_S = 10.0
SENSOR_COLS = list(range(10))
TIME_COL = 10

GOOD_TEXTURES = {"002", "003", "004", "005"}
EXTRA_SWAP_TEXTURES = {"000", "033", "034", "035"}

TRANSPOSE_IDX = [0, 3, 6, 1, 4, 7, 2, 5, 8]
ROTATE_PLUS3_IDX = [3, 4, 5, 6, 7, 8, 0, 1, 2]
SWAP_123_456_IDX = [3, 4, 5, 0, 1, 2, 6, 7, 8]


def project_root() -> Path:
    """Return the project root containing raw, data, and analysis directories."""

    path = Path(__file__).resolve()

    for parent in path.parents:
        if (parent / "raw").exists() or (parent / "data").exists() or (parent / "analysis").exists():
            return parent

    return path.parents[2]


def default_raw_root() -> Path:
    """Return the default raw-data directory."""

    return project_root() / "raw"


def default_processed_root() -> Path:
    """Return the default processed-data directory."""

    return project_root() / "data" / "processed"


def parse_speed_from_filename(path: str | Path) -> int:
    """Parse speed from a raw-data filename."""

    name = Path(path).name
    speed_part = [part for part in name.split("][") if part.startswith("Speed(")][0]
    return int(speed_part[len("Speed("):-1])


def texture_from_parent(path: str | Path) -> str:
    """Parse the texture label from a Texture### parent folder."""

    parent_name = Path(path).parent.name
    return parent_name.replace("Texture", "")


def apply_taxel_remap(sensors: pd.DataFrame, texture: str) -> pd.DataFrame:
    """Apply texture-specific taxel remapping while preserving the load-cell column."""

    if texture in GOOD_TEXTURES:
        return sensors

    taxels = sensors.iloc[:, :9].to_numpy(copy=True)
    load = sensors.iloc[:, [9]].to_numpy(copy=True)

    taxels = taxels[:, TRANSPOSE_IDX]
    taxels = taxels[:, ROTATE_PLUS3_IDX]

    if texture in EXTRA_SWAP_TEXTURES:
        taxels = taxels[:, SWAP_123_456_IDX]

    return pd.DataFrame(np.hstack([taxels, load]), index=sensors.index)


def process_one(src: str | Path, dst: str | Path) -> str | None:
    """Process one raw CSV file and return a skip reason when not written."""

    src = Path(src)
    dst = Path(dst)

    try:
        speed = parse_speed_from_filename(src)
        start_s, end_s = SPEED_WINDOWS[speed]
    except Exception:
        return "could not determine speed/window from filename"

    df = pd.read_csv(src, header=None)

    if df.shape[1] < 11:
        return "bad column count"

    sensors = df.iloc[:, SENSOR_COLS]
    time = df.iloc[:, TIME_COL].astype(float)

    ok = sensors.iloc[:, 9] <= OUTLIER_THRESH
    sensors = sensors[ok].reset_index(drop=True)
    time = time[ok].reset_index(drop=True)

    if time.empty:
        return "no data left after outlier removal"

    reset_points = np.where(np.diff(time) < 0)[0]

    if reset_points.size:
        cut = reset_points[0] + 1
        sensors = sensors.iloc[:cut]
        time = time.iloc[:cut]

    t_rel = time - time.iloc[0]

    if t_rel.iloc[-1] < end_s:
        return "data length too short for the required window"

    texture = texture_from_parent(src)
    sensors = apply_taxel_remap(sensors, texture)

    baseline_mask = t_rel < WINDOW_KEEP_START_S

    if baseline_mask.sum() == 0:
        return "offset subtraction failure"

    offsets = sensors[baseline_mask].mean()
    sensors_off = sensors - offsets

    window_mask = (t_rel >= start_s) & (t_rel < end_s)

    if not window_mask.any():
        return "no data found in the specified time window"

    sensors_final = sensors_off[window_mask].reset_index(drop=True)
    time_final = (t_rel[window_mask] - start_s).reset_index(drop=True)

    out_df = pd.concat(
        [sensors_final, time_final.rename("time_s")],
        axis=1,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst, index=False, header=False)

    return None


def process_all(raw_root: str | Path, processed_root: str | Path) -> None:
    """Process all texture folders from raw data into processed CSV data."""

    raw_root = Path(raw_root)
    processed_root = Path(processed_root)

    for texture in TEXTURE_LABELS:
        in_dir = raw_root / f"Texture{texture}"
        out_dir = processed_root / f"Texture{texture}"

        if not in_dir.is_dir():
            continue

        print(f"Processing data from Texture{texture}.")

        for csv_path in sorted(in_dir.glob("*.csv")):
            skip_reason = process_one(csv_path, out_dir / csv_path.name)

            if skip_reason:
                print(f"  Skipped file {csv_path.name} due to: {skip_reason}.")


def main() -> None:
    """Parse arguments and process raw taxel-data files."""

    parser = argparse.ArgumentParser(description="Process raw taxel CSV files.")
    parser.add_argument("--raw_root", default=None)
    parser.add_argument("--processed_root", default=None)
    args = parser.parse_args()

    process_all(
        raw_root=default_raw_root() if args.raw_root is None else args.raw_root,
        processed_root=default_processed_root() if args.processed_root is None else args.processed_root,
    )


if __name__ == "__main__":
    main()