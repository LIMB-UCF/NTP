"""
Neuromorphic encoding for processed taxel data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TEXTURE_LABELS = [
    "000", "001", "002", "003", "004", "005",
    "011", "012", "013", "014", "015",
    "021", "022", "023", "024", "025",
    "031", "032", "033", "034", "035",
]

FILENAME_RE = re.compile(
    r".*Texture\((?P<texture>\d{3})\).*?"
    r"Speed\((?P<speed>\d+)\).*?"
    r"Force\((?P<force>[0-9.]+)\).*?"
    r"Trial\((?P<trial>\d+)\).*"
)

A = 0.02
B = 0.2
C = -65.0
D = 8.0
K = 30.0
TAU = 1.0

TARGET_HZ = 1000.0


def project_root() -> Path:
    """Return the project root containing raw, data, and analysis directories."""

    path = Path(__file__).resolve()

    for parent in path.parents:
        if (parent / "raw").exists() or (parent / "data").exists() or (parent / "analysis").exists():
            return parent

    return path.parents[2]


def default_processed_root() -> Path:
    """Return the default processed-data directory."""

    return project_root() / "data" / "processed"


def default_spike_root() -> Path:
    """Return the default spike-train directory."""

    return project_root() / "data" / "spike_trains"


def parse_processed_filename(path: str | Path) -> tuple[str, str, str, str]:
    """Parse texture, speed, force, and trial from a processed-data filename."""

    match = FILENAME_RE.match(Path(path).name)

    return (
        match.group("texture"),
        match.group("speed"),
        match.group("force"),
        match.group("trial"),
    )


def izhikevich_encoding(current: np.ndarray) -> np.ndarray:
    """Encode input current with the Izhikevich neuron model."""

    time_steps, channels = current.shape

    v = np.full(channels, C)
    u = np.full(channels, B * C)
    spikes = np.zeros((time_steps, channels), dtype=np.uint8)

    for t in range(time_steps):
        dv = TAU * (0.04 * v * v + 5.0 * v + 140.0 - u + K * current[t])
        du = TAU * (A * (B * v - u))

        v += dv
        u += du

        fired = v >= 30.0

        if np.any(fired):
            spikes[t, fired] = 1
            v[fired] = C
            u[fired] += D

    return spikes


def resample_unique(channel: np.ndarray, t_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Linearly resample one channel after removing duplicate time stamps."""

    t_unique, unique_idx = np.unique(t_src, return_index=True)
    channel_unique = channel[unique_idx]

    if t_unique.size == 1:
        return np.full(t_target.shape, channel_unique[0])

    return np.interp(t_target, t_unique, channel_unique)


def encode_csv(csv_path: str | Path) -> np.ndarray:
    """Encode one processed CSV file into a binary spike matrix."""

    df = pd.read_csv(csv_path, header=None)

    sensors = df.iloc[:, :10].to_numpy(float)
    time_s = df.iloc[:, 10].to_numpy(float)

    duration_s = float(time_s.max())
    target_steps = int(round(duration_s * TARGET_HZ))

    if target_steps == 0:
        return np.zeros((0, 10), dtype=np.uint8)

    t_target = np.linspace(0.0, duration_s, num=target_steps, endpoint=True)

    current = np.vstack(
        [
            resample_unique(sensors[:, channel], time_s, t_target)
            for channel in range(sensors.shape[1])
        ]
    ).T

    return izhikevich_encoding(current)


def save_spikes(path: str | Path, spikes: np.ndarray) -> None:
    """Save a binary spike matrix to a compressed npz file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, spikes=spikes.astype(np.uint8))


def output_name(texture: str, speed: str, force: str, trial: str) -> str:
    """Return the spike-train filename for one sample."""

    return (
        f"SpikeTrains[Texture({texture})][Speed({speed})]"
        f"[Force({force})][Trial({trial})].npz"
    )


def encode_all(processed_root: str | Path, spike_root: str | Path) -> None:
    """Encode all processed texture folders into spike-train npz files."""

    processed_root = Path(processed_root)
    spike_root = Path(spike_root)

    for texture in TEXTURE_LABELS:
        texture_dir = processed_root / f"Texture{texture}"

        if not texture_dir.is_dir():
            continue

        print(f"Encoding data from Texture{texture}.")

        out_dir = spike_root / f"Texture{texture}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(texture_dir.glob("*.csv")):
            try:
                parsed_texture, speed, force, trial = parse_processed_filename(csv_path)
                spikes = encode_csv(csv_path)

                if spikes.size == 0:
                    print(f"  Skipped file {csv_path.name} due to: zero duration or no data.")
                    continue

                save_spikes(
                    out_dir / output_name(parsed_texture, speed, force, trial),
                    spikes,
                )
            except Exception as error:
                print(f"  Skipped file {csv_path.name} due to: {error}.")


def main() -> None:
    """Parse arguments and encode processed taxel data."""

    parser = argparse.ArgumentParser(description="Encode processed taxel CSV files into spike trains.")
    parser.add_argument("--processed_root", default=None)
    parser.add_argument("--spike_root", default=None)
    args = parser.parse_args()

    encode_all(
        processed_root=default_processed_root() if args.processed_root is None else args.processed_root,
        spike_root=default_spike_root() if args.spike_root is None else args.spike_root,
    )


if __name__ == "__main__":
    main()