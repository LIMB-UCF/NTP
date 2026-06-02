"""
Train-validation-test split generation for spike-train data.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(
    r"SpikeTrains\[Texture\((\d{3})\)\]"
    r"\[Speed\((\d+)\)\]"
    r"\[Force\(([0-9.]+)\)\]"
    r"\[Trial\((\d+)\)\]\.npz$"
)

DEFAULT_SEED = 0
DEFAULT_OUT_NAME = "split.npz"


def project_root() -> Path:
    """Return the project root containing raw, data, and analysis directories."""

    path = Path(__file__).resolve()

    for parent in path.parents:
        if (parent / "raw").exists() or (parent / "data").exists() or (parent / "analysis").exists():
            return parent

    return path.parents[2]


def default_spike_root() -> Path:
    """Return the default spike-train directory."""

    return project_root() / "data" / "spike_trains"


def default_split_path() -> Path:
    """Return the default split output path."""

    return project_root() / "data" / "splits" / DEFAULT_OUT_NAME


def parse_npz_filename(filename: str) -> tuple[str, int, float, int]:
    """Parse texture, speed, force, and trial from a spike-train filename."""

    match = FILENAME_RE.match(filename)
    texture, speed, force, trial = match.groups()

    return texture, int(speed), float(force), int(trial)


def gather_spike_files(spike_root: str | Path) -> list[Path]:
    """Return all matching spike-train npz files under the spike root."""

    spike_root = Path(spike_root)

    return sorted(
        path
        for path in spike_root.rglob("*.npz")
        if FILENAME_RE.match(path.name)
    )


def relpath_from_spike_root(path: str | Path, spike_root: str | Path) -> str:
    """Return a file path relative to the spike-train root."""

    return str(Path(path).resolve().relative_to(Path(spike_root).resolve()))


def split_counts(n: int) -> tuple[int, int, int]:
    """Return train, validation, and test counts for a group size."""

    if n == 100:
        return 80, 10, 10

    n_train = int(round(0.80 * n))
    n_val = int(round(0.10 * n))
    n_test = n - n_train - n_val

    if n_val == 0 and n >= 2:
        n_val = 1
        n_train = max(1, n_train - 1)

    if n_test == 0 and n >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)

    return n_train, n_val, n_test


def create_split(
    *,
    spike_root: str | Path,
    out_path: str | Path,
    seed: int,
    strict_100: bool = False,
) -> None:
    """Create and save a deterministic train-validation-test split."""

    spike_root = Path(spike_root).resolve()
    out_path = Path(out_path).resolve()

    groups: dict[tuple[str, int, float], list[tuple[int, Path]]] = defaultdict(list)

    for path in gather_spike_files(spike_root):
        texture, speed, force, trial = parse_npz_filename(path.name)
        groups[(texture, speed, force)].append((trial, path))

    rng = np.random.default_rng(seed)

    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []

    train_meta: list[tuple[str, int, float, int]] = []
    val_meta: list[tuple[str, int, float, int]] = []
    test_meta: list[tuple[str, int, float, int]] = []

    for key in sorted(groups.keys(), key=lambda item: (item[0], item[1], item[2])):
        texture, speed, force = key
        items = sorted(groups[key], key=lambda item: item[0])
        n = len(items)

        if strict_100 and n != 100:
            raise RuntimeError(
                f"Group (Texture{texture}, Speed{speed}, Force{force}) has {n} trials, expected 100."
            )

        if n < 10:
            raise RuntimeError(
                f"Group (Texture{texture}, Speed{speed}, Force{force}) has too few trials ({n}) to split."
            )

        n_train, n_val, n_test = split_counts(n)

        indices = np.arange(n)
        rng.shuffle(indices)

        split_indices = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:n_train + n_val + n_test],
        }

        for split_name, split_idx in split_indices.items():
            for index in split_idx:
                trial, path = items[int(index)]
                rel_path = relpath_from_spike_root(path, spike_root)
                meta = (texture, speed, force, trial)

                if split_name == "train":
                    train_files.append(rel_path)
                    train_meta.append(meta)
                elif split_name == "val":
                    val_files.append(rel_path)
                    val_meta.append(meta)
                else:
                    test_files.append(rel_path)
                    test_meta.append(meta)

    train_files_arr = np.asarray(train_files, dtype=object)
    val_files_arr = np.asarray(val_files, dtype=object)
    test_files_arr = np.asarray(test_files, dtype=object)

    train_set = set(train_files_arr.tolist())
    val_set = set(val_files_arr.tolist())
    test_set = set(test_files_arr.tolist())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Split sets are not disjoint.")

    total = len(train_files_arr) + len(val_files_arr) + len(test_files_arr)

    print(f"Groups found: {len(groups)}")
    print(f"Total files:  {total}")
    print(f"Train: {len(train_files_arr)} ({len(train_files_arr) / total * 100:.2f}%)")
    print(f"Val:   {len(val_files_arr)} ({len(val_files_arr) / total * 100:.2f}%)")
    print(f"Test:  {len(test_files_arr)} ({len(test_files_arr) / total * 100:.2f}%)")
    print(f"Output: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        seed=np.asarray([seed], dtype=np.int64),
        spike_root_abs=np.asarray([str(spike_root)], dtype=object),
        train_files=train_files_arr,
        val_files=val_files_arr,
        test_files=test_files_arr,
        train_meta=np.asarray(train_meta, dtype=object),
        val_meta=np.asarray(val_meta, dtype=object),
        test_meta=np.asarray(test_meta, dtype=object),
        version=np.asarray(["v3_paths_relative_to_data_spike_trains"], dtype=object),
    )

    print("Done.")


def main() -> None:
    """Parse arguments and create the spike-train data split."""

    parser = argparse.ArgumentParser(description="Create a deterministic train/validation/test split.")
    parser.add_argument("--spike_root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strict_100", action="store_true")
    args = parser.parse_args()

    create_split(
        spike_root=default_spike_root() if args.spike_root is None else args.spike_root,
        out_path=default_split_path() if args.out is None else args.out,
        seed=args.seed,
        strict_100=args.strict_100,
    )


if __name__ == "__main__":
    main()