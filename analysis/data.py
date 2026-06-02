"""
Dataset utilities for scripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


TEXTURE_LABELS = [
    "000", "001", "002", "003", "004", "005",
    "011", "012", "013", "014", "015",
    "021", "022", "023", "024", "025",
    "031", "032", "033", "034", "035",
]

TEXTURE_TO_CLASS = {texture: i for i, texture in enumerate(TEXTURE_LABELS)}
CLASS_TO_TEXTURE = {i: texture for texture, i in TEXTURE_TO_CLASS.items()}

NUM_CLASSES = len(TEXTURE_LABELS)
INPUT_SIZE = 9

SPEEDS_MM_MIN = [1200, 2400, 3600, 4800, 6000]
SPEEDS_MM_S = [speed / 60.0 for speed in SPEEDS_MM_MIN]
FORCES = [0.0, 0.5, 1.0, 1.5]

FILENAME_META_RE = re.compile(
    r"Texture\((?P<texture>\d{3})\).*?"
    r"Speed\((?P<speed>[-+]?\d*\.?\d+)\).*?"
    r"Force\((?P<force>[-+]?\d*\.?\d+)\).*?"
    r"Trial\((?P<trial>\d+)\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """Metadata for one spike-train sample."""

    path: Path
    rel_path: Optional[str]
    texture: str
    y: int
    speed_mm_min: Optional[int]
    speed_mm_s: Optional[float]
    force: Optional[float]
    trial: Optional[int]


@dataclass(frozen=True)
class Dataset:
    """Named collection of spike-train samples."""

    name: str
    samples: list[Sample]

    @property
    def paths(self) -> list[Path]:
        """Sample file paths."""

        return [sample.path for sample in self.samples]

    @property
    def labels(self) -> np.ndarray:
        """Integer class labels."""

        return np.asarray([sample.y for sample in self.samples], dtype=np.int64)

    @property
    def textures(self) -> list[str]:
        """Texture labels."""

        return [sample.texture for sample in self.samples]

    @property
    def speeds_mm_min(self) -> np.ndarray:
        """Speed labels in mm/min."""

        return np.asarray(
            [np.nan if sample.speed_mm_min is None else sample.speed_mm_min for sample in self.samples],
            dtype=float,
        )

    @property
    def speeds_mm_s(self) -> np.ndarray:
        """Speed labels in mm/s."""

        return np.asarray(
            [np.nan if sample.speed_mm_s is None else sample.speed_mm_s for sample in self.samples],
            dtype=float,
        )

    @property
    def forces(self) -> np.ndarray:
        """Force labels."""

        return np.asarray(
            [np.nan if sample.force is None else sample.force for sample in self.samples],
            dtype=float,
        )

    @property
    def trials(self) -> np.ndarray:
        """Trial numbers."""

        return np.asarray(
            [-1 if sample.trial is None else sample.trial for sample in self.samples],
            dtype=np.int64,
        )


def project_paths(anchor_file: Optional[str | Path] = None) -> tuple[Path, Path]:
    """Return the project root and analysis directory."""

    start = Path.cwd().resolve() if anchor_file is None else Path(anchor_file).resolve().parent

    for path in [start, *start.parents]:
        if (path / "data").is_dir() or (path / "raw").is_dir() or (path / "analysis").is_dir():
            return path, path / "analysis"

    project_root = start.parent
    return project_root, project_root / "analysis"


def default_processed_root(anchor_file=None) -> Path:
    return project_paths(anchor_file)[0] / "data" / "processed"


def default_spike_root(anchor_file=None) -> Path:
    return project_paths(anchor_file)[0] / "data" / "spike_trains"


def default_split_path(anchor_file=None) -> Path:
    return project_paths(anchor_file)[0] / "data" / "splits" / "split.npz"


def parse_metadata_from_path(path: str | Path) -> tuple[str, int, float, int]:
    """Parse texture, speed, force, and trial from a spike-train filename."""

    match = FILENAME_META_RE.search(Path(path).name)
    texture = match.group("texture")
    speed_mm_min = int(float(match.group("speed")))
    force = float(match.group("force"))
    trial = int(match.group("trial"))

    return texture, speed_mm_min, force, trial


def parse_texture_from_relpath(rel_path: str) -> str:
    """Parse the texture label from a split.npz relative path."""

    return rel_path.replace("\\", "/").split("/")[0][-3:]


def sample_from_path(path: str | Path, rel_path: Optional[str] = None) -> Sample:
    """Create sample metadata from a spike-train path."""

    path = Path(path).resolve()

    try:
        texture, speed_mm_min, force, trial = parse_metadata_from_path(path)
    except AttributeError:
        texture = parse_texture_from_relpath(rel_path or str(path))
        speed_mm_min = None
        force = None
        trial = None

    speed_mm_s = None if speed_mm_min is None else speed_mm_min / 60.0

    return Sample(
        path=path,
        rel_path=rel_path,
        texture=texture,
        y=TEXTURE_TO_CLASS[texture],
        speed_mm_min=speed_mm_min,
        speed_mm_s=speed_mm_s,
        force=force,
        trial=trial,
    )


def load_test_dataset(
    split_npz: str | Path,
    spike_root: str | Path,
    *,
    split_key: str = "test_files",
) -> Dataset:
    """Load test samples listed in split.npz."""

    split_npz = Path(split_npz).resolve()
    spike_root = Path(spike_root).resolve()
    rel_paths = np.load(split_npz, allow_pickle=True)[split_key].tolist()

    samples: list[Sample] = []
    missing = 0

    for rel_path in rel_paths:
        rel_path = str(rel_path)
        path = (spike_root / rel_path).resolve()

        if not path.is_file():
            missing += 1
            continue

        samples.append(sample_from_path(path, rel_path=rel_path))

    if missing:
        print(f"Warning: {missing} split files were missing and skipped.")

    return Dataset(name="test", samples=samples)


def load_all_dataset(
    spike_root: str | Path,
    *,
    textures: Optional[Iterable[str]] = None,
) -> Dataset:
    """Load all spike-train samples under Texture### folders."""

    spike_root = Path(spike_root).resolve()
    textures = list(TEXTURE_LABELS if textures is None else textures)

    samples: list[Sample] = []
    missing_folders = 0

    for texture in textures:
        texture_dir = spike_root / f"Texture{texture}"

        if not texture_dir.is_dir():
            missing_folders += 1
            continue

        for path in sorted(texture_dir.glob("*.npz")):
            samples.append(sample_from_path(path))

    if missing_folders:
        print(f"Warning: {missing_folders} texture folders were missing.")

    return Dataset(name="all", samples=samples)


def subsample_per_texture(
    dataset: Dataset,
    samples_per_texture: Optional[int],
    *,
    seed: int = 0,
) -> Dataset:
    """Return a dataset with a fixed number of samples per texture."""

    if samples_per_texture is None:
        return dataset

    rng = np.random.default_rng(seed)
    selected: list[Sample] = []

    for texture in TEXTURE_LABELS:
        texture_samples = [sample for sample in dataset.samples if sample.texture == texture]
        indices = np.arange(len(texture_samples))
        rng.shuffle(indices)
        selected.extend(texture_samples[int(index)] for index in indices[:samples_per_texture])

    return Dataset(name=dataset.name, samples=selected)


def load_spikes_npz(path: str | Path) -> np.ndarray:
    """Load a spike-train npz file as a binary array with shape [T, 9]."""

    data = np.load(path, allow_pickle=True)

    if "spikes" in data:
        array = data["spikes"]
    elif "data" in data:
        array = data["data"]
    else:
        array = data[list(data.keys())[0]]

    array = np.asarray(array)

    if array.ndim == 3:
        array = np.squeeze(array)

    if array.shape[0] == INPUT_SIZE and array.shape[1] != INPUT_SIZE:
        array = array.T

    return (array[:, :INPUT_SIZE] > 0).astype(np.uint8)


def first_spike_cutoff(spikes_tc: np.ndarray) -> Optional[int]:
    """Return the first-spike cutoff, or None when any channel never spikes."""

    first_times: list[int] = []

    for channel in range(spikes_tc.shape[1]):
        spike_indices = np.flatnonzero(spikes_tc[:, channel])

        if spike_indices.size == 0:
            return None

        first_times.append(int(spike_indices[0]))

    return min(max(first_times) + 1, spikes_tc.shape[0])


def spikes_to_rate_windows(
    spikes_ct: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Convert [C, T] spikes into rate-coded windows with shape [W, C]."""

    channels, time_steps = spikes_ct.shape
    windows: list[np.ndarray] = []

    start = 0
    while start < time_steps:
        end = min(start + window_size, time_steps)
        windows.append(spikes_ct[:, start:end].mean(axis=1).astype(np.float32))

        if end >= time_steps:
            break

        start += stride

    if not windows:
        return np.zeros((1, channels), dtype=np.float32)

    return np.stack(windows, axis=0)


def texture_to_group(texture: str) -> str:
    """Return the texture group name for a texture label."""

    if texture == "000":
        return "Flat"
    if texture in {"001", "002", "003", "004", "005"}:
        return "Ridges"
    if texture in {"011", "012", "013", "014", "015"}:
        return "Waves"
    if texture in {"021", "022", "023", "024", "025"}:
        return "Bumps"
    if texture in {"031", "032", "033", "034", "035"}:
        return "Spikes"

    return "Unknown"