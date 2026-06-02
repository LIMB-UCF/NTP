"""
Cache utilities for scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def make_cache_path(output_dir: str | Path, filename: str) -> Path:
    """Return an npz cache path inside an output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".npz"):
        filename = f"{filename}.npz"

    return output_dir / filename


def save_cache(path: str | Path, *, params: Optional[dict[str, Any]] = None, **arrays: Any) -> None:
    """Save arrays and optional parameter metadata to an npz cache."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(arrays)

    if params is not None:
        payload["params_json"] = json.dumps(params, sort_keys=True, default=str)

    np.savez_compressed(path, **payload)


def load_cache(
    path: str | Path,
    *,
    expected_params: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Load an npz cache when it exists and optional parameters match."""

    path = Path(path)

    if not path.is_file():
        return None

    loaded = np.load(path, allow_pickle=True)
    cache = {key: loaded[key] for key in loaded.files}

    if expected_params is not None:
        if "params_json" not in cache:
            return None

        expected_json = json.dumps(expected_params, sort_keys=True, default=str)
        cached_json = str(cache["params_json"].item())

        if cached_json != expected_json:
            return None

    return cache


def delete_cache(path: str | Path) -> None:
    """Delete an npz cache if it exists."""

    path = Path(path)

    if path.is_file():
        path.unlink()


def file_signature(path: str | Path) -> dict[str, Any]:
    """Return path, modified time, and size for one file."""

    path = Path(path).resolve()
    stat = path.stat()

    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def weights_signature(weights_by_model: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Return file signatures for model weights."""

    return {
        model_type: file_signature(path)
        for model_type, path in sorted(weights_by_model.items())
    }


def common_cache_params(
    *,
    plot_name: str,
    dataset_name: str,
    spike_root: str | Path,
    model_weights: Optional[dict[str, str | Path]] = None,
    split_npz: Optional[str | Path] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return shared parameter metadata for graphing caches."""

    params: dict[str, Any] = {
        "plot_name": plot_name,
        "dataset_name": dataset_name,
        "spike_root": str(Path(spike_root).resolve()),
    }

    if split_npz is not None:
        params["split_npz"] = file_signature(split_npz)

    if model_weights is not None:
        params["model_weights"] = weights_signature(model_weights)

    if extra is not None:
        params.update(extra)

    return params