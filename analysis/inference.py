"""
Model loading and inference utilities for scripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn

from data import (
    Dataset,
    INPUT_SIZE,
    NUM_CLASSES,
    first_spike_cutoff,
    load_spikes_npz,
    spikes_to_rate_windows,
)


MODEL_TYPES = ["SNN", "tSNN", "FS-SNN", "FS-tSNN"]

RATE_MODEL_TYPES = {"SNN", "FS-SNN"}
TEMPORAL_MODEL_TYPES = {"tSNN", "FS-tSNN"}
FS_MODEL_TYPES = {"FS-SNN", "FS-tSNN"}

NUM_HIDDEN1 = 1000
NUM_HIDDEN2 = 1000
BETA = 0.95

WINDOW_SIZE = 150
STRIDE = 90
NUM_STEPS = 150
DEFAULT_BATCH_SIZE = 128

WEIGHTS_PATTERN = re.compile(r"^Weights_\d{8}_\d{6}\.pt$")


@dataclass
class InferenceResult:
    """Model outputs and retained sample indices for one inference run."""

    model_type: str
    weights_path: Optional[str]
    sample_paths: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray
    kept_indices: np.ndarray
    skipped_indices: np.ndarray


class RateCodedSNN(nn.Module):
    """Fully connected rate-coded SNN for SNN and FS-SNN checkpoints."""

    def __init__(self) -> None:
        super().__init__()

        self.num_steps = NUM_STEPS

        self.fc1 = nn.Linear(INPUT_SIZE, NUM_HIDDEN1)
        self.lif1 = snn.Leaky(beta=BETA)

        self.fc2 = nn.Linear(NUM_HIDDEN1, NUM_HIDDEN2)
        self.lif2 = snn.Leaky(beta=BETA)

        self.fc3 = nn.Linear(NUM_HIDDEN2, NUM_CLASSES)
        self.lif3 = snn.Leaky(beta=BETA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class spike counts for rate-coded input with shape [B, C]."""

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        batch_size = x.shape[0]
        spike_sum = torch.zeros(batch_size, NUM_CLASSES, device=x.device)

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spike_sum += spk3

        return spike_sum


class TemporalSNN(nn.Module):
    """Fully connected temporal SNN for tSNN and FS-tSNN checkpoints."""

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, NUM_HIDDEN1)
        self.lif1 = snn.Leaky(beta=BETA)

        self.fc2 = nn.Linear(NUM_HIDDEN1, NUM_HIDDEN2)
        self.lif2 = snn.Leaky(beta=BETA)

        self.fc3 = nn.Linear(NUM_HIDDEN2, NUM_CLASSES)
        self.lif3 = snn.Leaky(beta=BETA)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return masked class spike counts for temporal input with shape [B, C, T]."""

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        batch_size = x.shape[0]
        time_steps = x.shape[2]
        spike_sum = torch.zeros(batch_size, NUM_CLASSES, device=x.device)

        for t in range(time_steps):
            spk_in = x[:, :, t]

            cur1 = self.fc1(spk_in)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            mask = (t < lengths).float().unsqueeze(1).to(x.device)
            spike_sum += spk3 * mask

        return spike_sum


def is_fs_model(model_type: str) -> bool:
    """Return whether a model type uses first-spike truncation."""

    return model_type in FS_MODEL_TYPES


def is_rate_model(model_type: str) -> bool:
    """Return whether a model type uses rate-coded window inference."""

    return model_type in RATE_MODEL_TYPES


def is_temporal_model(model_type: str) -> bool:
    """Return whether a model type uses temporal inference."""

    return model_type in TEMPORAL_MODEL_TYPES


def model_dir_for_type(analysis_dir: str | Path, model_type: str) -> Path:
    """Return the model checkpoint directory for a model type."""

    return Path(analysis_dir).resolve() / model_type / "Models"


def find_latest_weights(model_dir: str | Path) -> Path:
    """Return the newest Weights_*.pt checkpoint in a model directory."""

    model_dir = Path(model_dir)

    candidates = [
        path
        for path in model_dir.iterdir()
        if path.is_file() and WEIGHTS_PATTERN.match(path.name)
    ]

    if not candidates:
        candidates = [path for path in model_dir.glob("Weights_*.pt") if path.is_file()]

    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_weights_by_model(
    analysis_dir: str | Path,
    model_types: list[str] = MODEL_TYPES,
) -> dict[str, Path]:
    """Return newest checkpoint paths for multiple model types."""

    return {
        model_type: find_latest_weights(model_dir_for_type(analysis_dir, model_type))
        for model_type in model_types
    }


def empty_model_for_type(model_type: str) -> nn.Module:
    """Return the architecture corresponding to a model type."""

    if is_rate_model(model_type):
        return RateCodedSNN()

    return TemporalSNN()


def load_model(model_path: str | Path, model_type: str) -> nn.Module:
    """Load a model checkpoint into the architecture for a model type."""

    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        return checkpoint

    model = empty_model_for_type(model_type)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    return model


def load_models(weights_by_model: dict[str, str | Path]) -> dict[str, nn.Module]:
    """Load models from a model-type to checkpoint-path mapping."""

    return {
        model_type: load_model(path, model_type)
        for model_type, path in weights_by_model.items()
    }


def get_device(device: Optional[str | torch.device] = None) -> torch.device:
    """Return an explicit device or the default available torch device."""

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_spikes_for_model(
    spikes_tc: np.ndarray,
    model_type: str,
    *,
    deadline: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Apply deadline clipping and first-spike truncation for a model type."""

    end = spikes_tc.shape[0] if deadline is None else min(deadline, spikes_tc.shape[0])

    if is_fs_model(model_type):
        cutoff = first_spike_cutoff(spikes_tc)

        if cutoff is None:
            return None

        end = min(end, cutoff)

    return spikes_tc[:end, :]


def rate_features_from_spikes(
    spikes_tc: np.ndarray,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """Convert [T, C] spikes into rate-coded model input with shape [W, C]."""

    return spikes_to_rate_windows(
        spikes_tc.T,
        window_size=window_size,
        stride=stride,
    )


def pad_temporal_batch(spikes_list: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length [T, C] spike arrays into [B, C, T] input."""

    lengths = np.asarray([spikes.shape[0] for spikes in spikes_list], dtype=np.int64)
    max_length = int(lengths.max())
    batch = np.zeros((len(spikes_list), INPUT_SIZE, max_length), dtype=np.float32)

    for i, spikes_tc in enumerate(spikes_list):
        batch[i, :, :spikes_tc.shape[0]] = spikes_tc[:, :INPUT_SIZE].T

    return torch.from_numpy(batch), torch.from_numpy(lengths)


@torch.inference_mode()
def predict_one_sample(
    model: nn.Module,
    model_type: str,
    spikes_tc: np.ndarray,
    *,
    device: Optional[str | torch.device] = None,
    deadline: Optional[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Optional[np.ndarray]:
    """Return one class-score vector, or None when strict FS skipping removes the sample."""

    device = get_device(device)
    model.eval()
    model.to(device)

    results = predict_prepared_batch(
        model,
        model_type,
        [spikes_tc],
        device=device,
        deadline=deadline,
        window_size=window_size,
        stride=stride,
    )

    if not results:
        return None

    return results[0]


@torch.inference_mode()
def predict_prepared_batch(
    model: nn.Module,
    model_type: str,
    spikes_batch: list[np.ndarray],
    *,
    device: torch.device,
    deadline: Optional[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[Optional[np.ndarray]]:
    """Return class-score vectors for one batch of raw [T, C] spike arrays.

    The returned list is aligned to ``spikes_batch``. Entries are ``None`` when
    strict first-spike preprocessing skips a sample.
    """

    prepared_by_position: list[tuple[int, np.ndarray]] = []
    results: list[Optional[np.ndarray]] = [None] * len(spikes_batch)

    for position, spikes_tc in enumerate(spikes_batch):
        prepared = prepare_spikes_for_model(spikes_tc, model_type, deadline=deadline)

        if prepared is not None:
            prepared_by_position.append((position, prepared))

    if not prepared_by_position:
        return results

    if is_rate_model(model_type):
        window_batches: list[np.ndarray] = []
        window_counts: list[int] = []
        positions: list[int] = []

        for position, prepared in prepared_by_position:
            windows = rate_features_from_spikes(
                prepared,
                window_size=window_size,
                stride=stride,
            ).astype(np.float32, copy=False)

            if windows.shape[0] == 0:
                continue

            window_batches.append(windows)
            window_counts.append(int(windows.shape[0]))
            positions.append(position)

        if not window_batches:
            return results

        x = torch.from_numpy(np.concatenate(window_batches, axis=0)).to(device, non_blocking=True)
        window_scores = model(x).detach().cpu().numpy()

        offset = 0
        for position, count in zip(positions, window_counts):
            results[position] = window_scores[offset:offset + count].mean(axis=0)
            offset += count

        return results

    prepared_spikes = [prepared for _, prepared in prepared_by_position]
    x, lengths = pad_temporal_batch(prepared_spikes)
    x = x.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)

    batch_scores = model(x, lengths).detach().cpu().numpy()

    for local_index, (position, _) in enumerate(prepared_by_position):
        results[position] = batch_scores[local_index]

    return results


@torch.inference_mode()
def predict_dataset(
    model: nn.Module,
    model_type: str,
    dataset: Dataset,
    *,
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    deadline: Optional[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> InferenceResult:
    """Run batched model inference over a dataset."""

    device = get_device(device)

    model.eval()
    model.to(device)

    kept_indices: list[int] = []
    skipped_indices: list[int] = []
    scores: list[np.ndarray] = []
    y_true: list[int] = []
    sample_paths: list[str] = []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    for batch_start in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[batch_start:batch_start + batch_size]
        spikes_batch = [load_spikes_npz(sample.path) for sample in batch_samples]

        batch_scores = predict_prepared_batch(
            model,
            model_type,
            spikes_batch,
            device=device,
            deadline=deadline,
            window_size=window_size,
            stride=stride,
        )

        for local_index, sample_scores in enumerate(batch_scores):
            dataset_index = batch_start + local_index
            sample = batch_samples[local_index]

            if sample_scores is None:
                skipped_indices.append(dataset_index)
                continue

            kept_indices.append(dataset_index)
            scores.append(sample_scores)
            y_true.append(sample.y)
            sample_paths.append(str(sample.path))

    if not scores:
        scores_array = np.zeros((0, NUM_CLASSES), dtype=np.float32)
        y_true_array = np.zeros((0,), dtype=np.int64)
        y_pred_array = np.zeros((0,), dtype=np.int64)
    else:
        scores_array = np.stack(scores, axis=0)
        y_true_array = np.asarray(y_true, dtype=np.int64)
        y_pred_array = np.argmax(scores_array, axis=1).astype(np.int64)

    return InferenceResult(
        model_type=model_type,
        weights_path=None if weights_path is None else str(Path(weights_path).resolve()),
        sample_paths=sample_paths,
        y_true=y_true_array,
        y_pred=y_pred_array,
        scores=scores_array,
        kept_indices=np.asarray(kept_indices, dtype=np.int64),
        skipped_indices=np.asarray(skipped_indices, dtype=np.int64),
    )


def predict_all_models(
    models: dict[str, nn.Module],
    dataset: Dataset,
    *,
    weights_by_model: Optional[dict[str, str | Path]] = None,
    device: Optional[str | torch.device] = None,
    deadline: Optional[int] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, InferenceResult]:
    """Run batched inference for multiple models over one dataset."""

    return {
        model_type: predict_dataset(
            model,
            model_type,
            dataset,
            weights_path=None if weights_by_model is None else weights_by_model.get(model_type),
            device=device,
            deadline=deadline,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
        )
        for model_type, model in models.items()
    }

def make_deadlines(
    max_time_steps: int,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[int]:
    """Return decision deadlines using rate-window endpoints."""

    deadlines: list[int] = []
    start = 0

    while True:
        end = start + window_size
        deadlines.append(end)

        if end >= max_time_steps:
            break

        start += stride

    return deadlines


def accuracy_from_result(result: InferenceResult) -> float:
    """Return classification accuracy from an inference result."""

    return float(np.mean(result.y_true == result.y_pred))


def confusion_matrix_from_result(result: InferenceResult) -> np.ndarray:
    """Return a class-by-class confusion matrix from an inference result."""

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for true_label, pred_label in zip(result.y_true, result.y_pred):
        matrix[int(true_label), int(pred_label)] += 1

    return matrix