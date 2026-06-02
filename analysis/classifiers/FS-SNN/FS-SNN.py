"""
Rate-coded spiking neural network training for first-spike-truncated data.
"""

from __future__ import annotations

import os
import re
import sys
import time
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import snntorch as snn

SEED = 0

def set_seed(seed: int = 0) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

TEXTURE_LABEL_MAP: Dict[str, int] = {
    "000": 0, "001": 1, "002": 2, "003": 3, "004": 4, "005": 5,
    "011": 6, "012": 7, "013": 8, "014": 9, "015": 10,
    "021": 11, "022": 12, "023": 13, "024": 14, "025": 15,
    "031": 16, "032": 17, "033": 18, "034": 19, "035": 20,
}
NUM_CLASSES = len(TEXTURE_LABEL_MAP)

TYPE_NAMES = ["Flat", "Ridges", "Waves", "Bumps", "Spikes"]

CLASS_TO_TYPE_MAP = torch.zeros(NUM_CLASSES, dtype=torch.long)
for suffix, class_idx in TEXTURE_LABEL_MAP.items():
    if suffix == "000":
        CLASS_TO_TYPE_MAP[class_idx] = 0
    elif suffix.startswith("00"):
        CLASS_TO_TYPE_MAP[class_idx] = 1
    elif suffix.startswith("01"):
        CLASS_TO_TYPE_MAP[class_idx] = 2
    elif suffix.startswith("02"):
        CLASS_TO_TYPE_MAP[class_idx] = 3
    elif suffix.startswith("03"):
        CLASS_TO_TYPE_MAP[class_idx] = 4

FILENAME_PATTERN = re.compile(
    r"Texture\((\d{3})\)\]\[Speed\((\d+)\)\]\[Force\(([0-9.]+)\)\]\[Trial\((\d+)\)"
)

def parse_texture_code_from_filename(fname: str) -> str:
    """Return the texture code encoded in a spike-train filename."""
    m = FILENAME_PATTERN.search(fname)
    return m.group(1) if m else "UNK"

def gather_all_spike_files(spike_root: Path) -> List[Path]:
    """Return sorted spike-train files under the spike-train root."""
    if not spike_root.is_dir():
        return []

    files: List[Path] = []
    for tex_dir in spike_root.iterdir():
        if tex_dir.is_dir() and tex_dir.name.startswith("Texture"):
            for f in tex_dir.iterdir():
                if f.is_file() and f.suffix == ".npz" and FILENAME_PATTERN.search(f.name):
                    files.append(f)

    files.sort(key=lambda p: str(p))
    return files

def load_spikes_npz(path: Path, input_size: int) -> Optional[torch.Tensor]:
    """
    Load one spike-train npz file as a [C, T] tensor.
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            if "spikes" not in data:
                return None
            sp = data["spikes"]
            sp = sp.astype(bool)
            if sp.ndim != 2 or sp.shape[1] < input_size:
                return None
            sp = sp[:, :input_size]
            return torch.from_numpy(sp.T)
    except Exception:
        return None

class SpikeDataset(Dataset):
    """Dataset returning (spikes[C,T], label[int]) for each trial path."""

    def __init__(self, file_paths: List[Path], input_size: int):
        """Initialize the dataset."""
        self.file_paths = file_paths
        self.input_size = input_size

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], int]:
        """Return one spike tensor and label."""
        fpath = self.file_paths[idx]
        tex = parse_texture_code_from_filename(fpath.name)
        if tex not in TEXTURE_LABEL_MAP:
            return None, -1
        spikes = load_spikes_npz(fpath, self.input_size)
        return spikes, TEXTURE_LABEL_MAP[tex]

def raw_spikes_collate_fn(
    batch: List[Tuple[Optional[torch.Tensor], int]],
) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    """
    Collate variable-length spike tensors without padding.
    """
    spikes_list: List[torch.Tensor] = []
    labels: List[int] = []

    for spikes, y in batch:
        if spikes is None or y < 0:
            continue
        spikes_list.append(spikes)
        labels.append(y)

    if not spikes_list:
        return None, None

    return spikes_list, torch.tensor(labels, dtype=torch.long)

def _npz_get_first(existing: Dict[str, Any], keys: List[str]) -> Optional[np.ndarray]:
    """Return the first matching array from an npz mapping."""
    for k in keys:
        if k in existing:
            return existing[k]
    return None

def load_split(
    split_path: Path,
    all_files_sorted: List[Path],
    spike_root: Path,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Load split.npz and return train/val/test file lists.

    Supported split.npz formats are index arrays or file-list arrays. File-list entries may be absolute paths or paths relative to the spike-train root.
    """
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with np.load(split_path, allow_pickle=True) as z:
        train_idx = _npz_get_first(z, ["train_idx", "train_indices"])
        val_idx = _npz_get_first(z, ["val_idx", "val_indices", "valid_idx", "valid_indices"])
        test_idx = _npz_get_first(z, ["test_idx", "test_indices"])

        train_files_arr = _npz_get_first(z, ["train_files", "train_paths"])
        val_files_arr = _npz_get_first(z, ["val_files", "val_paths", "valid_files", "valid_paths"])
        test_files_arr = _npz_get_first(z, ["test_files", "test_paths"])

    if train_idx is not None and val_idx is not None and test_idx is not None:
        train_files = [all_files_sorted[int(i)] for i in train_idx]
        val_files = [all_files_sorted[int(i)] for i in val_idx]
        test_files = [all_files_sorted[int(i)] for i in test_idx]
        return train_files, val_files, test_files

    if train_files_arr is not None and val_files_arr is not None and test_files_arr is not None:
        def to_path_list(arr: np.ndarray) -> List[Path]:
            """Convert stored split entries into concrete file paths."""
            out: List[Path] = []
            for item in arr.tolist():
                p = Path(item)
                if not p.is_absolute():
                    p = spike_root / p
                out.append(p)
            return out

        train_files = to_path_list(train_files_arr)
        val_files = to_path_list(val_files_arr)
        test_files = to_path_list(test_files_arr)
        return train_files, val_files, test_files

    raise ValueError(
        "split.npz did not contain a recognized split format. "
        "Expected indices (train_idx/val_idx/test_idx) or files (train_files/val_files/test_files)."
    )

def first_spike_cutoff(spikes_ct: torch.Tensor) -> Optional[int]:
    """Return the cutoff where all channels have spiked at least once."""
    if spikes_ct.ndim != 2:
        raise ValueError(f"Expected spikes [C,T], got {tuple(spikes_ct.shape)}")

    c, _t = spikes_ct.shape
    first_times: List[int] = []
    for ch in range(c):
        idx = torch.where(spikes_ct[ch])[0]
        if idx.numel() == 0:
            return None
        first_times.append(int(idx[0].item()))
    return max(first_times) + 1

def truncate_to_first_spike(spikes_ct: torch.Tensor) -> Optional[torch.Tensor]:
    """Truncate a spike train to the strict first-spike cutoff."""
    cutoff = first_spike_cutoff(spikes_ct)
    if cutoff is None:
        return None
    return spikes_ct[:, :cutoff]

def spikes_to_rate_windows(
    spikes_ct: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor:
    """Convert a spike train into overlapping rate-coded windows."""
    if spikes_ct.ndim != 2:
        raise ValueError(f"Expected spikes [C,T], got {tuple(spikes_ct.shape)}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")

    c, t = spikes_ct.shape

    starts = list(range(0, max(t, 1), stride))
    rates: List[torch.Tensor] = []

    for s in starts:
        e = s + window_size
        if s < t:
            seg = spikes_ct[:, s:min(e, t)]
        else:
            seg = spikes_ct[:, 0:0]

        seg_len = seg.shape[1]
        if seg_len < window_size:
            seg = nn.functional.pad(seg, (0, window_size - seg_len))

        rate = seg.float().sum(dim=1) / max(seg_len, 1)
        rates.append(rate)

        if e >= t:
            break

    return torch.stack(rates, dim=0)

def batch_to_window_tensor_first_spike(
    spikes_list: List[torch.Tensor],
    window_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """Apply first-spike truncation and convert a batch into rate windows."""
    windows: List[torch.Tensor] = []
    counts: List[int] = []
    kept_indices: List[int] = []
    kept_lengths: List[int] = []

    for i, spikes_ct in enumerate(spikes_list):
        fs = truncate_to_first_spike(spikes_ct)
        if fs is None:
            continue
        kept_lengths.append(int(fs.shape[1]))
        w = spikes_to_rate_windows(fs, window_size=window_size, stride=stride)
        counts.append(w.shape[0])
        windows.append(w)
        kept_indices.append(i)

    if not windows:
        return None, None, None, None

    Xw = torch.cat(windows, dim=0).to(device, non_blocking=True)
    return Xw, counts, kept_indices, kept_lengths

class RateCodedSNN(nn.Module):
    """Fully connected rate-coded SNN used for window-level classification."""

    def __init__(
        self,
        input_size: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        beta: float,
        num_steps: int,
    ):
        """Initialize the dataset."""
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class spike-count outputs for a model input."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        bsz = x.shape[0]
        spk_sum = torch.zeros(bsz, NUM_CLASSES, device=x.device)

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_sum += spk3

        return spk_sum

@torch.no_grad()
def batch_metrics(
    spike_counts: torch.Tensor,
    targets: torch.Tensor,
    class_to_type_map: torch.Tensor,
) -> Tuple[int, int, int]:
    """Return sample count, class-correct count, and type-correct count."""
    pred = spike_counts.argmax(dim=1)
    correct = (pred == targets).sum().item()

    true_types = class_to_type_map[targets]
    pred_types = class_to_type_map[pred]
    correct_type = (pred_types == true_types).sum().item()

    return targets.size(0), correct, correct_type

def ensure_dir(p: Path) -> None:
    """Create a directory and parent directories when needed."""
    p.mkdir(parents=True, exist_ok=True)

def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    class_to_type_map: torch.Tensor,
    window_size: int,
    stride: int,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float, float]:
    """Train the model for one epoch and return loss and accuracy metrics."""
    model.train()
    running_loss = 0.0
    running_weight = 0.0
    total = 0
    correct = 0
    correct_type = 0

    num_batches = len(loader)
    pbar = tqdm(
        loader,
        total=num_batches,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=True,
        ncols=120,
        mininterval=0.0,
    )

    for _, (spikes_list, y) in enumerate(pbar, start=1):
        if spikes_list is None:
            continue

        Xw, counts, kept, kept_lengths = batch_to_window_tensor_first_spike(
            spikes_list,
            window_size=window_size,
            stride=stride,
            device=device,
        )
        if Xw is None:
            continue

        y = y.to(device, non_blocking=True)
        y = y[torch.tensor(kept, dtype=torch.long, device=device)]
        lengths = torch.tensor(kept_lengths, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)

        out_w = model(Xw)
        out_chunks = out_w.split(counts, dim=0)
        out = torch.stack([c.mean(dim=0) for c in out_chunks], dim=0)

        loss_vec = loss_fn(out, y)
        weights = 1.0 / lengths.float().clamp(min=1)
        loss = (loss_vec * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()

        running_loss += (loss_vec.detach() * weights).sum().item()
        running_weight += weights.sum().item()
        n, c, ct = batch_metrics(out.detach(), y, class_to_type_map)
        total += n
        correct += c
        correct_type += ct

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * c / max(1, n):.2f}%",
            type=f"{100.0 * ct / max(1, n):.2f}%",
            refresh=True,
        )

    pbar.close()

    avg_loss = running_loss / max(1e-12, running_weight)
    acc = 100.0 * correct / max(1, total)
    type_acc = 100.0 * correct_type / max(1, total)
    return avg_loss, acc, type_acc

@torch.no_grad()
def run_one_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_to_type_map: torch.Tensor,
    window_size: int,
    stride: int,
) -> Tuple[float, float, float]:
    """Evaluate the model for one epoch and return loss and accuracy metrics."""
    model.eval()
    running_loss = 0.0
    running_weight = 0.0
    total = 0
    correct = 0
    correct_type = 0

    for spikes_list, y in loader:
        if spikes_list is None:
            continue

        Xw, counts, kept, kept_lengths = batch_to_window_tensor_first_spike(
            spikes_list,
            window_size=window_size,
            stride=stride,
            device=device,
        )
        if Xw is None:
            continue

        y = y.to(device, non_blocking=True)
        y = y[torch.tensor(kept, dtype=torch.long, device=device)]
        lengths = torch.tensor(kept_lengths, dtype=torch.long, device=device)

        out_w = model(Xw)
        out_chunks = out_w.split(counts, dim=0)
        out = torch.stack([c.mean(dim=0) for c in out_chunks], dim=0)

        loss_vec = loss_fn(out, y)
        weights = 1.0 / lengths.float().clamp(min=1)
        loss = (loss_vec * weights).sum() / weights.sum()

        running_loss += (loss_vec * weights).sum().item()
        running_weight += weights.sum().item()
        n, c, ct = batch_metrics(out, y, class_to_type_map)
        total += n
        correct += c
        correct_type += ct

    avg_loss = running_loss / max(1e-12, running_weight)
    acc = 100.0 * correct / max(1, total)
    type_acc = 100.0 * correct_type / max(1, total)
    return avg_loss, acc, type_acc

def main() -> None:
    """Train the classifier and save model weights and performance metrics."""
    set_seed(SEED)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    spike_root = project_root / "data" / "spike_trains"
    split_path = project_root / "data" / "splits" / "split.npz"

    out_root = script_path.parent
    perf_acc_dir = out_root / "Performance" / "Accuracy"
    perf_type_dir = out_root / "Performance" / "TypeAccuracy"
    perf_loss_dir = out_root / "Performance" / "Loss"
    model_dir = out_root / "Models"

    ensure_dir(perf_acc_dir)
    ensure_dir(perf_type_dir)
    ensure_dir(perf_loss_dir)
    ensure_dir(model_dir)

    INPUT_SIZE = 9
    H1 = 1000
    H2 = 1000
    BETA = 0.95
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LR = 2e-4
    PLATEAU_PATIENCE = 3
    PLATEAU_FACTOR = 0.5

    WINDOW_SIZE = 150
    STRIDE = 90
    NUM_STEPS = 150

    num_workers = min(8, os.cpu_count() or 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = bool(num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Using device: {device}")
    print(f"Spike root: {spike_root}")
    print(f"Split path: {split_path}")
    print(f"Window size: {WINDOW_SIZE} | Stride: {STRIDE} (40% overlap)")
    print("Input data: FIRST-SPIKE truncated (all channels must spike at least once)")

    all_files = gather_all_spike_files(spike_root)
    if not all_files:
        sys.exit(f"No spike .npz files found under: {spike_root}")

    train_files, val_files, test_files = load_split(split_path, all_files, spike_root)
    print(f"Split sizes: train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")

    train_ds = SpikeDataset(train_files, INPUT_SIZE)
    val_ds = SpikeDataset(val_files, INPUT_SIZE)
    test_ds = SpikeDataset(test_files, INPUT_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=raw_spikes_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=raw_spikes_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=raw_spikes_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
    )

    model = RateCodedSNN(INPUT_SIZE, H1, H2, NUM_CLASSES, BETA, NUM_STEPS).to(device)
    class_to_type_map = CLASS_TO_TYPE_MAP.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.0,
    )

    train_acc = np.zeros(NUM_EPOCHS, dtype=np.float32)
    val_acc = np.zeros(NUM_EPOCHS, dtype=np.float32)

    train_type_acc = np.zeros(NUM_EPOCHS, dtype=np.float32)
    val_type_acc = np.zeros(NUM_EPOCHS, dtype=np.float32)

    train_loss = np.zeros(NUM_EPOCHS, dtype=np.float32)
    val_loss = np.zeros(NUM_EPOCHS, dtype=np.float32)

    test_metrics = {"test_acc": np.nan, "test_type_acc": np.nan, "test_loss": np.nan}

    print("\nStarting training...")
    weights_path = model_dir / f"Weights_{run_id}.pt"

    for ep in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_type = run_one_epoch_train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            class_to_type_map=class_to_type_map,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            epoch=ep,
            num_epochs=NUM_EPOCHS,
        )

        va_loss, va_acc, va_type = run_one_epoch_eval(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            class_to_type_map=class_to_type_map,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        train_loss[ep - 1] = tr_loss
        val_loss[ep - 1] = va_loss
        train_acc[ep - 1] = tr_acc
        val_acc[ep - 1] = va_acc
        train_type_acc[ep - 1] = tr_type
        val_type_acc[ep - 1] = va_type

        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        torch.save(model.state_dict(), weights_path)

        dt = time.time() - t0
        print(
            f"Epoch {ep:02d}/{NUM_EPOCHS} | "
            f"TR: Loss = {tr_loss:.4f}, Accuracy = {tr_acc:.2f}% | "
            f"V: Loss = {va_loss:.4f}, Accuracy = {va_acc:.2f}% | "
            f"LR = {current_lr:.2e} | {dt:.1f}s"
        )

        acc_path = perf_acc_dir / f"Accuracy_{run_id}.npz"
        type_path = perf_type_dir / f"TypeAccuracy_{run_id}.npz"
        loss_path = perf_loss_dir / f"Loss_{run_id}.npz"

        np.savez_compressed(
            acc_path,
            run_id=run_id,
            epochs=np.arange(1, NUM_EPOCHS + 1),
            train=train_acc,
            val=val_acc,
        )
        np.savez_compressed(
            type_path,
            run_id=run_id,
            epochs=np.arange(1, NUM_EPOCHS + 1),
            train=train_type_acc,
            val=val_type_acc,
        )
        np.savez_compressed(
            loss_path,
            run_id=run_id,
            epochs=np.arange(1, NUM_EPOCHS + 1),
            train=train_loss,
            val=val_loss,
        )

    print("\nFinal test evaluation...")
    te_loss, te_acc, te_type = run_one_epoch_eval(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        class_to_type_map=class_to_type_map,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    test_metrics["test_loss"] = float(te_loss)
    test_metrics["test_acc"] = float(te_acc)
    test_metrics["test_type_acc"] = float(te_type)

    print(f"\nTest: Loss = {te_loss:.4f}, Accuracy = {te_acc:.2f}%, Type Accuracy = {te_type:.2f}%")

    acc_path = perf_acc_dir / f"Accuracy_{run_id}.npz"
    type_path = perf_type_dir / f"TypeAccuracy_{run_id}.npz"
    loss_path = perf_loss_dir / f"Loss_{run_id}.npz"

    np.savez_compressed(
        acc_path,
        run_id=run_id,
        epochs=np.arange(1, NUM_EPOCHS + 1),
        train=train_acc,
        val=val_acc,
        test_acc=test_metrics["test_acc"],
    )
    np.savez_compressed(
        type_path,
        run_id=run_id,
        epochs=np.arange(1, NUM_EPOCHS + 1),
        train=train_type_acc,
        val=val_type_acc,
        test_type_acc=test_metrics["test_type_acc"],
    )
    np.savez_compressed(
        loss_path,
        run_id=run_id,
        epochs=np.arange(1, NUM_EPOCHS + 1),
        train=train_loss,
        val=val_loss,
        test_loss=test_metrics["test_loss"],
    )

    print(f"\nDone. Weights saved at: {weights_path}")
    print(f"Accuracy saved at: {acc_path}")
    print(f"TypeAccuracy saved at: {type_path}")
    print(f"Loss saved at: {loss_path}")

if __name__ == "__main__":
    main()
