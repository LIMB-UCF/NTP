"""
Repeated-seed training for the five standard classifiers.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SEEDS = (1, 2, 3, 4)
DEFAULT_MODELS = ("SNN", "tSNN", "FS-SNN", "FS-tSNN", "ANN")
BASE_HIDDEN_SIZE = 1000


def project_root() -> Path:
    """Return the repository root containing analysis/ and data/."""

    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "analysis").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("Could not locate the project root containing analysis/ and data/.")


def classifier_paths(root: Path) -> dict[str, Path]:
    """Return the expected training-file path for each classifier."""

    classifier_root = root / "analysis" / "classifiers"
    return {
        "SNN": classifier_root / "SNN" / "SNN.py",
        "tSNN": classifier_root / "tSNN" / "tSNN.py",
        "FS-SNN": classifier_root / "FS-SNN" / "FS-SNN.py",
        "FS-tSNN": classifier_root / "FS-tSNN" / "FS-tSNN.py",
        "ANN": classifier_root / "ANN" / "ANN.py",
    }


def split_seed(split_path: Path) -> int | None:
    """Return the seed stored in a split file, if present."""

    try:
        with np.load(split_path, allow_pickle=True) as data:
            if "seed" not in data:
                return None
            values = np.asarray(data["seed"]).reshape(-1)
            return int(values[0]) if len(values) else None
    except Exception:
        return None


def create_or_reuse_split(
    *,
    root: Path,
    seed: int,
    regenerate: bool,
    strict_100: bool,
) -> Path:
    """Create or reuse data/splits/split_S{seed}.npz."""

    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    tagged_split = split_dir / f"split_S{seed}.npz"

    if tagged_split.exists() and not regenerate:
        stored_seed = split_seed(tagged_split)
        if stored_seed != seed:
            raise RuntimeError(
                f"Existing {tagged_split} does not report seed {seed} "
                f"(stored seed: {stored_seed!r}). Use --regenerate-splits to replace it."
            )
        print(f"Reusing split: {tagged_split}")
        return tagged_split

    make_split = root / "analysis" / "preprocessing" / "make_split.py"
    if not make_split.is_file():
        raise FileNotFoundError(f"Could not find split generator: {make_split}")

    command = [
        sys.executable,
        str(make_split),
        "--seed",
        str(seed),
        "--out",
        str(tagged_split),
    ]
    if strict_100:
        command.append("--strict_100")

    print(f"Creating split for seed {seed}: {tagged_split}")
    subprocess.run(command, cwd=root, check=True)

    stored_seed = split_seed(tagged_split)
    if stored_seed != seed:
        raise RuntimeError(
            f"Generated split {tagged_split} did not store the expected seed {seed}."
        )

    return tagged_split


def extract_hidden_sizes(source: str) -> tuple[int, int]:
    """Extract the first H1/H2 integer assignments from a classifier source file."""

    h1_match = re.search(r"(?m)^\s*H1\s*=\s*(\d+)\s*$", source)
    h2_match = re.search(r"(?m)^\s*H2\s*=\s*(\d+)\s*$", source)

    if h1_match is None or h2_match is None:
        raise RuntimeError("Could not locate H1/H2 assignments in classifier source.")

    return int(h1_match.group(1)), int(h2_match.group(1))


def patch_classifier_source(source: str, seed: int) -> str:
    """Return classifier source patched only for RNG seed and output tag."""

    h1, h2 = extract_hidden_sizes(source)
    if h1 != BASE_HIDDEN_SIZE or h2 != BASE_HIDDEN_SIZE:
        raise RuntimeError(
            f"Expected base hidden sizes H1=H2={BASE_HIDDEN_SIZE}, found H1={h1}, H2={h2}. "
            "run_seeds.py is intentionally restricted to the base-size models."
        )

    patched, seed_count = re.subn(
        r"(?m)^SEED\s*=\s*\d+\s*$",
        f"SEED = {seed}",
        source,
        count=1,
    )
    if seed_count != 1:
        raise RuntimeError("Expected exactly one top-level SEED assignment in classifier source.")

    run_id_pattern = (
        r"run_id\s*=\s*datetime\.datetime\.now\(\)\.strftime\("
        r"(?P<quote>['\"])%Y%m%d_%H%M%S(?P=quote)\)"
    )
    replacement = (
        'run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") '
        f'+ "_S{seed}"'
    )
    patched, run_id_count = re.subn(run_id_pattern, replacement, patched, count=1)
    if run_id_count != 1:
        raise RuntimeError(
            "Could not locate the standard run_id timestamp assignment in classifier source."
        )

    return patched


def validate_classifier(path: Path, model_name: str, seed: int) -> None:
    """Check that a classifier can be safely patched for a seeded run."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {model_name} trainer: {path}\n"
            "For ANN, place ANN.py at analysis/classifiers/ANN/ANN.py first."
        )

    source = path.read_text(encoding="utf-8")
    patch_classifier_source(source, seed)


def run_classifier(*, root: Path, model_name: str, source_path: Path, seed: int) -> None:
    """Run one classifier from a temporary seeded copy beside the original file."""

    source = source_path.read_text(encoding="utf-8")
    patched = patch_classifier_source(source, seed)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=f"_S{seed}.py",
            prefix="_run_seeds_",
            dir=source_path.parent,
            delete=False,
        ) as handle:
            handle.write(patched)
            temp_path = Path(handle.name)

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"

        print("\n" + "=" * 78)
        print(f"Training {model_name} | hidden size={BASE_HIDDEN_SIZE} | seed={seed}")
        print(f"Split: data/splits/split_S{seed}.npz")
        print(f"Output tag: _S{seed}")
        print("=" * 78)

        subprocess.run(
            [sys.executable, str(temp_path)],
            cwd=root,
            env=env,
            check=True,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def activate_split(tagged_split: Path, active_split: Path) -> None:
    """Copy a tagged split to the standard split.npz path used by the trainers."""

    active_split.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tagged_split, active_split)


def validate_requested_models(paths: dict[str, Path], models: Iterable[str], seed: int) -> None:
    """Validate all requested trainers before starting expensive experiments."""

    for model_name in models:
        validate_classifier(paths[model_name], model_name, seed)


def main() -> None:
    """Run the repeated-seed experiment."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the five base-width classifiers across matched model/data-split seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to run (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Classifier subset to run (default: all five).",
    )
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Regenerate split_S#.npz even when a matching tagged split already exists.",
    )
    parser.add_argument(
        "--strict-100",
        action="store_true",
        help="Pass --strict_100 to make_split.py when generating splits.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later model/seed combinations if one training run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate trainers and print the planned runs without creating splits or training.",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicate values.")

    root = project_root()
    paths = classifier_paths(root)

    # Validate source structure before spending time generating data splits or training.
    validation_seed = args.seeds[0]
    validate_requested_models(paths, args.models, validation_seed)

    print(f"Project root: {root}")
    print(f"Seeds:        {args.seeds}")
    print(f"Models:       {args.models}")
    print(f"Hidden size:  {BASE_HIDDEN_SIZE}")

    if args.dry_run:
        print("\nDry run only. Planned experiments:")
        for seed in args.seeds:
            for model_name in args.models:
                # Patch each exact seed during dry-run too, so tag substitution is validated.
                source = paths[model_name].read_text(encoding="utf-8")
                patch_classifier_source(source, seed)
                print(
                    f"  seed={seed:<4} model={model_name:<7} "
                    f"split=split_S{seed}.npz output_tag=_S{seed}"
                )
        print(f"\nTotal planned training runs: {len(args.seeds) * len(args.models)}")
        return

    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    active_split = split_dir / "split.npz"

    # Preserve the user's normal active split and restore it even if a child process fails.
    with tempfile.TemporaryDirectory(prefix="ntp_run_seeds_") as backup_dir_name:
        backup_dir = Path(backup_dir_name)
        backup_split = backup_dir / "split.npz"
        had_active_split = active_split.exists()
        if had_active_split:
            shutil.copy2(active_split, backup_split)

        failures: list[tuple[int, str, str]] = []

        try:
            for seed in args.seeds:
                tagged_split = create_or_reuse_split(
                    root=root,
                    seed=seed,
                    regenerate=args.regenerate_splits,
                    strict_100=args.strict_100,
                )
                activate_split(tagged_split, active_split)

                for model_name in args.models:
                    try:
                        run_classifier(
                            root=root,
                            model_name=model_name,
                            source_path=paths[model_name],
                            seed=seed,
                        )
                    except subprocess.CalledProcessError as exc:
                        failures.append((seed, model_name, f"exit code {exc.returncode}"))
                        if not args.continue_on_error:
                            raise
                    except Exception as exc:
                        failures.append((seed, model_name, str(exc)))
                        if not args.continue_on_error:
                            raise
        finally:
            if had_active_split:
                shutil.copy2(backup_split, active_split)
                print(f"\nRestored original active split: {active_split}")
            else:
                active_split.unlink(missing_ok=True)
                print(f"\nRemoved temporary active split: {active_split}")

        print("\nSeed experiment complete.")
        print(f"Tagged splits are stored in: {split_dir}")
        print(f"Successful/attempted runs: {len(args.seeds) * len(args.models) - len(failures)}/"
              f"{len(args.seeds) * len(args.models)}")

        if failures:
            print("Failures:")
            for seed, model_name, reason in failures:
                print(f"  seed={seed}, model={model_name}: {reason}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
