"""
Run hidden-layer-width ablations for every classifier.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


MODEL_PATHS = {
    "SNN": Path("classifiers/SNN/SNN.py"),
    "tSNN": Path("classifiers/tSNN/tSNN.py"),
    "FS-SNN": Path("classifiers/FS-SNN/FS-SNN.py"),
    "FS-tSNN": Path("classifiers/FS-tSNN/FS-tSNN.py"),
    "ANN": Path("classifiers/ANN/ANN.py"),
}

DEFAULT_HIDDEN_SIZES = (100, 200, 500)


class ClassifierPatchError(RuntimeError):
    """Raised when a classifier no longer matches the expected structure."""


class HiddenSizeTransformer(ast.NodeTransformer):
    """Patch H1, H2, and run_id assignments inside a classifier's main()."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.tag = f"_H{hidden_size}"
        self._inside_main = False
        self.replacements = {"H1": 0, "H2": 0, "run_id": 0}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Only transform assignments contained in main()."""
        was_inside_main = self._inside_main
        self._inside_main = node.name == "main"
        node = self.generic_visit(node)
        self._inside_main = was_inside_main
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Do not descend into asynchronous functions nested in main()."""
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Replace the expected simple-name assignments."""
        if not self._inside_main or len(node.targets) != 1:
            return self.generic_visit(node)

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return self.generic_visit(node)

        if target.id in {"H1", "H2"}:
            node.value = ast.copy_location(ast.Constant(self.hidden_size), node.value)
            self.replacements[target.id] += 1
            return node

        if target.id == "run_id":
            tagged_value = ast.BinOp(
                left=node.value,
                op=ast.Add(),
                right=ast.Constant(self.tag),
            )
            node.value = ast.copy_location(tagged_value, node.value)
            self.replacements["run_id"] += 1
            return node

        return self.generic_visit(node)


def project_analysis_dir() -> Path:
    """Return the repository's analysis directory."""
    return Path(__file__).resolve().parents[1]


def classifier_path(model_name: str) -> Path:
    """Return the classifier script path for one model name."""
    path = project_analysis_dir() / MODEL_PATHS[model_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier script for {model_name} was not found: {path}"
        )
    return path


def patched_source(path: Path, hidden_size: int) -> str:
    """Return validated classifier source patched for one hidden width."""
    if hidden_size <= 0:
        raise ValueError(f"Hidden size must be positive, got {hidden_size}.")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    transformer = HiddenSizeTransformer(hidden_size)
    transformed = transformer.visit(tree)
    ast.fix_missing_locations(transformed)

    expected = {"H1": 1, "H2": 1, "run_id": 1}
    if transformer.replacements != expected:
        raise ClassifierPatchError(
            f"Could not safely patch {path}. Expected replacements {expected}, "
            f"found {transformer.replacements}. The classifier structure may "
            "have changed."
        )

    output = ast.unparse(transformed) + "\n"
    compile(output, str(path), "exec")
    return output


def run_classifier(model_name: str, hidden_size: int) -> None:
    """Execute one classifier/width combination in a fresh process."""
    path = classifier_path(model_name)
    source = patched_source(path, hidden_size)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=f"_H{hidden_size}.py",
            prefix=".hidden_size_",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(source)
            temporary_path = Path(temporary_file.name)

        print("\n" + "=" * 80, flush=True)
        print(
            f"Training {model_name} with H1=H2={hidden_size} "
            f"(output tag: _H{hidden_size})",
            flush=True,
        )
        print("=" * 80, flush=True)

        subprocess.run(
            [sys.executable, str(temporary_path)],
            cwd=path.parent,
            check=True,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def unique_positive_sizes(values: Sequence[int]) -> list[int]:
    """Validate widths and return them in first-seen order without duplicates."""
    result: list[int] = []
    seen: set[int] = set()

    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"Hidden sizes must be positive integers; received {value}."
            )
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train classifier width ablations while preserving each existing "
            "trainer's standard hyperparameters and output directories."
        )
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_HIDDEN_SIZES),
        metavar="N",
        help="Hidden widths to test for both H1 and H2 (default: 100 200 500).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_PATHS),
        default=list(MODEL_PATHS),
        help="Classifier models to run (default: all models).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later runs if one model/width combination fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every requested patch without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested model-width grid."""
    args = parse_args()
    hidden_sizes = unique_positive_sizes(args.hidden_sizes)

    jobs = [
        (model_name, hidden_size)
        for model_name in args.models
        for hidden_size in hidden_sizes
    ]

    if args.dry_run:
        for model_name, hidden_size in jobs:
            path = classifier_path(model_name)
            patched_source(path, hidden_size)
            print(f"Validated {model_name}: H1=H2={hidden_size}, tag=_H{hidden_size}")
        print(f"\nDry run complete: {len(jobs)} configuration(s) validated.")
        return

    failures: list[tuple[str, int, str]] = []

    for model_name, hidden_size in jobs:
        try:
            run_classifier(model_name, hidden_size)
        except (OSError, subprocess.CalledProcessError, ClassifierPatchError) as exc:
            failures.append((model_name, hidden_size, str(exc)))
            print(
                f"ERROR: {model_name} H{hidden_size} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                raise SystemExit(1) from exc

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for model_name, hidden_size, message in failures:
            print(
                f"  - {model_name} H{hidden_size}: {message}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    print(f"\nCompleted all {len(jobs)} hidden-size training runs.")


if __name__ == "__main__":
    main()
