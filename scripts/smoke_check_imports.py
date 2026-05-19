"""Smoke check for the PKP-DiffMol release draft.

The check parses every Python file in the release tree. It imports only the
lightweight package-facing modules, not runtime_legacy files that may require
torch, fairseq, transformers, rdkit, LMDB data, or local checkpoints.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


LIGHTWEIGHT_IMPORTS = [
    "pkpdiffmol",
    "pkpdiffmol.backbones",
    "pkpdiffmol.backbones.smi_editor_wrapper",
    "pkpdiffmol.data",
    "pkpdiffmol.data.moleculenet",
    "pkpdiffmol.data.scaffold_split",
    "pkpdiffmol.utils",
    "pkpdiffmol.utils.paths",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_python_files(root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}: {exc}")
        except UnicodeDecodeError as exc:
            failures.append(f"{path}: {exc}")
    return failures


def import_lightweight_modules(root: Path) -> list[str]:
    failures: list[str] = []
    sys.path.insert(0, str(root))
    for name in LIGHTWEIGHT_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - smoke script should report all failures.
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def main() -> int:
    root = repo_root()
    parse_failures = parse_python_files(root)
    import_failures = import_lightweight_modules(root)

    print(f"repo_root={root}")
    print(f"python_files={len(list(root.rglob('*.py')))}")
    print(f"parse_failures={len(parse_failures)}")
    print(f"lightweight_import_failures={len(import_failures)}")

    for failure in parse_failures + import_failures:
        print(f"FAIL {failure}")

    return 1 if parse_failures or import_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
