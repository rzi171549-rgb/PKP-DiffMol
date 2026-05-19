"""Draft MoleculeNet/LMDB data-path helpers.

The legacy dataset implementation is preserved in
pkpdiffmol/runtime_legacy/original_repo/tasks/modules/chemprior_peak_general.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LMDBSplitPaths:
    train: Path
    valid: Path
    test: Path

    def missing(self) -> list[Path]:
        return [path for path in (self.train, self.valid, self.test) if not path.exists()]


def split_paths(dataset_root: Path) -> LMDBSplitPaths:
    return LMDBSplitPaths(
        train=dataset_root / "train.lmdb",
        valid=dataset_root / "valid.lmdb",
        test=dataset_root / "test.lmdb",
    )

