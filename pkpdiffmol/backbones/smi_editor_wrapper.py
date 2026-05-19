"""Lightweight SMI-Editor asset path helpers.

The executable legacy checkpoint-loading code is preserved under
pkpdiffmol/runtime_legacy/original_repo/tasks/sider/scripts/checkpoint_adapter.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SMIEditorAssets:
    checkpoint: Path
    dictionary: Path

    def missing(self) -> list[Path]:
        return [path for path in (self.checkpoint, self.dictionary) if not path.exists()]


def default_assets(asset_root: Path) -> SMIEditorAssets:
    return SMIEditorAssets(
        checkpoint=asset_root / "smi_editor.pt",
        dictionary=asset_root / "smi_dict_token.txt",
    )

