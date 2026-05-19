"""Path helpers for the release draft."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_legacy_root() -> Path:
    return project_root() / "pkpdiffmol" / "runtime_legacy"

