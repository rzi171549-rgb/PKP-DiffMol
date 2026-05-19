#!/usr/bin/env bash
set -euo pipefail

# Dry-run only. Remove --dry-run from an individual command after assets and
# data/<dataset> LMDB splits are prepared to launch real training.

python scripts/train.py --config configs/final/bace.yaml --dry-run
python scripts/train.py --config configs/final/bbbp.yaml --dry-run
python scripts/train.py --config configs/final/muv.yaml --dry-run
python scripts/train.py --config configs/final/sider.yaml --dry-run
python scripts/train.py --config configs/final/tox21.yaml --dry-run
python scripts/train.py --config configs/final/toxcast.yaml --dry-run
python scripts/train.py --config configs/final/clintox.yaml --dry-run
