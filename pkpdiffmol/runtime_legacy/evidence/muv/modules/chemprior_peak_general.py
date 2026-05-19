#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TASKS_DIR = os.path.dirname(_MODULE_DIR)
_ROOT_DIR = os.path.dirname(_TASKS_DIR)
_SIDER_SCRIPTS_DIR = os.path.join(_TASKS_DIR, "sider", "scripts")

for _path in (_ROOT_DIR, _MODULE_DIR, _SIDER_SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from checkpoint_adapter import _FinetuneTask, load_smi_editor_dictionary, tokenize_smiles
from fairseq.models.roberta.levenshtein_encoder import LevenshteinEncoderModel
from heads_chemprior_peak import BaselineEquivalentChemPriorHead

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str
    output_dim: int
    batch_size: int
    lr: float
    epochs: int
    pooler_dropout: float
    warmup_ratio: float
    expected_counts: Tuple[int, int, int]

    @property
    def is_multitask(self) -> bool:
        return self.task_type == "multitask"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "bace": DatasetSpec(
        name="bace",
        task_type="binary",
        output_dim=1,
        batch_size=64,
        lr=1e-4,
        epochs=60,
        pooler_dropout=0.2,
        warmup_ratio=0.06,
        expected_counts=(1210, 151, 152),
    ),
    "clintox": DatasetSpec(
        name="clintox",
        task_type="multitask",
        output_dim=2,
        batch_size=256,
        lr=5e-5,
        epochs=100,
        pooler_dropout=0.5,
        warmup_ratio=0.1,
        expected_counts=(1182, 148, 148),
    ),
    "tox21": DatasetSpec(
        name="tox21",
        task_type="multitask",
        output_dim=12,
        batch_size=128,
        lr=1e-4,
        epochs=80,
        pooler_dropout=0.1,
        warmup_ratio=0.06,
        expected_counts=(6264, 783, 784),
    ),
    "toxcast": DatasetSpec(
        name="toxcast",
        task_type="multitask",
        output_dim=617,
        batch_size=64,
        lr=1e-4,
        epochs=80,
        pooler_dropout=0.1,
        warmup_ratio=0.06,
        expected_counts=(6860, 858, 858),
    ),
    "sider": DatasetSpec(
        name="sider",
        task_type="multitask",
        output_dim=27,
        batch_size=32,
        lr=5e-4,
        epochs=100,
        pooler_dropout=0.0,
        warmup_ratio=0.4,
        expected_counts=(1141, 143, 143),
    ),
    "muv": DatasetSpec(
        name="muv",
        task_type="multitask",
        output_dim=17,
        batch_size=128,
        lr=2e-5,
        epochs=40,
        pooler_dropout=0.1,
        warmup_ratio=0.2,
        expected_counts=(74469, 9309, 9309),
    ),
}


def build_parser(dataset_name: str) -> argparse.ArgumentParser:
    spec = DATASET_SPECS[dataset_name]
    script_dir = os.path.join(_TASKS_DIR, dataset_name, "scripts")
    output_ckpt_dir = os.path.join(
        _TASKS_DIR, dataset_name, "outputs", "checkpoints", "baselineeq_chemprior_peak"
    )
    output_log_dir = os.path.join(
        _TASKS_DIR, dataset_name, "outputs", "logs", "baselineeq_chemprior_peak"
    )

    parser = argparse.ArgumentParser(
        description=f"{dataset_name} baselineeq chemprior peak training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_ROOT_DIR, "smi_editor.pt"),
        help="Path to SMI-Editor pretrained checkpoint",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default=os.path.join(_ROOT_DIR, "smi_dict_token.txt"),
        help="Path to smi_dict_token.txt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(_TASKS_DIR, dataset_name, "data"),
        help="Directory containing train.lmdb / valid.lmdb / test.lmdb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=output_ckpt_dir,
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.path.join(
            output_log_dir, f"{dataset_name}_baselineeq_chemprior_peak_seed0.log"
        ),
        help="Training log file",
    )
    parser.add_argument("--epochs", type=int, default=spec.epochs)
    parser.add_argument("--batch-size", type=int, default=spec.batch_size)
    parser.add_argument("--lr", type=float, default=spec.lr)
    parser.add_argument("--warmup-ratio", type=float, default=spec.warmup_ratio)
    parser.add_argument("--pooler-dropout", type=float, default=spec.pooler_dropout)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backbone-weight-decay", type=float, default=None)
    parser.add_argument("--head-weight-decay", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-best-epoch", type=int, default=1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--desc-hidden", type=int, default=64)
    parser.add_argument("--desc-dropout", type=float, default=0.0)
    parser.add_argument("--fusion-dropout", type=float, default=0.0)
    parser.add_argument("--use-desc-layernorm", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument(
        "--script-dir",
        type=str,
        default=script_dir,
        help=argparse.SUPPRESS,
    )
    return parser


def configure_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(checkpoint_path: str, dict_path: str):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"Dictionary not found: {dict_path}")

    LOGGER.info("Loading backbone checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError("Invalid fairseq checkpoint: missing 'model'")
    if "args" not in checkpoint and "cfg" not in checkpoint:
        raise RuntimeError("Invalid fairseq checkpoint: missing args/cfg")

    args = checkpoint.get("args")
    if args is None:
        args = checkpoint.get("cfg", {}).get("model")
        if args is None:
            raise RuntimeError("Unable to recover model args from checkpoint")
        LOGGER.info("ckpt['args'] is None, using ckpt['cfg']['model']")

    state_dict = {
        key: value for key, value in checkpoint["model"].items() if isinstance(value, torch.Tensor)
    }
    dictionary = load_smi_editor_dictionary(dict_path)
    task = _FinetuneTask(dictionary)
    model = LevenshteinEncoderModel.build_model(args, task)
    model.load_state_dict(state_dict, strict=False)

    model_core_keys = {key for key in model.state_dict() if "classification_heads" not in key}
    missing_core = model_core_keys - set(state_dict.keys())
    if missing_core:
        missing_keys = list(missing_core)[:5]
        raise RuntimeError(f"Backbone keys missing after load: {missing_keys}")

    embed_dim = getattr(args, "encoder_embed_dim", 768)
    LOGGER.info("Backbone loaded. encoder_embed_dim=%s", embed_dim)
    return model, dictionary, embed_dim


class MoleculeNetLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, dictionary, task_dim: int, max_len: int = 512):
        import lmdb

        self.lmdb_path = lmdb_path
        self.dictionary = dictionary
        self.task_dim = task_dim
        self.max_len = max_len
        self._lmdb = lmdb
        self._env = None
        self._keys = self._read_keys()
        LOGGER.info("Loaded %s keys from %s", len(self._keys), lmdb_path)

    def _open_env(self):
        return self._lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

    def _read_keys(self):
        env = self._open_env()
        try:
            with env.begin() as txn:
                return list(txn.cursor().iternext(values=False))
        finally:
            env.close()

    def _get_env(self):
        if self._env is None:
            self._env = self._open_env()
        return self._env

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int):
        with self._get_env().begin(write=False) as txn:
            raw = txn.get(self._keys[index])
        item = pickle.loads(raw)
        smiles = item["smi"]
        target_array = np.asarray(item["target"], dtype=np.float32).reshape(-1)
        if target_array.shape[0] != self.task_dim:
            raise ValueError(
                f"{self.lmdb_path}: expected target dim {self.task_dim}, got {target_array.shape[0]}"
            )
        tokens = tokenize_smiles(smiles, self.dictionary, self.max_len)
        target = torch.tensor(target_array, dtype=torch.float32)
        return tokens, target, smiles

    def collect_smiles(self) -> List[str]:
        smiles_list: List[str] = []
        env = self._open_env()
        try:
            with env.begin() as txn:
                for key in self._keys:
                    raw = txn.get(key)
                    item = pickle.loads(raw)
                    smiles_list.append(item["smi"])
        finally:
            env.close()
        return smiles_list

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()


def collate_with_smiles(batch, pad_idx: int):
    tokens_list, targets_list, smiles_list = zip(*batch)
    max_len = max(tokens.size(0) for tokens in tokens_list)
    padded = torch.full((len(tokens_list), max_len), pad_idx, dtype=torch.long)
    for idx, tokens in enumerate(tokens_list):
        padded[idx, : tokens.size(0)] = tokens
    targets = torch.stack(targets_list, dim=0)
    return padded, targets, list(smiles_list)


def get_linear_warmup_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_binary_auc(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    if targets.size == 0:
        return 0.0
    if np.max(targets) == np.min(targets):
        return 0.0
    return float(roc_auc_score(targets, preds))


def compute_multitask_auc(preds: np.ndarray, targets: np.ndarray) -> float:
    task_aucs: List[float] = []
    for task_idx in range(targets.shape[1]):
        target_column = targets[:, task_idx]
        valid_mask = target_column > -0.5
        if int(valid_mask.sum()) == 0:
            continue
        labeled_targets = target_column[valid_mask]
        labeled_preds = preds[:, task_idx][valid_mask]
        if labeled_targets.max() == labeled_targets.min():
            continue
        task_aucs.append(float(roc_auc_score(labeled_targets, labeled_preds)))
    if not task_aucs:
        raise RuntimeError("No valid tasks available for ROC-AUC computation")
    return float(np.mean(task_aucs))


def compute_dataset_auc(spec: DatasetSpec, preds: np.ndarray, targets: np.ndarray) -> float:
    if spec.is_multitask:
        return compute_multitask_auc(preds, targets)
    return compute_binary_auc(preds, targets)


def compute_masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid_mask = targets > -0.5
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[valid_mask], targets[valid_mask], reduction="mean"
    )


def split_decay_param_groups(module: nn.Module):
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if (
            name_lower.endswith("bias")
            or "layernorm.weight" in name_lower
            or name_lower.endswith(".norm.weight")
            or param.ndim == 1
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return decay_params, no_decay_params


def build_optimizer_and_scheduler(args, model: nn.Module, head: nn.Module, train_loader):
    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    head_lr = args.lr if args.head_lr is None else args.head_lr
    backbone_weight_decay = (
        args.weight_decay if args.backbone_weight_decay is None else args.backbone_weight_decay
    )
    head_weight_decay = (
        args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    )

    param_groups = []
    backbone_decay, backbone_no_decay = split_decay_param_groups(model)
    head_decay, head_no_decay = split_decay_param_groups(head)

    if backbone_decay:
        param_groups.append(
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": backbone_weight_decay}
        )
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": head_weight_decay})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)
    return optimizer, scheduler, num_training_steps, num_warmup_steps


def _compute_loss_and_probs(spec: DatasetSpec, logits: torch.Tensor, targets: torch.Tensor):
    if spec.is_multitask:
        loss = compute_masked_bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
    else:
        logits = logits.squeeze(-1)
        targets = targets.squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        probs = torch.sigmoid(logits)
    return loss, probs


def run_epoch(
    spec: DatasetSpec,
    model: nn.Module,
    head: nn.Module,
    loader,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    grad_clip: float = 1.0,
    max_batches: int = 0,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
        head.train()
    else:
        model.eval()
        head.eval()

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    for step_idx, (tokens, targets, smiles_batch) in enumerate(loader, start=1):
        if max_batches > 0 and step_idx > max_batches:
            break
        tokens = tokens.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            hidden_states, _ = model(
                src_tokens=tokens,
                src_lengths=None,
                features_only=True,
                levenshtein=False,
            )
            logits = head(hidden_states, smiles_batch)
            loss, probs = _compute_loss_and_probs(spec, logits, targets)

            if is_train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()), grad_clip
                    )
                optimizer.step()
                scheduler.step()

        total_loss += float(loss.item())
        batch_count += 1
        all_preds.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("No batches were executed")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def build_dataloaders(args, dictionary, spec: DatasetSpec):
    train_dataset = MoleculeNetLMDBDataset(
        os.path.join(args.data_dir, "train.lmdb"), dictionary, spec.output_dim, args.max_len
    )
    valid_dataset = MoleculeNetLMDBDataset(
        os.path.join(args.data_dir, "valid.lmdb"), dictionary, spec.output_dim, args.max_len
    )
    test_dataset = MoleculeNetLMDBDataset(
        os.path.join(args.data_dir, "test.lmdb"), dictionary, spec.output_dim, args.max_len
    )

    expected_train, expected_valid, expected_test = spec.expected_counts
    assert len(train_dataset) == expected_train, (
        f"{spec.name}: expected {expected_train} train samples, got {len(train_dataset)}"
    )
    assert len(valid_dataset) == expected_valid, (
        f"{spec.name}: expected {expected_valid} valid samples, got {len(valid_dataset)}"
    )
    assert len(test_dataset) == expected_test, (
        f"{spec.name}: expected {expected_test} test samples, got {len(test_dataset)}"
    )

    pad_idx = dictionary.pad_index
    collate_fn = lambda batch: collate_with_smiles(batch, pad_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def log_run_header(args, spec: DatasetSpec, device: torch.device):
    LOGGER.info("=" * 72)
    LOGGER.info("%s baselineeq chemprior peak", spec.name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  python:              %s", sys.executable)
    LOGGER.info("  checkpoint:          %s", args.checkpoint)
    LOGGER.info("  dict:                %s", args.dict)
    LOGGER.info("  data_dir:            %s", args.data_dir)
    LOGGER.info("  output_dir:          %s", args.output_dir)
    LOGGER.info("  log_file:            %s", args.log_file)
    LOGGER.info("  device:              %s", device)
    LOGGER.info("  seed:                %s", args.seed)
    LOGGER.info("  task_type:           %s", spec.task_type)
    LOGGER.info("  output_dim:          %s", spec.output_dim)
    LOGGER.info("  epochs:              %s", args.epochs)
    LOGGER.info("  batch_size:          %s", args.batch_size)
    LOGGER.info("  lr:                  %s", args.lr)
    LOGGER.info("  warmup_ratio:        %s", args.warmup_ratio)
    LOGGER.info("  pooler_dropout:      %s", args.pooler_dropout)
    LOGGER.info("  desc_hidden:         %s", args.desc_hidden)
    LOGGER.info("  desc_dropout:        %s", args.desc_dropout)
    LOGGER.info("  fusion_dropout:      %s", args.fusion_dropout)
    LOGGER.info("  use_desc_layernorm:  %s", args.use_desc_layernorm)
    LOGGER.info("  weight_decay:        %s", args.weight_decay)
    LOGGER.info("  backbone_weight_decay:%s", args.backbone_weight_decay)
    LOGGER.info("  head_weight_decay:   %s", args.head_weight_decay)
    LOGGER.info("  backbone_lr:         %s", args.backbone_lr)
    LOGGER.info("  head_lr:             %s", args.head_lr)
    LOGGER.info("  grad_clip:           %s", args.grad_clip)
    LOGGER.info("  freeze_backbone:     %s", args.freeze_backbone)
    LOGGER.info("  max_train_batches:   %s", args.max_train_batches)
    LOGGER.info("  max_valid_batches:   %s", args.max_valid_batches)
    LOGGER.info("  max_test_batches:    %s", args.max_test_batches)


def run_chemprior_peak_experiment(dataset_name: str):
    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(args.log_file)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_run_header(args, spec, device)

    LOGGER.info("[1] Loading backbone...")
    model, dictionary, embed_dim = load_backbone(args.checkpoint, args.dict)
    model = model.to(device)

    LOGGER.info("[2] Building chemprior peak head...")
    head = BaselineEquivalentChemPriorHead(
        in_dim=embed_dim,
        out_dim=spec.output_dim,
        desc_hidden=args.desc_hidden,
        dropout=args.pooler_dropout,
        desc_dropout=args.desc_dropout,
        fusion_dropout=args.fusion_dropout,
        use_desc_layernorm=args.use_desc_layernorm,
    ).to(device)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    backbone_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    LOGGER.info("  backbone params: %s", f"{backbone_params:,}")
    LOGGER.info("  head params:     %s", f"{head_params:,}")
    LOGGER.info("  total params:    %s", f"{backbone_params + head_params:,}")

    LOGGER.info("[3] Loading LMDB data...")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader,
    ) = build_dataloaders(args, dictionary, spec)
    LOGGER.info(
        "  split sizes: train=%s valid=%s test=%s",
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )

    LOGGER.info("[4] Fitting descriptor normalizer on train split...")
    train_smiles = train_dataset.collect_smiles()
    head.fit_normalizer(train_smiles, device)

    LOGGER.info("[5] Building optimizer and scheduler...")
    optimizer, scheduler, num_training_steps, num_warmup_steps = build_optimizer_and_scheduler(
        args, model, head, train_loader
    )
    LOGGER.info("  total steps:  %s", num_training_steps)
    LOGGER.info("  warmup steps: %s", num_warmup_steps)

    best_val_auc = float("-inf")
    best_epoch = 0
    best_ckpt_path = os.path.join(
        args.output_dir, f"{dataset_name}_baselineeq_chemprior_peak_best.pt"
    )

    LOGGER.info("[6] Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = run_epoch(
            spec,
            model,
            head,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )
        val_loss, val_auc = run_epoch(
            spec,
            model,
            head,
            valid_loader,
            device,
            optimizer=None,
            scheduler=None,
            max_batches=args.max_valid_batches,
        )
        LOGGER.info(
            "Epoch %3d/%d  train_loss=%.4f  train_auc=%.4f  val_loss=%.4f  val_auc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
        )
        if epoch >= args.min_best_epoch and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            LOGGER.info("  *** New best val_auc=%.4f at epoch %d ***", val_auc, epoch)

    LOGGER.info("[7] Evaluating best checkpoint on test split...")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    test_loss, test_auc = run_epoch(
        spec,
        model,
        head,
        test_loader,
        device,
        optimizer=None,
        scheduler=None,
        max_batches=args.max_test_batches,
    )

    LOGGER.info("=" * 72)
    LOGGER.info("%s final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  Best epoch:   %s", best_epoch)
    LOGGER.info("  Best val AUC: %.4f", best_val_auc)
    LOGGER.info("  Test AUC:     %.4f", test_auc)
    LOGGER.info("  Best ckpt:    %s", best_ckpt_path)
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()

    return {
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "best_ckpt_path": best_ckpt_path,
    }


def main(dataset_name: str):
    return run_chemprior_peak_experiment(dataset_name)
