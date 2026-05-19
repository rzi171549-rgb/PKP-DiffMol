#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BBBP 最小可行 Exp-1：Baseline + Diff-Cls

阶段 A：
    先训练 baseline 并联 encoder + classifier。
阶段 B：
    冻结 encoder，训练 latent diffusion，并只对 fused latent 建模。
阶段 C：
    用 real + synthetic fused features 训练 / 微调最终分类头 out_proj。

关键约束：
1. 只做单任务二分类；
2. diffusion 只吃最终 fused representation；
3. 推理阶段仍然走原并联 encoder -> classifier；
4. 不做 QGate / Priority / 在线 refiner / 多任务。
"""

import logging
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SMI_EDITOR_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_SIDER_SCRIPTS = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../sider/scripts"))
_MODULES_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../modules"))

sys.path.insert(0, _SMI_EDITOR_ROOT)
sys.path.insert(0, _SIDER_SCRIPTS)
sys.path.insert(0, _MODULES_DIR)

from heads_chemprior_peak_qsarprompt_semantic_tunev2_latentdiff import (  # noqa: E402
    PoolPoolerBaseTuneV2LatentDiffHead,
)
from latent_diffusion_core import ContinuousLatentDiffusion  # noqa: E402
from latent_diffusion_denoiser import LatentDiffusionMLPDenoiser  # noqa: E402
from latent_qgate import (  # noqa: E402
    apply_qgate,
    compute_class_stats_for_qgate,
)
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (  # noqa: E402
    BBBPLMDBDataset,
    build_optimizer_param_groups,
    collate_fn,
    compute_best_score,
    compute_roc_auc,
    evaluate,
    load_backbone,
    post_optimizer_step,
    prepare_epoch_training_policy,
    save_json,
    str2bool,
    summarize_optimizer_groups,
    train_one_epoch,
)
from train_bbbp_pool_pooler_base_tunev2 import (  # noqa: E402
    build_parser as build_tunev2_parser,
    build_scheduler,
    should_replace_best,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pool_pooler_base_tunev2_latentdiff_exp1"
# 中文注释：第一版 Exp-1 将扩散步数硬锁死为 50，不允许通过命令行覆盖。
STAGE_B_NUM_TIMESTEPS = 50


def build_parser():
    """
    直接复用旧 tunev2 parser，并补充 Stage B / C 相关参数。
    """
    parser = build_tunev2_parser()
    parser.description = "BBBP 最小可行 Exp-1: Baseline + Diff-Cls"
    parser.set_defaults(
        output_dir=os.path.join(_SCRIPT_DIR, "../outputs/pool_pooler_base_tunev2_latentdiff_exp1")
    )

    parser.add_argument("--diffusion-epochs", type=int, default=30)
    parser.add_argument("--diffusion-batch-size", type=int, default=256)
    parser.add_argument("--diffusion-lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-weight-decay", type=float, default=1e-4)
    parser.add_argument("--diffusion-grad-clip", type=float, default=1.0)
    parser.add_argument("--diffusion-min-best-epoch", type=int, default=1)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=256)
    parser.add_argument("--diffusion-cond-dim", type=int, default=128)
    parser.add_argument("--diffusion-time-embed-dim", type=int, default=128)
    parser.add_argument("--diffusion-num-blocks", type=int, default=4)
    parser.add_argument("--diffusion-dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-beta-schedule", type=str, choices=["cosine"], default="cosine")
    parser.add_argument("--diffusion-prediction-type", type=str, choices=["v"], default="v")

    parser.add_argument("--synthetic-rho", type=float, default=1.0)
    parser.add_argument(
        "--use-qgate",
        "--use_qgate",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--qgate-quantile", type=float, default=0.95)
    parser.add_argument("--qgate-cov-eps", type=float, default=1e-6)

    parser.add_argument("--stage-c-epochs", type=int, default=15)
    parser.add_argument("--stage-c-batch-size", type=int, default=256)
    parser.add_argument("--stage-c-lr", type=float, default=3e-4)
    parser.add_argument("--stage-c-weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-c-grad-clip", type=float, default=1.0)
    parser.add_argument("--stage-c-min-best-epoch", type=int, default=1)
    return parser


def resolve_run_paths(args) -> Dict[str, str]:
    """
    为 Exp-1 构建独立输出目录。
    """
    root_output_dir = os.path.abspath(args.output_dir)
    run_name = f"{EXPERIMENT_NAME}_seed{args.seed}"
    run_dir = os.path.join(root_output_dir, run_name)
    return {
        "root_output_dir": root_output_dir,
        "run_name": run_name,
        "run_dir": run_dir,
        "log_file": os.path.join(run_dir, f"{run_name}.log"),
        "stage_a_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_a_best.pt"),
        "stage_b_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_b_diffusion_best.pt"),
        "stage_c_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_c_out_proj_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """
    批量控制模块参数是否参与训练。
    """
    for parameter in module.parameters():
        parameter.requires_grad = bool(flag)


def build_criterion(args, device: torch.device) -> nn.Module:
    """
    统一构建 BCEWithLogitsLoss。
    """
    if args.pos_weight is None:
        return nn.BCEWithLogitsLoss(reduction="mean")
    return nn.BCEWithLogitsLoss(
        reduction="mean",
        pos_weight=torch.tensor([args.pos_weight], device=device, dtype=torch.float32),
    )


def build_data_loaders(args, dictionary):
    """
    复用旧 BBBP LMDB 数据读取与 collate 逻辑。
    """
    pad_idx = dictionary.pad_index
    train_dataset = BBBPLMDBDataset(
        os.path.join(args.data_dir, "train.lmdb"),
        dictionary,
        args.max_len,
    )
    valid_dataset = BBBPLMDBDataset(
        os.path.join(args.data_dir, "valid.lmdb"),
        dictionary,
        args.max_len,
    )
    test_dataset = BBBPLMDBDataset(
        os.path.join(args.data_dir, "test.lmdb"),
        dictionary,
        args.max_len,
    )

    collate = lambda batch: collate_fn(batch, pad_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def run_stage_a_baseline(
    model,
    head,
    train_loader,
    valid_loader,
    criterion,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage A：尽量复用旧 tunev2 训练逻辑。
    """
    if args.freeze_backbone:
        set_requires_grad(model, False)

    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("Stage A 未能构建出任何可训练参数组。")

    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    optimizer_group_summary = summarize_optimizer_groups(optimizer)
    effective_min_best_epoch = min(max(1, int(args.min_best_epoch)), int(args.epochs))
    best_state = None
    history: List[Dict[str, Any]] = []

    logger.info("[Stage A] 开始训练 baseline 主链。")
    for group_info in optimizer_group_summary:
        logger.info(
            "[Stage A] group=%s lr=%g wd=%g params=%d",
            group_info["name"],
            group_info["lr"],
            group_info["weight_decay"],
            group_info["param_count"],
        )

    for epoch in range(1, args.epochs + 1):
        epoch_policy = prepare_epoch_training_policy(head, epoch, args)
        train_loss, train_auc = train_one_epoch(
            model=model,
            head=head,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_loss, val_auc = evaluate(
            model=model,
            head=head,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        score_info = compute_best_score(args, val_auc=val_auc, val_loss=val_loss)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "best_score": float(score_info["score"]),
                "hybrid_score": float(score_info["hybrid_score"]),
                "alpha_runtime_scale": float(epoch_policy["alpha_runtime_scale"]),
                "raw_alpha": float(epoch_policy["raw_alpha"]),
                "effective_alpha": float(epoch_policy["effective_alpha"]),
            }
        )

        logger.info(
            "[Stage A] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
        )

        if epoch >= effective_min_best_epoch and should_replace_best(
            score_info=score_info,
            val_auc=val_auc,
            val_loss=val_loss,
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "best_score": float(score_info["score"]),
                "best_metric_value": float(score_info["metric_value"]),
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
                "best_hybrid_score": float(score_info["hybrid_score"]),
            }
            torch.save(
                {
                    "backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "epoch": int(epoch),
                    "stage": "A",
                    "experiment": EXPERIMENT_NAME,
                    "best_metric": args.best_metric,
                    "best_score": float(best_state["best_score"]),
                    "best_metric_value": float(best_state["best_metric_value"]),
                },
                run_paths["stage_a_ckpt_path"],
            )
            logger.info(
                "[Stage A] 新 best checkpoint: epoch=%d val_auc=%.4f val_loss=%.4f",
                epoch,
                val_auc,
                val_loss,
            )

    if best_state is None:
        raise RuntimeError("Stage A 训练结束但没有生成 best checkpoint。")

    ckpt = torch.load(run_paths["stage_a_ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    post_optimizer_step(head)

    return {
        "best_state": best_state,
        "history": history,
        "optimizer_groups": optimizer_group_summary,
        "best_ckpt_path": run_paths["stage_a_ckpt_path"],
    }


@torch.no_grad()
def extract_fused_bank(
    model,
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从冻结后的 backbone + head 中抽取 fused latent bank。
    """
    model.eval()
    head.eval()

    fused_list = []
    label_list = []
    for tokens, targets, smiles_batch in loader:
        tokens = tokens.to(device)
        x, _ = model(
            src_tokens=tokens,
            src_lengths=None,
            features_only=True,
            levenshtein=False,
        )
        fused = head.extract_fused_latent(x, smiles=smiles_batch, detach=True)
        fused_list.append(fused.cpu())
        label_list.append(targets.long().cpu())

    return torch.cat(fused_list, dim=0), torch.cat(label_list, dim=0)


def build_train_class_counts(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    只基于 train split 标签统计类别计数。
    """
    labels = labels.view(-1).long()
    counts = torch.bincount(labels, minlength=num_classes)
    return counts.long()


def build_synthetic_class_counts(train_class_counts: torch.Tensor, rho: float) -> torch.Tensor:
    """
    根据 train split 的类别分布和 rho 生成 synthetic 样本配额。
    """
    total_train = int(train_class_counts.sum().item())
    total_synth = int(round(float(rho) * float(total_train)))
    if total_synth <= 0:
        return torch.zeros_like(train_class_counts)

    probs = train_class_counts.float() / max(1.0, float(total_train))
    raw = probs * float(total_synth)
    synth_counts = torch.floor(raw).long()

    remainder = total_synth - int(synth_counts.sum().item())
    if remainder > 0:
        frac = raw - synth_counts.float()
        order = torch.argsort(frac, descending=True)
        for index in order[:remainder]:
            synth_counts[int(index.item())] += 1

    return synth_counts.long()


def train_diffusion_stage(
    latent_dim: int,
    fused_train: torch.Tensor,
    y_train: torch.Tensor,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage B：冻结 encoder 后训练 latent diffusion。
    """
    denoiser = LatentDiffusionMLPDenoiser(
        latent_dim=latent_dim,
        num_classes=2,
        time_embed_dim=args.diffusion_time_embed_dim,
        cond_dim=args.diffusion_cond_dim,
        hidden_dim=args.diffusion_hidden_dim,
        num_blocks=args.diffusion_num_blocks,
        dropout=args.diffusion_dropout,
    )
    diffusion = ContinuousLatentDiffusion(
        denoiser=denoiser,
        latent_dim=latent_dim,
        num_timesteps=STAGE_B_NUM_TIMESTEPS,
        beta_schedule=args.diffusion_beta_schedule,
        prediction_type=args.diffusion_prediction_type,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(fused_train.float(), y_train.long()),
        batch_size=args.diffusion_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float(), y_valid.long()),
        batch_size=args.diffusion_batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        diffusion.parameters(),
        lr=args.diffusion_lr,
        weight_decay=args.diffusion_weight_decay,
    )

    best_val_loss = None
    best_state = None
    history: List[Dict[str, Any]] = []
    effective_min_best_epoch = min(
        max(1, int(args.diffusion_min_best_epoch)),
        int(args.diffusion_epochs),
    )

    logger.info("[Stage B] 开始训练 diffusion 分支。")
    for epoch in range(1, args.diffusion_epochs + 1):
        diffusion.train()
        train_losses = []

        for z0, labels in train_loader:
            z0 = z0.to(device)
            labels = labels.to(device)
            loss_dict = diffusion.training_loss(z0=z0, c=labels)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.diffusion_grad_clip is not None and args.diffusion_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(),
                    max_norm=args.diffusion_grad_clip,
                )
            optimizer.step()
            train_losses.append(float(loss.item()))

        diffusion.eval()
        valid_losses = []
        with torch.no_grad():
            for z0, labels in valid_loader:
                z0 = z0.to(device)
                labels = labels.to(device)
                loss_dict = diffusion.training_loss(z0=z0, c=labels)
                valid_losses.append(float(loss_dict["loss"].item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        logger.info(
            "[Stage B] epoch=%d/%d train_loss=%.6f val_loss=%.6f",
            epoch,
            args.diffusion_epochs,
            train_loss,
            val_loss,
        )

        if epoch >= effective_min_best_epoch and (
            best_val_loss is None or val_loss < best_val_loss
        ):
            best_val_loss = val_loss
            best_state = {
                "best_epoch": int(epoch),
                "best_val_loss": float(val_loss),
            }
            torch.save(
                {
                    "diffusion": diffusion.state_dict(),
                    "latent_dim": int(latent_dim),
                    "stage": "B",
                    "experiment": EXPERIMENT_NAME,
                },
                run_paths["stage_b_ckpt_path"],
            )
            logger.info("[Stage B] 新 best diffusion checkpoint: epoch=%d val_loss=%.6f", epoch, val_loss)

    if best_state is None:
        raise RuntimeError("Stage B 训练结束但没有生成 best checkpoint。")

    ckpt = torch.load(run_paths["stage_b_ckpt_path"], map_location=device, weights_only=False)
    diffusion.load_state_dict(ckpt["diffusion"])

    return {
        "diffusion": diffusion,
        "best_state": best_state,
        "history": history,
        "best_ckpt_path": run_paths["stage_b_ckpt_path"],
    }


def generate_synthetic_fused_bank(
    diffusion: ContinuousLatentDiffusion,
    train_class_counts: torch.Tensor,
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    根据 train split 类别分布和 rho 生成 synthetic fused bank。
    """
    synth_class_counts = build_synthetic_class_counts(
        train_class_counts=train_class_counts,
        rho=args.synthetic_rho,
    )
    total_synth = int(synth_class_counts.sum().item())
    if total_synth == 0:
        empty_latent = torch.zeros((0, diffusion.latent_dim), dtype=torch.float32)
        empty_label = torch.zeros((0,), dtype=torch.long)
        return empty_latent, empty_label, synth_class_counts

    label_chunks = []
    for class_id, class_count in enumerate(synth_class_counts.tolist()):
        if class_count <= 0:
            continue
        label_chunks.append(torch.full((class_count,), class_id, dtype=torch.long))

    synthetic_labels = torch.cat(label_chunks, dim=0)
    permutation = torch.randperm(synthetic_labels.shape[0])
    synthetic_labels = synthetic_labels[permutation]

    logger.info(
        "[Stage B] 开始按 train split 分布采样 synthetic latents: total=%d class0=%d class1=%d",
        total_synth,
        int(synth_class_counts[0].item()),
        int(synth_class_counts[1].item()),
    )

    synthetic_latents = []
    for start in range(0, total_synth, args.diffusion_batch_size):
        end = min(start + args.diffusion_batch_size, total_synth)
        batch_labels = synthetic_labels[start:end].to(device)
        batch_latents = diffusion.sample(class_labels=batch_labels, device=device)
        synthetic_latents.append(batch_latents.cpu())

    return torch.cat(synthetic_latents, dim=0), synthetic_labels, synth_class_counts


def evaluate_latent_classifier(
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """
    只在 fused latent 上评估分类头。
    """
    head.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for fused_batch, labels_batch in loader:
            fused_batch = fused_batch.to(device)
            labels_batch = labels_batch.float().to(device)
            logits = head.forward_from_fused(fused_batch).squeeze(-1)
            loss = criterion(logits, labels_batch)

            total_loss += float(loss.item())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels_batch.cpu().numpy())

    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_targets)
    auc = compute_roc_auc(preds, tgts)
    return total_loss / max(1, len(loader)), auc


def train_classifier_stage(
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    criterion,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage C：只训练 head.out_proj。
    """
    set_requires_grad(head, False)
    set_requires_grad(head.out_proj, True)

    if fused_syn_train.shape[0] > 0:
        fused_train = torch.cat([fused_real_train, fused_syn_train], dim=0)
        y_train = torch.cat([y_real_train, y_syn_train], dim=0)
    else:
        fused_train = fused_real_train
        y_train = y_real_train

    train_loader = DataLoader(
        TensorDataset(fused_train.float(), y_train.long()),
        batch_size=args.stage_c_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float(), y_valid.long()),
        batch_size=args.stage_c_batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        head.out_proj.parameters(),
        lr=args.stage_c_lr,
        weight_decay=args.stage_c_weight_decay,
    )

    best_state = None
    history: List[Dict[str, Any]] = []
    effective_min_best_epoch = min(
        max(1, int(args.stage_c_min_best_epoch)),
        int(args.stage_c_epochs),
    )

    logger.info("[Stage C] 开始只训练 out_proj。")
    for epoch in range(1, args.stage_c_epochs + 1):
        head.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for fused_batch, labels_batch in train_loader:
            fused_batch = fused_batch.to(device)
            labels_batch = labels_batch.float().to(device)
            logits = head.forward_from_fused(fused_batch).squeeze(-1)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            if args.stage_c_grad_clip is not None and args.stage_c_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    head.out_proj.parameters(),
                    max_norm=args.stage_c_grad_clip,
                )
            optimizer.step()

            total_loss += float(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(labels_batch.detach().cpu().numpy())

        train_preds = np.concatenate(all_preds)
        train_tgts = np.concatenate(all_targets)
        train_auc = compute_roc_auc(train_preds, train_tgts)
        train_loss = total_loss / max(1, len(train_loader))

        val_loss, val_auc = evaluate_latent_classifier(
            head=head,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        score_info = compute_best_score(args, val_auc=val_auc, val_loss=val_loss)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "best_score": float(score_info["score"]),
                "hybrid_score": float(score_info["hybrid_score"]),
            }
        )

        logger.info(
            "[Stage C] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            epoch,
            args.stage_c_epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
        )

        if epoch >= effective_min_best_epoch and should_replace_best(
            score_info=score_info,
            val_auc=val_auc,
            val_loss=val_loss,
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "best_score": float(score_info["score"]),
                "best_metric_value": float(score_info["metric_value"]),
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
                "best_hybrid_score": float(score_info["hybrid_score"]),
            }
            torch.save(
                {
                    "out_proj": head.out_proj.state_dict(),
                    "stage": "C",
                    "experiment": EXPERIMENT_NAME,
                },
                run_paths["stage_c_ckpt_path"],
            )
            logger.info("[Stage C] 新 best out_proj checkpoint: epoch=%d val_auc=%.4f", epoch, val_auc)

    if best_state is None:
        raise RuntimeError("Stage C 训练结束但没有生成 best checkpoint。")

    ckpt = torch.load(run_paths["stage_c_ckpt_path"], map_location=device, weights_only=False)
    head.out_proj.load_state_dict(ckpt["out_proj"])

    return {
        "best_state": best_state,
        "history": history,
        "best_ckpt_path": run_paths["stage_c_ckpt_path"],
        "train_size_real": int(fused_real_train.shape[0]),
        "train_size_synth": int(fused_syn_train.shape[0]),
        "train_size_total": int(fused_train.shape[0]),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_paths = resolve_run_paths(args)
    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)

    file_handler = logging.FileHandler(run_paths["log_file"], mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("BBBP Pool Pooler Base TuneV2 LatentDiff Exp-1")
    logger.info("=" * 80)
    logger.info("  run dir:                    %s", run_paths["run_dir"])
    logger.info("  log file:                   %s", run_paths["log_file"])
    logger.info("  stage_a_ckpt:               %s", run_paths["stage_a_ckpt_path"])
    logger.info("  stage_b_ckpt:               %s", run_paths["stage_b_ckpt_path"])
    logger.info("  stage_c_ckpt:               %s", run_paths["stage_c_ckpt_path"])
    logger.info("  summary path:               %s", run_paths["summary_path"])
    logger.info("  diffusion_num_timesteps:    %s", STAGE_B_NUM_TIMESTEPS)
    logger.info("  diffusion_beta_schedule:    %s", args.diffusion_beta_schedule)
    logger.info("  diffusion_prediction_type:  %s", args.diffusion_prediction_type)
    logger.info("  synthetic_rho:              %s", args.synthetic_rho)
    logger.info("  QGate enabled:              %s", args.use_qgate)
    logger.info("  QGate quantile:             %s", args.qgate_quantile)
    logger.info("  QGate covariance eps:       %s", args.qgate_cov_eps)
    logger.info("")

    logger.info("[0] 加载 backbone ...")
    model, dictionary, embed_dim = load_backbone(
        checkpoint_path=args.checkpoint,
        dict_path=args.dict,
    )
    model = model.to(device)

    logger.info("[1] 构建并行 latentdiff head ...")
    head = PoolPoolerBaseTuneV2LatentDiffHead(
        in_dim=embed_dim,
        out_dim=1,
        desc_hidden=args.desc_hidden,
        dropout=args.pooler_dropout,
        desc_dropout=args.desc_dropout,
        fusion_dropout=args.fusion_dropout,
        use_desc_layernorm=args.use_desc_layernorm,
        use_qsar_prompt_semantic_branch=args.use_qsar_prompt_semantic_branch,
        prompt_round_digits=args.prompt_round_digits,
        text_encoder_name_or_path=args.text_encoder_name_or_path,
        text_pooling=args.text_pooling,
        text_proj_dim=args.text_proj_dim,
        text_dropout=args.text_dropout,
        freeze_text_encoder=args.freeze_text_encoder,
        prompt_max_length=args.prompt_max_length,
        desc_text_fusion_dropout=args.desc_text_fusion_dropout,
        fusion_mode=args.fusion_mode,
        fusion_alpha_init=args.fusion_alpha_init,
        alpha_max_value=args.alpha_max_value,
        clamp_alpha=args.clamp_alpha,
        use_separate_semantic_param_group=args.use_separate_semantic_param_group,
        use_pooler_layernorm=args.use_pooler_layernorm,
        use_post_fusion_desc_layernorm=args.use_post_fusion_desc_layernorm,
        use_fused_layernorm=args.use_fused_layernorm,
    ).to(device)
    logger.info("  final_fused_dim:            %d", head.final_fused_dim)

    logger.info("[2] 加载 BBBP 数据集 ...")
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = build_data_loaders(
        args=args,
        dictionary=dictionary,
    )
    logger.info(
        "  train=%d valid=%d test=%d",
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )

    logger.info("[3] 在 train split 上拟合 descriptor normalizer ...")
    train_smiles = [item["smi"] for item in train_dataset._data]
    head.fit_normalizer(train_smiles, device)

    criterion = build_criterion(args=args, device=device)

    stage_a = run_stage_a_baseline(
        model=model,
        head=head,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        args=args,
        device=device,
        run_paths=run_paths,
    )

    set_requires_grad(model, False)
    set_requires_grad(head, False)
    model.eval()
    head.eval()

    logger.info("[4] 抽取 fused latent bank ...")
    fused_train, y_train = extract_fused_bank(model=model, head=head, loader=train_loader, device=device)
    fused_valid, y_valid = extract_fused_bank(model=model, head=head, loader=valid_loader, device=device)
    logger.info(
        "  fused_train=%s fused_valid=%s",
        tuple(fused_train.shape),
        tuple(fused_valid.shape),
    )

    train_class_counts = build_train_class_counts(y_train, num_classes=2)
    logger.info(
        "  train_class_counts: class0=%d class1=%d",
        int(train_class_counts[0].item()),
        int(train_class_counts[1].item()),
    )

    stage_b = train_diffusion_stage(
        latent_dim=head.final_fused_dim,
        fused_train=fused_train,
        y_train=y_train,
        fused_valid=fused_valid,
        y_valid=y_valid,
        args=args,
        device=device,
        run_paths=run_paths,
    )

    fused_syn, y_syn, synth_class_counts = generate_synthetic_fused_bank(
        diffusion=stage_b["diffusion"],
        train_class_counts=train_class_counts,
        args=args,
        device=device,
    )

    synth_before_counts = build_train_class_counts(y_syn, num_classes=2)
    qgate_summary = {
        "before_count": int(fused_syn.shape[0]),
        "after_count": int(fused_syn.shape[0]),
        "keep_ratio": 1.0 if fused_syn.shape[0] > 0 else 0.0,
        "per_class_before": {
            0: int(synth_before_counts[0].item()),
            1: int(synth_before_counts[1].item()),
        },
        "per_class_after": {
            0: int(synth_before_counts[0].item()),
            1: int(synth_before_counts[1].item()),
        },
    }

    logger.info(
        "[QGate] synthetic before filtering: total=%d class0=%d class1=%d",
        qgate_summary["before_count"],
        qgate_summary["per_class_before"][0],
        qgate_summary["per_class_before"][1],
    )

    if args.use_qgate and fused_syn.shape[0] > 0:
        class_stats = compute_class_stats_for_qgate(
            fused_real_train=fused_train,
            y_real_train=y_train,
            num_classes=2,
            quantile=args.qgate_quantile,
            eps=args.qgate_cov_eps,
        )
        fused_syn, y_syn, qgate_summary = apply_qgate(
            fused_syn=fused_syn,
            y_syn=y_syn,
            class_stats=class_stats,
        )
    elif not args.use_qgate:
        logger.info("[QGate] disabled; synthetic latents pass through unchanged.")

    logger.info(
        "[QGate] synthetic after filtering: total=%d keep_ratio=%.4f",
        qgate_summary["after_count"],
        qgate_summary["keep_ratio"],
    )
    for class_id in range(2):
        class_before = int(qgate_summary["per_class_before"].get(class_id, 0))
        class_after = int(qgate_summary["per_class_after"].get(class_id, 0))
        if class_before > 0 and class_after == 0:
            logger.warning(
                "[QGate] class %d before=%d after=%d (all synthetic filtered)",
                class_id,
                class_before,
                class_after,
            )
        else:
            logger.info(
                "[QGate] class %d before=%d after=%d",
                class_id,
                class_before,
                class_after,
            )

    stage_c = train_classifier_stage(
        head=head,
        fused_real_train=fused_train,
        y_real_train=y_train,
        fused_syn_train=fused_syn,
        y_syn_train=y_syn,
        fused_valid=fused_valid,
        y_valid=y_valid,
        criterion=criterion,
        args=args,
        device=device,
        run_paths=run_paths,
    )

    logger.info("[5] 用完整推理链做最终 test 评估 ...")
    test_loss, test_auc = evaluate(
        model=model,
        head=head,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary_payload = {
        "experiment": EXPERIMENT_NAME,
        "seed": int(args.seed),
        "device": str(device),
        "run_name": run_paths["run_name"],
        "run_dir": run_paths["run_dir"],
        "log_file": run_paths["log_file"],
        "stage_a_ckpt": run_paths["stage_a_ckpt_path"],
        "stage_b_ckpt": run_paths["stage_b_ckpt_path"],
        "stage_c_ckpt": run_paths["stage_c_ckpt_path"],
        "summary_path": run_paths["summary_path"],
        "final_fused_dim": int(head.final_fused_dim),
        "diffusion_num_timesteps": int(STAGE_B_NUM_TIMESTEPS),
        "train_class_counts": train_class_counts.tolist(),
        "synthetic_class_counts": synth_class_counts.tolist(),
        "synthetic_rho": float(args.synthetic_rho),
        "synthetic_total": int(fused_syn.shape[0]),
        "qgate": {
            "enabled": bool(args.use_qgate),
            "quantile": float(args.qgate_quantile),
            "cov_eps": float(args.qgate_cov_eps),
            "summary": qgate_summary,
        },
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "stage_a": stage_a,
        "stage_b": {
            "best_state": stage_b["best_state"],
            "history": stage_b["history"],
            "best_ckpt_path": stage_b["best_ckpt_path"],
        },
        "stage_c": stage_c,
        "args": vars(args),
    }
    save_json(summary_payload, run_paths["summary_path"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("Exp-1 Final Results")
    logger.info("=" * 80)
    logger.info("  Test AUC:      %.4f", test_auc)
    logger.info("  Test loss:     %.4f", test_loss)
    logger.info("  Summary json:  %s", run_paths["summary_path"])
    logger.info("=" * 80)

    return summary_payload


if __name__ == "__main__":
    main()
