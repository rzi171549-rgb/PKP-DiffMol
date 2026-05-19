#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只针对 BBBP pool_pooler_base 主线的 tunev2 训练脚本。

改动目标只有一个：在不重写结构的前提下，尽量把 test AUC 推过 0.7815。
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# 中文注释：先禁用 WandB，避免导入阶段产生副作用。
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SMI_EDITOR_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_SIDER_SCRIPTS = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../sider/scripts"))
_MODULES_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../modules"))

sys.path.insert(0, _SMI_EDITOR_ROOT)
sys.path.insert(0, _SIDER_SCRIPTS)
sys.path.insert(0, _MODULES_DIR)

from heads_chemprior_peak_qsarprompt_semantic_tunev2 import (  # noqa: E402
    PoolPoolerBaseTuneV2Head,
)
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (  # noqa: E402
    BBBPLMDBDataset,
    build_optimizer_param_groups,
    collate_fn,
    compute_best_score,
    evaluate,
    load_backbone,
    post_optimizer_step,
    prepare_epoch_training_policy,
    save_json,
    str2bool,
    summarize_optimizer_groups,
    train_one_epoch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "pool_pooler_base_tunev2"
# 中文注释：文件名已统一为 chemprior_parallel，但实验名保持原值以避免影响旧结果目录。


def build_scheduler(
    optimizer,
    scheduler_name: str,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
):
    """
    构建更平滑的学习率调度器。
    默认用 warmup + cosine，保留 linear 作为对照开关。
    """

    scheduler_name = str(scheduler_name).lower()
    min_lr_ratio = float(min_lr_ratio)
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)

        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = max(0.0, min(1.0, progress))

        if scheduler_name == "linear":
            return max(min_lr_ratio, 1.0 - progress)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def resolve_run_paths(args) -> Dict[str, str]:
    """
    输出目录、日志、best checkpoint 与旧实验完全隔离。
    """
    root_output_dir = os.path.abspath(args.output_dir)
    run_name = f"{EXPERIMENT_NAME}_seed{args.seed}"
    run_dir = os.path.join(root_output_dir, run_name)

    log_name = f"{run_name}.log"
    if args.log_file:
        custom_log_name = os.path.basename(args.log_file)
        if f"seed{args.seed}" in custom_log_name:
            log_name = custom_log_name
        else:
            stem, ext = os.path.splitext(custom_log_name)
            if not ext:
                ext = ".log"
            log_name = f"{run_name}_{stem}{ext}"

    return {
        "root_output_dir": root_output_dir,
        "run_name": run_name,
        "run_dir": run_dir,
        "log_file": os.path.join(run_dir, log_name),
        "best_ckpt_path": os.path.join(run_dir, f"{run_name}_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def build_parser():
    """
    构建 tunev2 参数。
    默认值就是这次基于日志证据给出的推荐配置。
    """
    parser = argparse.ArgumentParser(
        description="BBBP pool_pooler_base 定向调参脚本（tunev2）"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_SMI_EDITOR_ROOT, "smi_editor.pt"),
    )
    parser.add_argument(
        "--dict",
        type=str,
        default=os.path.join(_SMI_EDITOR_ROOT, "smi_dict_token.txt"),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "../data/lmdb/bbbp"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "../outputs/pool_pooler_base_tunev2"),
    )
    parser.add_argument("--log-file", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--text-lr", "--text_lr", type=float, default=1e-5)
    parser.add_argument(
        "--semantic-proj-lr",
        "--semantic_proj_lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--alpha-lr", "--alpha_lr", type=float, default=1e-4)

    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backbone-weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.02)
    parser.add_argument(
        "--text-weight-decay",
        "--text_weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--semantic-proj-weight-decay",
        "--semantic_proj_weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--alpha-weight-decay",
        "--alpha_weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["linear", "cosine"],
        default="cosine",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=0.8)

    parser.add_argument("--min-best-epoch", type=int, default=8)
    parser.add_argument(
        "--best-metric",
        type=str,
        choices=["val_auc", "val_loss", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--hybrid-best-score-alpha",
        "--hybrid_best_score_alpha",
        type=float,
        default=0.6,
    )
    parser.add_argument("--checkpoint-score-eps", type=float, default=1e-4)
    parser.add_argument("--checkpoint-tie-auc-eps", type=float, default=5e-4)
    parser.add_argument("--checkpoint-tie-loss-eps", type=float, default=1e-4)

    parser.add_argument("--pooler-dropout", type=float, default=0.15)
    parser.add_argument("--desc-hidden", type=int, default=64)
    parser.add_argument("--desc-dropout", type=float, default=0.15)
    parser.add_argument("--fusion-dropout", type=float, default=0.25)
    parser.add_argument(
        "--desc-text-fusion-dropout",
        "--desc_text_fusion_dropout",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--use-desc-layernorm",
        "--use_desc_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--use-pooler-layernorm",
        "--use_pooler_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--use-post-fusion-desc-layernorm",
        "--use_post_fusion_desc_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--use-fused-layernorm",
        "--use_fused_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--use-separate-semantic-param-group",
        "--use_separate_semantic_param_group",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )

    parser.add_argument(
        "--use-qsar-prompt-semantic-branch",
        "--use_qsar_prompt_semantic_branch",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--prompt-round-digits", type=int, default=2)
    parser.add_argument(
        "--text-encoder-name-or-path",
        "--text_encoder_name_or_path",
        type=str,
        default=os.path.join(_SMI_EDITOR_ROOT, "text_encoder_name_or_path"),
    )
    parser.add_argument(
        "--text-pooling",
        "--text_pooling",
        type=str,
        choices=["pooler", "cls", "mean"],
        default="pooler",
    )
    parser.add_argument("--text-proj-dim", "--text_proj_dim", type=int, default=64)
    parser.add_argument("--text-dropout", "--text_dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze-text-encoder",
        "--freeze_text_encoder",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--fusion-mode",
        "--fusion_mode",
        type=str,
        choices=["residual", "gate"],
        default="residual",
    )
    parser.add_argument(
        "--fusion-alpha-init",
        "--fusion_alpha_init",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--freeze-semantic-proj-epochs",
        "--freeze_semantic_proj_epochs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--alpha-warmup-epochs",
        "--alpha_warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--alpha-max-value",
        "--alpha_max_value",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--clamp-alpha",
        "--clamp_alpha",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--prompt-max-length",
        "--prompt_max_length",
        type=int,
        default=128,
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=None)

    return parser


def should_replace_best(
    score_info: Dict[str, float],
    val_auc: float,
    val_loss: float,
    best_state: Dict[str, float],
    args,
) -> bool:
    """
    先按主指标比较；如果分数几乎持平，再用更保守的 tie-break。

    中文注释：日志显示单纯改 hybrid 权重并不稳定，因此这里只做低风险的近似并列裁决。
    """
    if best_state is None:
        return True

    score = float(score_info["score"])
    best_score = float(best_state["best_score"])
    score_eps = float(args.checkpoint_score_eps)

    if score > best_score + score_eps:
        return True
    if score < best_score - score_eps:
        return False

    auc_eps = float(args.checkpoint_tie_auc_eps)
    loss_eps = float(args.checkpoint_tie_loss_eps)

    if float(val_auc) > float(best_state["best_val_auc"]) + auc_eps:
        return True
    if abs(float(val_auc) - float(best_state["best_val_auc"])) <= auc_eps:
        if float(val_loss) < float(best_state["best_val_loss"]) - loss_eps:
            return True

    return False


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
    effective_min_best_epoch = min(max(1, int(args.min_best_epoch)), int(args.epochs))

    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)

    file_handler = logging.FileHandler(run_paths["log_file"], mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("BBBP Pool Pooler Base TuneV2")
    logger.info("=" * 80)
    logger.info("  ablation_tag:                     %s", EXPERIMENT_NAME)
    logger.info("  Python:                           %s", sys.executable)
    logger.info("  checkpoint:                       %s", args.checkpoint)
    logger.info("  data dir:                         %s", args.data_dir)
    logger.info("  output root:                      %s", run_paths["root_output_dir"])
    logger.info("  run dir:                          %s", run_paths["run_dir"])
    logger.info("  log file:                         %s", run_paths["log_file"])
    logger.info("  best checkpoint:                  %s", run_paths["best_ckpt_path"])
    logger.info("  summary path:                     %s", run_paths["summary_path"])
    logger.info("  device:                           %s", device)
    logger.info("  seed:                             %s", args.seed)
    logger.info("  epochs:                           %s", args.epochs)
    logger.info("  batch_size:                       %s", args.batch_size)
    logger.info("  lr:                               %s", args.lr)
    logger.info("  backbone_lr:                      %s", args.backbone_lr)
    logger.info("  head_lr:                          %s", args.head_lr)
    logger.info("  text_lr:                          %s", args.text_lr)
    logger.info("  semantic_proj_lr:                 %s", args.semantic_proj_lr)
    logger.info("  alpha_lr:                         %s", args.alpha_lr)
    logger.info("  weight_decay:                     %s", args.weight_decay)
    logger.info("  backbone_weight_decay:            %s", args.backbone_weight_decay)
    logger.info("  head_weight_decay:                %s", args.head_weight_decay)
    logger.info("  text_weight_decay:                %s", args.text_weight_decay)
    logger.info("  semantic_proj_weight_decay:       %s", args.semantic_proj_weight_decay)
    logger.info("  alpha_weight_decay:               %s", args.alpha_weight_decay)
    logger.info("  warmup_ratio:                     %s", args.warmup_ratio)
    logger.info("  scheduler:                        %s", args.scheduler)
    logger.info("  min_lr_ratio:                     %s", args.min_lr_ratio)
    logger.info("  grad_clip:                        %s", args.grad_clip)
    logger.info("  min_best_epoch:                   %s", effective_min_best_epoch)
    logger.info("  best_metric:                      %s", args.best_metric)
    logger.info("  hybrid_best_score_alpha:          %s", args.hybrid_best_score_alpha)
    logger.info("  pooler_dropout:                   %s", args.pooler_dropout)
    logger.info("  desc_hidden:                      %s", args.desc_hidden)
    logger.info("  desc_dropout:                     %s", args.desc_dropout)
    logger.info("  fusion_dropout:                   %s", args.fusion_dropout)
    logger.info("  desc_text_fusion_dropout:         %s", args.desc_text_fusion_dropout)
    logger.info("  use_desc_layernorm:               %s", args.use_desc_layernorm)
    logger.info("  use_pooler_layernorm:             %s", args.use_pooler_layernorm)
    logger.info(
        "  use_post_fusion_desc_layernorm:   %s",
        args.use_post_fusion_desc_layernorm,
    )
    logger.info("  use_fused_layernorm:              %s", args.use_fused_layernorm)
    logger.info("  text_pooling:                     %s", args.text_pooling)
    logger.info("  text_dropout:                     %s", args.text_dropout)
    logger.info("  freeze_text_encoder:              %s", args.freeze_text_encoder)
    logger.info("  freeze_semantic_proj_epochs:      %s", args.freeze_semantic_proj_epochs)
    logger.info("  alpha_warmup_epochs:              %s", args.alpha_warmup_epochs)
    logger.info("  alpha_max_value:                  %s", args.alpha_max_value)
    logger.info("  freeze_backbone:                  %s", args.freeze_backbone)
    logger.info("")

    logger.info("[1] Loading backbone...")
    model, dictionary, embed_dim = load_backbone(
        checkpoint_path=args.checkpoint,
        dict_path=args.dict,
    )
    model = model.to(device)

    logger.info("[2] Building tunev2 head...")
    head = PoolPoolerBaseTuneV2Head(
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

    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    logger.info("[3] Loading BBBP LMDB data...")
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
    logger.info(
        "  train: %d, valid: %d, test: %d",
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )
    logger.info("")

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

    logger.info("[4] Fitting descriptor normalizer on train set...")
    train_smiles = [item["smi"] for item in train_dataset._data]
    head.fit_normalizer(train_smiles, device)
    logger.info("")

    if args.pos_weight is None:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        criterion = nn.BCEWithLogitsLoss(
            reduction="mean",
            pos_weight=torch.tensor(
                [args.pos_weight],
                device=device,
                dtype=torch.float32,
            ),
        )

    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("未构建出任何可训练参数组，请检查冻结配置。")

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
    logger.info("[5] Optimizer groups:")
    for group_info in optimizer_group_summary:
        logger.info(
            "  - %-24s lr=%-10g wd=%-8g params=%d",
            group_info["name"],
            group_info["lr"],
            group_info["weight_decay"],
            group_info["param_count"],
        )
    logger.info("  total steps: %d, warmup steps: %d", num_training_steps, num_warmup_steps)
    logger.info("")

    logger.info("[6] Training loop start...")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_policy = prepare_epoch_training_policy(head, epoch, args)
        logger.info(
            "Epoch %3d policy | semantic_proj_trainable=%s alpha_scale=%.4f raw_alpha=%.6f effective_alpha=%.6f",
            epoch,
            epoch_policy["semantic_proj_trainable"],
            epoch_policy["alpha_runtime_scale"],
            epoch_policy["raw_alpha"],
            epoch_policy["effective_alpha"],
        )

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
        history_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_auc": float(train_auc),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc),
            "best_score": float(score_info["score"]),
            "loss_score": float(score_info["loss_score"]),
            "hybrid_score": float(score_info["hybrid_score"]),
            "semantic_proj_trainable": bool(epoch_policy["semantic_proj_trainable"]),
            "alpha_runtime_scale": float(epoch_policy["alpha_runtime_scale"]),
            "raw_alpha": float(epoch_policy["raw_alpha"]),
            "effective_alpha": float(epoch_policy["effective_alpha"]),
        }
        history.append(history_record)

        logger.info(
            "Epoch %3d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f hybrid=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
            score_info["hybrid_score"],
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
                    "best_metric": args.best_metric,
                    "best_score": float(best_state["best_score"]),
                    "best_metric_value": float(best_state["best_metric_value"]),
                    "val_auc": float(val_auc),
                    "val_loss": float(val_loss),
                    "hybrid_score": float(score_info["hybrid_score"]),
                    "run_dir": run_paths["run_dir"],
                    "seed": int(args.seed),
                    "experiment": EXPERIMENT_NAME,
                },
                run_paths["best_ckpt_path"],
            )
            logger.info(
                "  *** New best %s=%.6f at epoch %d | val_auc=%.4f val_loss=%.4f ***",
                args.best_metric,
                best_state["best_metric_value"],
                epoch,
                val_auc,
                val_loss,
            )

    if best_state is None:
        raise RuntimeError("训练结束但未生成 best checkpoint，请检查 min_best_epoch 与 epochs 配置。")

    logger.info("")
    logger.info("[7] Loading best checkpoint for test evaluation...")
    ckpt = torch.load(run_paths["best_ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    post_optimizer_step(head)

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
        "best_checkpoint": run_paths["best_ckpt_path"],
        "summary_path": run_paths["summary_path"],
        "best_metric": args.best_metric,
        "best_score": float(best_state["best_score"]),
        "best_metric_value": float(best_state["best_metric_value"]),
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_hybrid_score": float(best_state["best_hybrid_score"]),
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "effective_min_best_epoch": int(effective_min_best_epoch),
        "optimizer_groups": optimizer_group_summary,
        "args": vars(args),
        "history": history,
    }
    save_json(summary_payload, run_paths["summary_path"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("BBBP Pool Pooler Base TuneV2 Final Results")
    logger.info("=" * 80)
    logger.info("  Best epoch:    %d", best_state["best_epoch"])
    logger.info("  Best score:    %.6f", best_state["best_score"])
    logger.info("  Best val AUC:  %.4f", best_state["best_val_auc"])
    logger.info("  Best val loss: %.4f", best_state["best_val_loss"])
    logger.info("  Test AUC:      %.4f", test_auc)
    logger.info("  Test loss:     %.4f", test_loss)
    logger.info("  Summary json:  %s", run_paths["summary_path"])
    logger.info("=" * 80)

    return summary_payload


if __name__ == "__main__":
    main()
