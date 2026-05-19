# -*- coding: utf-8 -*-
"""
六数据集通用的 tunev2 / semantic / pool_pooler_base 训练骨架。

设计原则：
1. 不修改任何旧 baseline 文件和旧 latentdiff stage1 文件；
2. 只负责“并联 encoder，无 diffusion”训练主线；
3. 严格复用六数据集现有 dataset specs、LMDB dataloader、masked BCE、multitask AUC 骨架；
4. 头部结构严格使用 BBBP 的 tunev2 / semantic / pool_pooler_base 并联头链。
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TASKS_DIR = os.path.dirname(_MODULE_DIR)
_ROOT_DIR = os.path.dirname(_TASKS_DIR)
_BBBP_SCRIPTS_DIR = os.path.join(_TASKS_DIR, "bbbp", "scripts")
_SIDER_SCRIPTS_DIR = os.path.join(_TASKS_DIR, "sider", "scripts")

for _path in (_ROOT_DIR, _MODULE_DIR, _BBBP_SCRIPTS_DIR, _SIDER_SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from chemprior_peak_general import (
    DATASET_SPECS,
    DatasetSpec,
    _compute_loss_and_probs,
    build_dataloaders,
    build_parser as build_general_parser,
    compute_dataset_auc,
    configure_logging,
    load_backbone,
    set_random_seed,
)
from heads_chemprior_peak_qsarprompt_semantic_tunev2 import PoolPoolerBaseTuneV2Head
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (
    build_optimizer_param_groups,
    compute_best_score,
    post_optimizer_step,
    prepare_epoch_training_policy,
    str2bool,
    summarize_optimizer_groups,
)
from train_bbbp_chemprior_parallel import build_scheduler, should_replace_best


LOGGER = logging.getLogger(__name__)
EXPERIMENT_SUFFIX = "baselineeq_chemprior_peak_tunev2"


def save_json(payload: Dict[str, Any], output_path: str) -> None:
    """
    保存 summary.json，便于后续审计。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_parser(dataset_name: str):
    """
    在六数据集通用 parser 基础上补充 tunev2 参数。
    """
    parser = build_general_parser(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} baselineeq chemprior peak tunev2 training"
    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        lr=spec.lr,
        warmup_ratio=spec.warmup_ratio,
        pooler_dropout=spec.pooler_dropout,
    )

    parser.add_argument("--text-lr", "--text_lr", type=float, default=1e-5)
    parser.add_argument("--semantic-proj-lr", "--semantic_proj_lr", type=float, default=5e-5)
    parser.add_argument("--alpha-lr", "--alpha_lr", type=float, default=1e-5)
    parser.add_argument("--text-weight-decay", "--text_weight_decay", type=float, default=None)
    parser.add_argument("--semantic-proj-weight-decay", "--semantic_proj_weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha-weight-decay", "--alpha_weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--best-metric", type=str, choices=["val_auc", "val_loss", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid-best-score-alpha", "--hybrid_best_score_alpha", type=float, default=0.6)
    parser.add_argument("--checkpoint-score-eps", type=float, default=1e-4)
    parser.add_argument("--checkpoint-tie-auc-eps", type=float, default=5e-4)
    parser.add_argument("--checkpoint-tie-loss-eps", type=float, default=1e-4)
    parser.add_argument("--desc-text-fusion-dropout", "--desc_text_fusion_dropout", type=float, default=0.1)
    parser.add_argument("--use-pooler-layernorm", "--use_pooler_layernorm", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--use-post-fusion-desc-layernorm",
        "--use_post_fusion_desc_layernorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--use-fused-layernorm", "--use_fused_layernorm", type=str2bool, nargs="?", const=True, default=True)
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
        default=os.path.join(os.path.dirname(tasks_dir), "text_encoder_name_or_path"),
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
    parser.add_argument("--fusion-mode", "--fusion_mode", type=str, choices=["residual", "gate"], default="residual")
    parser.add_argument("--fusion-alpha-init", "--fusion_alpha_init", type=float, default=0.005)
    parser.add_argument("--freeze-semantic-proj-epochs", "--freeze_semantic_proj_epochs", type=int, default=0)
    parser.add_argument("--alpha-warmup-epochs", "--alpha_warmup_epochs", type=int, default=5)
    parser.add_argument("--alpha-max-value", "--alpha_max_value", type=float, default=0.2)
    parser.add_argument("--clamp-alpha", "--clamp_alpha", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--prompt-max-length", "--prompt_max_length", type=int, default=128)
    if dataset_name == "clintox":
        parser.set_defaults(
            batch_size=128,
            best_metric="val_auc",
        )
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    """
    为 tunev2 实验生成独立输出路径。
    """
    root_output_dir = os.path.abspath(args.output_dir)
    run_name = f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed{args.seed}"
    run_dir = os.path.join(root_output_dir, run_name)
    return {
        "root_output_dir": root_output_dir,
        "run_name": run_name,
        "run_dir": run_dir,
        "best_ckpt_path": os.path.join(run_dir, f"{run_name}_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def build_head(args, spec: DatasetSpec, embed_dim: int, device: torch.device):
    """
    构建六数据集 tunev2 并联头。
    """
    return PoolPoolerBaseTuneV2Head(
        in_dim=embed_dim,
        out_dim=spec.output_dim,
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


def run_epoch_tunev2(
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
    """
    适配六数据集单任务/多任务的 tunev2 训练循环。
    """
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
                    trainable_params = [param for param in model.parameters() if param.requires_grad]
                    trainable_params.extend(param for param in head.parameters() if param.requires_grad)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                optimizer.step()
                post_optimizer_step(head)
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


def run_tunev2_experiment(dataset_name: str):
    """
    运行单个数据集的 tunev2 实验。
    """
    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    run_paths = resolve_run_paths(args, dataset_name)
    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)
    configure_logging(args.log_file)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("=" * 72)
    LOGGER.info("%s chemprior peak tunev2", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  task_type:                 %s", spec.task_type)
    LOGGER.info("  output_dim:                %s", spec.output_dim)
    LOGGER.info("  run_dir:                   %s", run_paths["run_dir"])
    LOGGER.info("  best_ckpt:                 %s", run_paths["best_ckpt_path"])
    LOGGER.info("  summary_path:              %s", run_paths["summary_path"])

    LOGGER.info("[1] Loading backbone...")
    model, dictionary, embed_dim = load_backbone(args.checkpoint, args.dict)
    model = model.to(device)

    LOGGER.info("[2] Building tunev2 head...")
    head = build_head(args=args, spec=spec, embed_dim=embed_dim, device=device)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    LOGGER.info("[3] Loading LMDB data...")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader,
    ) = build_dataloaders(args, dictionary, spec)

    LOGGER.info("[4] Fitting descriptor normalizer on train split...")
    head.fit_normalizer(train_dataset.collect_smiles(), device)

    LOGGER.info("[5] Building optimizer and scheduler...")
    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("未能构建任何可训练参数组。")
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    optimizer_group_summary = summarize_optimizer_groups(optimizer)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    best_state = None
    history: List[Dict[str, Any]] = []
    effective_min_best_epoch = min(max(1, int(args.min_best_epoch)), int(args.epochs))

    LOGGER.info("[6] Start training...")
    for group_info in optimizer_group_summary:
        LOGGER.info(
            "[OPT] group=%s lr=%g wd=%g params=%d",
            group_info["name"],
            group_info["lr"],
            group_info["weight_decay"],
            group_info["param_count"],
        )

    for epoch in range(1, args.epochs + 1):
        epoch_policy = prepare_epoch_training_policy(head, epoch, args)
        train_loss, train_auc = run_epoch_tunev2(
            spec=spec,
            model=model,
            head=head,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )
        val_loss, val_auc = run_epoch_tunev2(
            spec=spec,
            model=model,
            head=head,
            loader=valid_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            grad_clip=args.grad_clip,
            max_batches=args.max_valid_batches,
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
        LOGGER.info(
            "[Train] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
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
                    "best_metric": args.best_metric,
                    "best_score": float(best_state["best_score"]),
                    "best_metric_value": float(best_state["best_metric_value"]),
                },
                run_paths["best_ckpt_path"],
            )
            LOGGER.info("[Train] new best checkpoint at epoch %d", epoch)

    if best_state is None:
        raise RuntimeError("训练结束但没有生成 best checkpoint。")

    checkpoint = torch.load(run_paths["best_ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    post_optimizer_step(head)

    LOGGER.info("[7] Final test evaluation...")
    test_loss, test_auc = run_epoch_tunev2(
        spec=spec,
        model=model,
        head=head,
        loader=test_loader,
        device=device,
        optimizer=None,
        scheduler=None,
        grad_clip=args.grad_clip,
        max_batches=args.max_test_batches,
    )

    summary = {
        "dataset": dataset_name,
        "experiment": EXPERIMENT_SUFFIX,
        "task_type": spec.task_type,
        "output_dim": spec.output_dim,
        "best_state": best_state,
        "optimizer_groups": optimizer_group_summary,
        "history": history,
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "args": vars(args),
        "run_paths": run_paths,
    }
    save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s tunev2 final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  best_val_auc: %.4f", best_state["best_val_auc"])
    LOGGER.info("  test_auc:     %.4f", test_auc)
    LOGGER.info("  summary_path: %s", run_paths["summary_path"])
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    return summary


def main(dataset_name: str):
    """
    供各数据集入口脚本调用。
    """
    return run_tunev2_experiment(dataset_name)
