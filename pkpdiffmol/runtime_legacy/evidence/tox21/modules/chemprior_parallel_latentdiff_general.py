# -*- coding: utf-8 -*-
"""
六个 MoleculeNet 目标数据集的第4组通用训练入口。
文件命名采用 chemprior_parallel_latentdiff 正式方案。

实现原则：
1. 只新增并行文件，不修改旧 baseline、旧 head、旧 general module。
2. Stage A 复用 tunev2 的并联 encoder + 化学先验 + semantic prompt 训练风格。
3. Stage B / Stage C 复用 latentdiff_stage1 已验证过的多数据集流程。
4. diffusion 只接在并联 encoder 与化学先验融合后的最终 fused latent 上。
"""

import copy
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

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

from chemprior_peak_general import (  # noqa: E402
    DATASET_SPECS,
    DatasetSpec,
    _compute_loss_and_probs,
    build_dataloaders,
    build_parser as build_general_parser,
    compute_dataset_auc,
    compute_masked_bce_loss,
    compute_multitask_auc,
    configure_logging,
    load_backbone,
    set_random_seed,
)
from chemprior_peak_latentdiff_stage1 import (  # noqa: E402
    STAGE_B_NUM_TIMESTEPS,
    apply_bace_qgate_if_needed,
    apply_multitask_qgate_if_needed,
    build_binary_diffusion_dataloaders,
    build_binary_synthetic_class_counts,
    build_binary_train_class_counts,
    build_diffusion_model,
    build_multitask_diffusion_dataloaders,
    build_synthetic_count,
    evaluate_latent_head,
    evaluate_full_model,
    extract_fused_bank,
    generate_binary_synthetic_bank,
    generate_multitask_synthetic_bank,
    load_stage_c_best,
    set_requires_grad,
    train_diffusion_stage,
)
from heads_chemprior_peak_qsarprompt_semantic_tunev2_latentdiff import (  # noqa: E402
    PoolPoolerBaseTuneV2LatentDiffHead,
)
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (  # noqa: E402
    build_optimizer_param_groups,
    compute_best_score,
    post_optimizer_step,
    prepare_epoch_training_policy,
    str2bool,
    summarize_optimizer_groups,
)
from train_bbbp_chemprior_parallel import (  # noqa: E402
    build_scheduler,
    should_replace_best,
)


LOGGER = logging.getLogger(__name__)
EXPERIMENT_SUFFIX = "baselineeq_chemprior_peak_tunev2_latentdiff"


def save_json(payload: Dict[str, Any], output_path: str) -> None:
    """
    保存 JSON 摘要，便于后续审计与对比。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_parser(dataset_name: str):
    """
    以 tunev2 general 的参数风格为主，并补充 Stage B / Stage C 参数。
    """
    parser = build_general_parser(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} baselineeq chemprior peak tunev2 latentdiff training"
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
    parser.add_argument("--diffusion-num-timesteps", type=int, default=STAGE_B_NUM_TIMESTEPS)

    parser.add_argument("--synthetic-rho", type=float, default=0.05)
    parser.add_argument(
        "--use-qgate",
        "--use_qgate",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--no_qgate",
        action="store_true",
        default=False,
        help="Disable QGate quality control for ablation study. "
        "When set, all generated synthetic latents are used "
        "without Mahalanobis filtering.",
    )
    parser.add_argument("--qgate-quantile", type=float, default=0.95)
    parser.add_argument("--qgate-cov-eps", type=float, default=1e-6)

    parser.add_argument("--stage-c-epochs", type=int, default=15)
    parser.add_argument("--stage-c-batch-size", type=int, default=256)
    parser.add_argument("--stage-c-lr", type=float, default=3e-4)
    parser.add_argument("--stage-c-weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-c-grad-clip", type=float, default=1.0)
    parser.add_argument("--stage-c-min-best-epoch", type=int, default=1)
    parser.add_argument("--stage-c-best-metric", type=str, choices=["val_auc", "val_loss", "hybrid"], default="val_auc")
    parser.add_argument("--stage-c-best-score-alpha", type=float, default=0.6)
    parser.add_argument("--stage-c-real-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-c-synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-c-real-warmup-epochs", type=int, default=0)
    parser.add_argument("--stage-c-synthetic-ramp-epochs", type=int, default=0)
    parser.add_argument("--stage-c-synthetic-decay-start-epoch", type=int, default=0)
    parser.add_argument("--stage-c-synthetic-decay-end-factor", type=float, default=1.0)
    parser.add_argument("--stage-c-mix-best-metric", type=str, choices=["val_auc", "val_loss", "hybrid"], default="val_auc")
    parser.add_argument("--stage-c-mix-best-score-alpha", type=float, default=0.6)
    parser.add_argument(
        "--stage-c-mix-auto-include-endpoints",
        "--stage_c_mix_auto_include_endpoints",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--stage-c-logit-mix-grid",
        type=str,
        default="",
        help="可选的 Stage A / Stage C logit 混合 beta 网格，例如 0,0.15,0.3,0.5,1",
    )
    parser.add_argument(
        "--stage-c-mix-prefer-lower-beta-on-auc-tie",
        "--stage_c_mix_prefer_lower_beta_on_auc_tie",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--stage-c-mix-tie-auc-eps",
        "--stage_c_mix_tie_auc_eps",
        type=float,
        default=0.0,
        help="中文注释：Stage C mix 选择时，把验证 AUC 视为近似持平的阈值；阈值内优先保留更小 beta。",
    )

    parser.add_argument("--max-diffusion-train-batches", type=int, default=0)
    parser.add_argument("--max-diffusion-valid-batches", type=int, default=0)
    parser.add_argument("--max-stagec-train-batches", type=int, default=0)
    parser.add_argument("--max-stagec-valid-batches", type=int, default=0)

    parser.add_argument(
        "--multitask-synthetic-label-mode",
        type=str,
        choices=["inherit"],
        default="inherit",
    )
    if dataset_name == "clintox":
        parser.set_defaults(
            batch_size=128,
            best_metric="val_auc",
            diffusion_batch_size=64,
            stage_c_batch_size=64,
            stage_c_logit_mix_grid="0,0.1,0.2,0.35,0.5,0.65,0.8,1",
            stage_c_mix_prefer_lower_beta_on_auc_tie=True,
            stage_c_mix_tie_auc_eps=8e-4,
        )
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    """
    为第4组生成与旧实验隔离的输出路径。
    """
    root_output_dir = os.path.abspath(args.output_dir)
    run_name = f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed{args.seed}"
    run_dir = os.path.join(root_output_dir, run_name)
    return {
        "root_output_dir": root_output_dir,
        "run_name": run_name,
        "run_dir": run_dir,
        "stage_a_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_a_best.pt"),
        "stage_b_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_b_diffusion_best.pt"),
        "stage_c_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_c_out_proj_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def build_head(args, spec: DatasetSpec, embed_dim: int, device: torch.device):
    """
    构建第4组并联 encoder + 化学先验 + diffusion head。
    """
    head = PoolPoolerBaseTuneV2LatentDiffHead(
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

    if head.out_proj.out_features != spec.output_dim:
        raise RuntimeError(
            f"head 输出维度不匹配：expected {spec.output_dim}, got {head.out_proj.out_features}"
        )
    return head


def build_stage_c_head(head: PoolPoolerBaseTuneV2LatentDiffHead):
    """
    为 Stage C 复制一个结构相同的新 head，只训练 out_proj。
    """
    head_stage_c = copy.deepcopy(head)
    set_requires_grad(head_stage_c, False)
    set_requires_grad(head_stage_c.out_proj, True)
    return head_stage_c


def compute_selection_score_info(metric_name: str, hybrid_alpha: float, val_auc: float, val_loss: float) -> Dict[str, float]:
    """
    中文注释：把 valid_auc / valid_loss 统一折算成可比较的选点评分，便于 Stage C 与 blend 使用同一套规则。
    """
    loss_score = 1.0 / (1.0 + float(val_loss))
    hybrid_score = float(hybrid_alpha) * float(val_auc) + (1.0 - float(hybrid_alpha)) * loss_score
    metric_name = str(metric_name).lower()

    if metric_name == "val_auc":
        score = float(val_auc)
        metric_value = float(val_auc)
    elif metric_name == "val_loss":
        score = -float(val_loss)
        metric_value = float(val_loss)
    elif metric_name == "hybrid":
        score = float(hybrid_score)
        metric_value = float(hybrid_score)
    else:
        raise ValueError(f"unsupported selection metric: {metric_name}")

    return {
        "metric_name": metric_name,
        "score": float(score),
        "metric_value": float(metric_value),
        "loss_score": float(loss_score),
        "hybrid_score": float(hybrid_score),
    }


def should_replace_stage_c_best_candidate(
    *,
    score_info: Dict[str, float],
    val_auc: float,
    val_loss: float,
    best_state: Any,
    args,
) -> bool:
    """
    中文注释：Stage C 单独选点沿用 Stage A 的低风险 tie-break，只是允许用独立 metric。
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

    auc_eps = float(getattr(args, "stage_c_mix_tie_auc_eps", 0.0) or args.checkpoint_tie_auc_eps)
    loss_eps = float(args.checkpoint_tie_loss_eps)
    if float(val_auc) > float(best_state["best_val_auc"]) + auc_eps:
        return True
    if abs(float(val_auc) - float(best_state["best_val_auc"])) <= auc_eps:
        if float(val_loss) < float(best_state["best_val_loss"]) - loss_eps:
            return True
    return False


def compute_stage_c_synthetic_epoch_weight(args, epoch: int) -> float:
    """
    中文注释：Stage C 对 synthetic 的权重支持 warmup、ramp、decay，避免小数据多标签任务被合成 latent 拖偏。
    """
    weight = max(0.0, float(getattr(args, "stage_c_synthetic_loss_weight", 1.0)))
    if weight <= 0.0:
        return 0.0

    warmup_epochs = max(0, int(getattr(args, "stage_c_real_warmup_epochs", 0)))
    if epoch <= warmup_epochs:
        return 0.0

    ramp_epochs = max(0, int(getattr(args, "stage_c_synthetic_ramp_epochs", 0)))
    if ramp_epochs > 0:
        ramp_progress = min(1.0, float(epoch - warmup_epochs) / float(ramp_epochs))
        weight *= max(0.0, ramp_progress)

    decay_start_epoch = max(0, int(getattr(args, "stage_c_synthetic_decay_start_epoch", 0)))
    if decay_start_epoch > 0 and epoch > decay_start_epoch:
        tail_total = max(1, int(args.stage_c_epochs) - decay_start_epoch)
        tail_progress = min(1.0, float(epoch - decay_start_epoch) / float(tail_total))
        end_factor = max(0.0, float(getattr(args, "stage_c_synthetic_decay_end_factor", 1.0)))
        weight *= 1.0 + (end_factor - 1.0) * tail_progress

    return float(max(0.0, weight))


def compute_stage_c_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    中文注释：real / synthetic 共用一个 BCE，但按样本来源加权，并继续忽略缺失标签。
    """
    logits_2d = logits.unsqueeze(-1) if logits.ndim == 1 else logits
    targets_2d = targets.unsqueeze(-1) if targets.ndim == 1 else targets
    valid_mask = targets_2d > -0.5
    if not torch.any(valid_mask):
        return logits_2d.sum() * 0.0, torch.sigmoid(logits), 0.0

    weights = sample_weights.to(dtype=logits_2d.dtype).view(-1, 1).expand_as(targets_2d)
    weights = weights * valid_mask.to(dtype=logits_2d.dtype)
    weight_sum = float(weights.sum().detach().item())
    if weight_sum <= 0.0:
        return logits_2d.sum() * 0.0, torch.sigmoid(logits), 0.0

    per_element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits_2d,
        targets_2d,
        reduction="none",
    )
    loss = (per_element_loss * weights).sum() / weights.sum()
    return loss, torch.sigmoid(logits), weight_sum


def train_classifier_stage(
    spec: DatasetSpec,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage C：冻结 backbone，只训练 out_proj。
    中文注释：默认 real/synth 同权时与旧逻辑一致；打开新参数后，仅以最小改动实现真实样本优先。
    """
    if fused_syn_train.shape[0] > 0:
        fused_train = torch.cat([fused_real_train, fused_syn_train], dim=0)
        y_train = torch.cat([y_real_train, y_syn_train], dim=0)
        source_train = torch.cat(
            [
                torch.zeros(fused_real_train.shape[0], dtype=torch.long),
                torch.ones(fused_syn_train.shape[0], dtype=torch.long),
            ],
            dim=0,
        )
    else:
        fused_train = fused_real_train
        y_train = y_real_train
        source_train = torch.zeros(fused_real_train.shape[0], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(fused_train.float(), y_train.float(), source_train),
        batch_size=args.stage_c_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float(), y_valid.float()),
        batch_size=args.stage_c_batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        head_stage_c.out_proj.parameters(),
        lr=args.stage_c_lr,
        weight_decay=args.stage_c_weight_decay,
    )
    best_state = None
    history: List[Dict[str, Any]] = []
    real_loss_weight = max(0.0, float(getattr(args, "stage_c_real_loss_weight", 1.0)))
    stage_c_best_metric = str(getattr(args, "stage_c_best_metric", "val_auc")).lower()
    stage_c_best_alpha = float(getattr(args, "stage_c_best_score_alpha", 0.6))

    for epoch in range(1, args.stage_c_epochs + 1):
        head_stage_c.train()
        total_loss = 0.0
        batch_count = 0
        all_preds = []
        all_targets = []
        synthetic_loss_weight = compute_stage_c_synthetic_epoch_weight(args, epoch)

        for step_idx, (fused_batch, targets_batch, source_batch) in enumerate(train_loader, start=1):
            if args.max_stagec_train_batches > 0 and step_idx > args.max_stagec_train_batches:
                break
            fused_batch = fused_batch.to(device)
            targets_batch = targets_batch.to(device)
            source_batch = source_batch.to(device)

            sample_weights = torch.where(
                source_batch > 0,
                torch.full((source_batch.shape[0],), synthetic_loss_weight, device=device, dtype=torch.float32),
                torch.full((source_batch.shape[0],), real_loss_weight, device=device, dtype=torch.float32),
            )

            logits = head_stage_c.forward_from_fused(fused_batch)
            loss, probs, weight_sum = compute_stage_c_weighted_loss(
                logits=logits,
                targets=targets_batch,
                sample_weights=sample_weights,
            )
            if weight_sum <= 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.stage_c_grad_clip is not None and args.stage_c_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head_stage_c.out_proj.parameters(), args.stage_c_grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets_batch.detach().cpu().numpy())

        if batch_count == 0:
            raise RuntimeError("Stage C 璁粌闃舵娌℃湁鎵ц浠讳綍 batch銆?")

        train_preds = np.concatenate(all_preds, axis=0)
        train_targets = np.concatenate(all_targets, axis=0)
        train_loss = total_loss / batch_count
        train_auc = compute_dataset_auc(spec, train_preds, train_targets)

        val_loss, val_auc = evaluate_latent_head(
            spec=spec,
            head=head_stage_c,
            loader=valid_loader,
            device=device,
            max_batches=args.max_stagec_valid_batches,
        )
        score_info = compute_selection_score_info(
            metric_name=stage_c_best_metric,
            hybrid_alpha=stage_c_best_alpha,
            val_auc=val_auc,
            val_loss=val_loss,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "best_score": float(score_info["score"]),
                "hybrid_score": float(score_info["hybrid_score"]),
                "real_loss_weight": float(real_loss_weight),
                "synthetic_loss_weight": float(synthetic_loss_weight),
            }
        )
        LOGGER.info(
            "[Stage C] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f real_w=%.3f synth_w=%.3f",
            epoch,
            args.stage_c_epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
            real_loss_weight,
            synthetic_loss_weight,
        )

        if epoch >= args.stage_c_min_best_epoch and should_replace_stage_c_best_candidate(
            score_info=score_info,
            val_auc=val_auc,
            val_loss=val_loss,
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "best_score": float(score_info["score"]),
                "best_metric_value": float(score_info["metric_value"]),
                "best_metric": stage_c_best_metric,
                "best_hybrid_score": float(score_info["hybrid_score"]),
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
            }
            torch.save(
                {
                    "epoch": int(epoch),
                    "val_auc": float(val_auc),
                    "val_loss": float(val_loss),
                    "best_score": float(score_info["score"]),
                    "best_metric": stage_c_best_metric,
                    "out_proj": head_stage_c.out_proj.state_dict(),
                },
                run_paths["stage_c_ckpt_path"],
            )
            LOGGER.info(
                "[Stage C] new best checkpoint: epoch=%d val_auc=%.4f val_loss=%.4f score=%.4f",
                epoch,
                val_auc,
                val_loss,
                score_info["score"],
            )

    if best_state is None:
        raise RuntimeError("Stage C 璁粌缁撴潫浣嗘病鏈夌敓鎴?best checkpoint銆?")

    load_stage_c_best(head_stage_c=head_stage_c, path=run_paths["stage_c_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_score": float(best_state["best_score"]),
        "best_metric": str(best_state["best_metric"]),
        "best_metric_value": float(best_state["best_metric_value"]),
        "best_hybrid_score": float(best_state["best_hybrid_score"]),
        "best_ckpt_path": run_paths["stage_c_ckpt_path"],
        "history": history,
        "loss_weights": {
            "real": float(real_loss_weight),
            "synthetic_base": float(max(0.0, float(getattr(args, "stage_c_synthetic_loss_weight", 1.0)))),
            "real_warmup_epochs": int(getattr(args, "stage_c_real_warmup_epochs", 0)),
            "synthetic_ramp_epochs": int(getattr(args, "stage_c_synthetic_ramp_epochs", 0)),
            "synthetic_decay_start_epoch": int(getattr(args, "stage_c_synthetic_decay_start_epoch", 0)),
            "synthetic_decay_end_factor": float(getattr(args, "stage_c_synthetic_decay_end_factor", 1.0)),
        },
        "train_size_real": int(fused_real_train.shape[0]),
        "train_size_synth": int(fused_syn_train.shape[0]),
        "train_size_total": int(fused_train.shape[0]),
    }


def load_stage_a_best(
    model: nn.Module,
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    path: str,
    device: torch.device,
) -> None:
    """
    重新加载 Stage A 最优参数。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    post_optimizer_step(head)


def run_epoch_tunev2_latentdiff(
    spec: DatasetSpec,
    model: nn.Module,
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    grad_clip: float = 1.0,
    max_batches: int = 0,
):
    """
    第4组 Stage A 单轮训练/评测。
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
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        batch_count += 1
        all_preds.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if batch_count == 0:
        raise RuntimeError("Stage A 未执行任何 batch。")

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def run_stage_a_baseline(
    spec: DatasetSpec,
    model: nn.Module,
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    train_loader,
    valid_loader,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage A：按 tunev2 风格训练第4组主模型。
    """
    if args.freeze_backbone:
        set_requires_grad(model, False)

    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("Stage A 未能构建出任何可训练参数组。")

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

    LOGGER.info("[Stage A] 开始训练 baseline 主链。")
    LOGGER.info("[Stage A] total steps: %s", num_training_steps)
    LOGGER.info("[Stage A] warmup steps: %s", num_warmup_steps)
    for group_info in optimizer_group_summary:
        LOGGER.info(
            "[Stage A] group=%s lr=%g wd=%g params=%d",
            group_info["name"],
            group_info["lr"],
            group_info["weight_decay"],
            group_info["param_count"],
        )

    for epoch in range(1, args.epochs + 1):
        epoch_policy = prepare_epoch_training_policy(head, epoch, args)
        train_loss, train_auc = run_epoch_tunev2_latentdiff(
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
        val_loss, val_auc = run_epoch_tunev2_latentdiff(
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
                    "dataset": spec.name,
                    "epoch": int(epoch),
                    "best_metric": args.best_metric,
                    "best_score": float(best_state["best_score"]),
                    "best_metric_value": float(best_state["best_metric_value"]),
                    "backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "args": vars(args),
                },
                run_paths["stage_a_ckpt_path"],
            )
            LOGGER.info(
                "[Stage A] 新 best checkpoint: epoch=%d val_auc=%.4f val_loss=%.4f",
                epoch,
                val_auc,
                val_loss,
            )

    if best_state is None:
        raise RuntimeError("Stage A 训练结束但没有生成 best checkpoint。")

    load_stage_a_best(model=model, head=head, path=run_paths["stage_a_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_hybrid_score": float(best_state["best_hybrid_score"]),
        "best_ckpt_path": run_paths["stage_a_ckpt_path"],
        "optimizer_groups": optimizer_group_summary,
        "history": history,
    }


def log_run_header(args, spec: DatasetSpec, device: torch.device, run_paths: Dict[str, str]) -> None:
    """
    记录第4组实验头信息。
    """
    LOGGER.info("=" * 72)
    LOGGER.info("%s chemprior peak tunev2 latentdiff", spec.name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  device:                    %s", device)
    LOGGER.info("  task_type:                 %s", spec.task_type)
    LOGGER.info("  output_dim:                %s", spec.output_dim)
    LOGGER.info("  run_dir:                   %s", run_paths["run_dir"])
    LOGGER.info("  stage_a_ckpt:              %s", run_paths["stage_a_ckpt_path"])
    LOGGER.info("  stage_b_ckpt:              %s", run_paths["stage_b_ckpt_path"])
    LOGGER.info("  stage_c_ckpt:              %s", run_paths["stage_c_ckpt_path"])
    LOGGER.info("  summary_path:              %s", run_paths["summary_path"])
    LOGGER.info("  epochs:                    %s", args.epochs)
    LOGGER.info("  batch_size:                %s", args.batch_size)
    LOGGER.info("  synthetic_rho:             %s", args.synthetic_rho)
    LOGGER.info("  use_qgate:                 %s", bool(args.use_qgate and not getattr(args, "no_qgate", False)))
    LOGGER.info("  qgate_quantile:            %s", args.qgate_quantile)
    LOGGER.info("  qgate_cov_eps:             %s", args.qgate_cov_eps)
    LOGGER.info("  stage_c_best_metric:       %s", args.stage_c_best_metric)
    LOGGER.info("  stage_c_best_score_alpha:  %s", args.stage_c_best_score_alpha)
    LOGGER.info("  stage_c_real_loss_weight:  %s", args.stage_c_real_loss_weight)
    LOGGER.info("  stage_c_synth_loss_weight: %s", args.stage_c_synthetic_loss_weight)
    LOGGER.info("  stage_c_real_warmup:       %s", args.stage_c_real_warmup_epochs)
    LOGGER.info("  stage_c_synth_ramp:        %s", args.stage_c_synthetic_ramp_epochs)
    LOGGER.info("  stage_c_synth_decay_start: %s", args.stage_c_synthetic_decay_start_epoch)
    LOGGER.info("  stage_c_synth_decay_end:   %s", args.stage_c_synthetic_decay_end_factor)
    LOGGER.info("  stage_c_mix_best_metric:   %s", args.stage_c_mix_best_metric)
    LOGGER.info("  stage_c_mix_best_alpha:    %s", args.stage_c_mix_best_score_alpha)
    LOGGER.info(
        "  stage_c_logit_mix_grid:    %s",
        args.stage_c_logit_mix_grid if str(args.stage_c_logit_mix_grid).strip() else "disabled",
    )
    LOGGER.info("  diffusion_num_timesteps:   %s", args.diffusion_num_timesteps)
    LOGGER.info("=" * 72)


def parse_stage_c_logit_mix_grid(raw_value: str) -> List[float]:
    """
    解析 Stage C 的 logit 混合 beta 网格。
    """
    text = str(raw_value).strip()
    if not text:
        return []

    values: List[float] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        beta = float(item)
        if beta < 0.0 or beta > 1.0:
            raise ValueError("stage_c_logit_mix_grid 中的 beta 必须位于 [0, 1]")
        values.append(float(beta))

    if not values:
        return []

    deduped: List[float] = []
    seen = set()
    for beta in values:
        key = f"{beta:.8f}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(beta))
    return deduped


def infer_eval_mode_from_beta(beta: float) -> str:
    """
    中文注释：把 beta 直接映射成最终推理模式，方便 summary 和日志快速审计。
    """
    beta = float(beta)
    if abs(beta) <= 1e-8:
        return "stage_a"
    if abs(beta - 1.0) <= 1e-8:
        return "stage_c"
    return "blend"


def should_replace_stage_c_mix_candidate(
    *,
    beta: float,
    score_info: Dict[str, float],
    val_auc: float,
    val_loss: float,
    best_state: Any,
    args,
) -> bool:
    """
    Stage C mix 固定按验证 AUC 优先；ClinTox 可在 AUC 持平时优先更小 beta。
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

    auc_eps = float(getattr(args, "stage_c_mix_tie_auc_eps", 0.0) or args.checkpoint_tie_auc_eps)
    loss_eps = float(args.checkpoint_tie_loss_eps)
    best_val_auc = float(best_state["best_val_auc"])
    best_val_loss = float(best_state["best_val_loss"])
    best_beta = float(best_state["beta"])

    if float(val_auc) > best_val_auc + auc_eps:
        return True
    if float(val_auc) < best_val_auc - auc_eps:
        return False
    if bool(getattr(args, "stage_c_mix_prefer_lower_beta_on_auc_tie", False)):
        if float(beta) < best_beta - 1e-8:
            return True
        if abs(float(beta) - best_beta) <= 1e-8 and float(val_loss) < best_val_loss - loss_eps:
            return True
        return False
    if float(val_loss) < best_val_loss - loss_eps:
        return True
    return False


def evaluate_stage_c_logit_mix(
    spec: DatasetSpec,
    head_stage_a: PoolPoolerBaseTuneV2LatentDiffHead,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    fused_eval: torch.Tensor,
    y_eval: torch.Tensor,
    beta: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    在验证 latent 上评估 Stage A / Stage C 的 logit 混合。
    """
    head_stage_a.eval()
    head_stage_c.eval()

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []
    beta = float(beta)

    with torch.no_grad():
        for start in range(0, int(fused_eval.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(fused_eval.shape[0]))
            fused_batch = fused_eval[start:end].to(device)
            targets_batch = y_eval[start:end].to(device)

            logits_stage_a = head_stage_a.forward_from_fused(fused_batch)
            logits_stage_c = head_stage_c.forward_from_fused(fused_batch)
            # 中文注释：这里直接混合 logits，等价于在同一 fused 表示上做保守头插值。
            logits = (1.0 - beta) * logits_stage_a + beta * logits_stage_c
            loss, probs = _compute_loss_and_probs(spec, logits, targets_batch)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu())
            all_targets.append(targets_batch.detach().cpu())

    if batch_count == 0:
        raise RuntimeError("Stage C logit mix 验证阶段没有执行任何 batch。")

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    val_auc = compute_dataset_auc(spec, preds, targets)
    val_loss = total_loss / batch_count
    return {
        "beta": float(beta),
        "val_loss": float(val_loss),
        "val_auc": float(val_auc),
    }


def select_stage_c_logit_mix_beta(
    spec: DatasetSpec,
    head_stage_a: PoolPoolerBaseTuneV2LatentDiffHead,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    args,
    device: torch.device,
) -> Dict[str, Any]:
    """
    在验证集上选择 Stage A / Stage C 的最佳 logit 混合 beta。
    """
    beta_grid = parse_stage_c_logit_mix_grid(getattr(args, "stage_c_logit_mix_grid", ""))
    if not beta_grid:
        return {
            "enabled": False,
            "selected_beta": 1.0,
            "selected_mode": "stage_c",
            "selection_metric": str(getattr(args, "stage_c_mix_best_metric", "val_auc")).lower(),
            "selection_alpha": float(getattr(args, "stage_c_mix_best_score_alpha", 0.6)),
            "selected_score": None,
            "grid": [],
            "best_val_auc": None,
            "best_val_loss": None,
            "history": [],
        }

    if bool(getattr(args, "stage_c_mix_auto_include_endpoints", True)):
        beta_grid = sorted({float(beta) for beta in beta_grid} | {0.0, 1.0})

    best_state = None
    history: List[Dict[str, float]] = []
    eval_batch_size = max(1, int(args.stage_c_batch_size))
    mix_metric_name = str(getattr(args, "stage_c_mix_best_metric", "val_auc")).lower()
    mix_metric_alpha = float(getattr(args, "stage_c_mix_best_score_alpha", 0.6))

    for beta in beta_grid:
        metrics = evaluate_stage_c_logit_mix(
            spec=spec,
            head_stage_a=head_stage_a,
            head_stage_c=head_stage_c,
            fused_eval=fused_valid,
            y_eval=y_valid,
            beta=beta,
            batch_size=eval_batch_size,
            device=device,
        )
        score_info = compute_selection_score_info(
            metric_name=mix_metric_name,
            hybrid_alpha=mix_metric_alpha,
            val_auc=metrics["val_auc"],
            val_loss=metrics["val_loss"],
        )
        row = {
            "beta": float(beta),
            "mode": infer_eval_mode_from_beta(beta),
            "val_auc": float(metrics["val_auc"]),
            "val_loss": float(metrics["val_loss"]),
            "score": float(score_info["score"]),
            "hybrid_score": float(score_info["hybrid_score"]),
        }
        history.append(row)
        LOGGER.info(
            "[Stage C Mix] beta=%.3f mode=%s val_loss=%.4f val_auc=%.4f score=%.4f",
            beta,
            row["mode"],
            metrics["val_loss"],
            metrics["val_auc"],
            score_info["score"],
        )

        if should_replace_stage_c_mix_candidate(
            beta=beta,
            score_info=score_info,
            val_auc=metrics["val_auc"],
            val_loss=metrics["val_loss"],
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "beta": float(beta),
                "mode": infer_eval_mode_from_beta(beta),
                "best_score": float(score_info["score"]),
                "selection_metric": mix_metric_name,
                "selection_alpha": float(mix_metric_alpha),
                "best_val_auc": float(metrics["val_auc"]),
                "best_val_loss": float(metrics["val_loss"]),
            }

    if best_state is None:
        raise RuntimeError("Stage C logit mix 选择阶段未生成有效 beta。")

    LOGGER.info(
        "[Stage C Mix] selected beta=%.3f mode=%s val_loss=%.4f val_auc=%.4f score=%.4f",
        best_state["beta"],
        best_state["mode"],
        best_state["best_val_loss"],
        best_state["best_val_auc"],
        best_state["best_score"],
    )
    return {
        "enabled": True,
        "selected_beta": float(best_state["beta"]),
        "selected_mode": str(best_state["mode"]),
        "selection_metric": str(best_state["selection_metric"]),
        "selection_alpha": float(best_state["selection_alpha"]),
        "selected_score": float(best_state["best_score"]),
        "grid": [float(beta) for beta in beta_grid],
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "history": history,
    }


def evaluate_full_model_with_stage_c_mix(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: PoolPoolerBaseTuneV2LatentDiffHead,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    device: torch.device,
    mix_beta: float = 1.0,
    max_batches: int = 0,
) -> tuple[float, float]:
    """
    使用完整推理链评估，并支持 Stage A / Stage C 的 logit 混合。
    """
    model.eval()
    head_for_feature.eval()
    head_stage_c.eval()

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []
    mix_beta = float(mix_beta)

    with torch.no_grad():
        for step_idx, (tokens, targets, smiles_batch) in enumerate(loader, start=1):
            if max_batches > 0 and step_idx > max_batches:
                break
            tokens = tokens.to(device)
            targets = targets.to(device)
            hidden_states, _ = model(
                src_tokens=tokens,
                src_lengths=None,
                features_only=True,
                levenshtein=False,
            )
            fused = head_for_feature.extract_fused_latent(hidden_states, smiles=smiles_batch, detach=False)
            logits_stage_a = head_for_feature.forward_from_fused(fused)
            logits_stage_c = head_stage_c.forward_from_fused(fused)
            logits = (1.0 - mix_beta) * logits_stage_a + mix_beta * logits_stage_c
            loss, probs = _compute_loss_and_probs(spec, logits, targets)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if batch_count == 0:
        raise RuntimeError("测试阶段没有执行任何 batch。")

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def run_tunev2_latentdiff_experiment(dataset_name: str):
    """
    运行单个数据集的第4组三阶段实验。
    """
    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    configure_logging(args.log_file)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_paths = resolve_run_paths(args, dataset_name)
    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)

    log_run_header(args, spec, device, run_paths)

    LOGGER.info("[1] Loading backbone...")
    model, dictionary, embed_dim = load_backbone(args.checkpoint, args.dict)
    model = model.to(device)

    LOGGER.info("[2] Building tunev2 latentdiff head...")
    head = build_head(args=args, spec=spec, embed_dim=embed_dim, device=device)
    if args.freeze_backbone:
        set_requires_grad(model, False)

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
    head.fit_normalizer(train_dataset.collect_smiles(), device)

    LOGGER.info("[5] Stage A training...")
    stage_a = run_stage_a_baseline(
        spec=spec,
        model=model,
        head=head,
        train_loader=train_loader,
        valid_loader=valid_loader,
        args=args,
        device=device,
        run_paths=run_paths,
    )

    LOGGER.info("[6] Extracting fused latent bank...")
    set_requires_grad(model, False)
    set_requires_grad(head, False)
    fused_train, y_train = extract_fused_bank(
        model=model,
        head=head,
        loader=train_loader,
        device=device,
        max_batches=args.max_train_batches,
    )
    fused_valid, y_valid = extract_fused_bank(
        model=model,
        head=head,
        loader=valid_loader,
        device=device,
        max_batches=args.max_valid_batches,
    )
    LOGGER.info("  fused_train=%s fused_valid=%s", tuple(fused_train.shape), tuple(fused_valid.shape))

    if fused_train.shape[1] != head.final_fused_dim:
        raise RuntimeError(
            f"extract_fused_latent 维度不匹配：expected {head.final_fused_dim}, got {fused_train.shape[1]}"
        )

    if spec.is_multitask:
        LOGGER.info("[7] Stage B global diffusion for multitask...")
        diffusion = build_diffusion_model(
            args=args,
            latent_dim=head.final_fused_dim,
            is_binary_conditioned=False,
            device=device,
        )
        diffusion_train_loader, diffusion_valid_loader = build_multitask_diffusion_dataloaders(
            fused_train=fused_train,
            fused_valid=fused_valid,
            batch_size=args.diffusion_batch_size,
        )
        stage_b = train_diffusion_stage(
            diffusion=diffusion,
            train_loader=diffusion_train_loader,
            valid_loader=diffusion_valid_loader,
            args=args,
            device=device,
            run_paths=run_paths,
        )
        fused_syn, y_syn = generate_multitask_synthetic_bank(
            diffusion=stage_b["diffusion"],
            fused_real_train=fused_train,
            y_real_train=y_train,
            args=args,
            device=device,
        )
        fused_syn, y_syn, qgate_summary = apply_multitask_qgate_if_needed(
            fused_real_train=fused_train,
            fused_syn=fused_syn,
            y_syn=y_syn,
            args=args,
        )
    else:
        LOGGER.info("[7] Stage B binary-conditioned diffusion for BACE...")
        y_train_binary = y_train.view(-1).long()
        y_valid_binary = y_valid.view(-1).long()
        train_class_counts = build_binary_train_class_counts(y_train_binary, num_classes=2)
        LOGGER.info(
            "  train_class_counts: class0=%d class1=%d",
            int(train_class_counts[0].item()),
            int(train_class_counts[1].item()),
        )
        diffusion = build_diffusion_model(
            args=args,
            latent_dim=head.final_fused_dim,
            is_binary_conditioned=True,
            device=device,
        )
        diffusion_train_loader, diffusion_valid_loader = build_binary_diffusion_dataloaders(
            fused_train=fused_train,
            y_train=y_train_binary,
            fused_valid=fused_valid,
            y_valid=y_valid_binary,
            batch_size=args.diffusion_batch_size,
        )
        stage_b = train_diffusion_stage(
            diffusion=diffusion,
            train_loader=diffusion_train_loader,
            valid_loader=diffusion_valid_loader,
            args=args,
            device=device,
            run_paths=run_paths,
        )
        fused_syn, y_syn, _ = generate_binary_synthetic_bank(
            diffusion=stage_b["diffusion"],
            train_class_counts=train_class_counts,
            args=args,
            device=device,
        )
        fused_syn, y_syn, qgate_summary = apply_bace_qgate_if_needed(
            fused_real_train=fused_train,
            y_real_train=y_train_binary,
            fused_syn=fused_syn,
            y_syn=y_syn,
            args=args,
        )
        y_syn = y_syn.view(-1, 1).float()

    LOGGER.info(
        "[QGate] synthetic before/after: %d -> %d",
        int(qgate_summary["before_count"]),
        int(qgate_summary["after_count"]),
    )

    LOGGER.info("[8] Stage C latent classifier training...")
    head_stage_c = build_stage_c_head(head).to(device)
    stage_c = train_classifier_stage(
        spec=spec,
        head_stage_c=head_stage_c,
        fused_real_train=fused_train.float(),
        y_real_train=y_train.float(),
        fused_syn_train=fused_syn.float(),
        y_syn_train=y_syn.float(),
        fused_valid=fused_valid.float(),
        y_valid=y_valid.float(),
        args=args,
        device=device,
        run_paths=run_paths,
    )
    stage_c_logit_mix = select_stage_c_logit_mix_beta(
        spec=spec,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        fused_valid=fused_valid.float(),
        y_valid=y_valid.float(),
        args=args,
        device=device,
    )

    selected_beta = float(stage_c_logit_mix["selected_beta"])
    final_eval_mode = infer_eval_mode_from_beta(selected_beta)

    LOGGER.info("[9] Auditing stage_a / stage_c / selected blend on valid split...")
    stage_a_valid_metrics = evaluate_stage_c_logit_mix(
        spec=spec,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        fused_eval=fused_valid.float(),
        y_eval=y_valid.float(),
        beta=0.0,
        batch_size=max(1, int(args.stage_c_batch_size)),
        device=device,
    )
    stage_c_valid_metrics = evaluate_stage_c_logit_mix(
        spec=spec,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        fused_eval=fused_valid.float(),
        y_eval=y_valid.float(),
        beta=1.0,
        batch_size=max(1, int(args.stage_c_batch_size)),
        device=device,
    )
    selected_valid_metrics = evaluate_stage_c_logit_mix(
        spec=spec,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        fused_eval=fused_valid.float(),
        y_eval=y_valid.float(),
        beta=selected_beta,
        batch_size=max(1, int(args.stage_c_batch_size)),
        device=device,
    )

    LOGGER.info("[10] Evaluating stage_a / stage_c / selected blend on test split...")
    stage_a_test_loss, stage_a_test_auc = evaluate_full_model_with_stage_c_mix(
        spec=spec,
        model=model,
        head_for_feature=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
        mix_beta=0.0,
        max_batches=args.max_test_batches,
    )
    stage_c_test_loss, stage_c_test_auc = evaluate_full_model_with_stage_c_mix(
        spec=spec,
        model=model,
        head_for_feature=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
        mix_beta=1.0,
        max_batches=args.max_test_batches,
    )
    test_loss, test_auc = evaluate_full_model_with_stage_c_mix(
        spec=spec,
        model=model,
        head_for_feature=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
        mix_beta=selected_beta,
        max_batches=args.max_test_batches,
    )

    summary = {
        "dataset": dataset_name,
        "experiment": EXPERIMENT_SUFFIX,
        "task_type": spec.task_type,
        "output_dim": spec.output_dim,
        "stage_a": stage_a,
        "stage_b": {
            "best_epoch": stage_b["best_epoch"],
            "best_val_loss": stage_b["best_val_loss"],
            "best_ckpt_path": stage_b["best_ckpt_path"],
            "history": stage_b["history"],
            "diffusion_num_timesteps": int(args.diffusion_num_timesteps),
        },
        "stage_c": stage_c,
        "stage_c_logit_mix": stage_c_logit_mix,
        "final_eval_mode": final_eval_mode,
        "final_eval": {
            "mode": final_eval_mode,
            "selected_beta": float(selected_beta),
            "selection_metric": stage_c_logit_mix.get("selection_metric"),
            "selection_alpha": stage_c_logit_mix.get("selection_alpha"),
            "selected_score": stage_c_logit_mix.get("selected_score"),
            "selected_valid_auc": float(selected_valid_metrics["val_auc"]),
            "selected_valid_loss": float(selected_valid_metrics["val_loss"]),
            "selected_test_auc": float(test_auc),
            "selected_test_loss": float(test_loss),
        },
        "mode_metrics": {
            "stage_a": {
                "mode": "stage_a",
                "beta": 0.0,
                "valid_auc": float(stage_a_valid_metrics["val_auc"]),
                "valid_loss": float(stage_a_valid_metrics["val_loss"]),
                "test_auc": float(stage_a_test_auc),
                "test_loss": float(stage_a_test_loss),
            },
            "stage_c": {
                "mode": "stage_c",
                "beta": 1.0,
                "valid_auc": float(stage_c_valid_metrics["val_auc"]),
                "valid_loss": float(stage_c_valid_metrics["val_loss"]),
                "test_auc": float(stage_c_test_auc),
                "test_loss": float(stage_c_test_loss),
            },
            "selected": {
                "mode": final_eval_mode,
                "beta": float(selected_beta),
                "valid_auc": float(selected_valid_metrics["val_auc"]),
                "valid_loss": float(selected_valid_metrics["val_loss"]),
                "test_auc": float(test_auc),
                "test_loss": float(test_loss),
            },
        },
        "qgate": qgate_summary,
        "synthetic_total_after_qgate": int(fused_syn.shape[0]),
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "args": vars(args),
        "run_paths": run_paths,
    }
    save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s tunev2 latentdiff final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  best stage A val_auc: %.4f", stage_a["best_val_auc"])
    LOGGER.info("  best stage C val_auc: %.4f", stage_c["best_val_auc"])
    LOGGER.info("  final_eval_mode:      %s", final_eval_mode)
    LOGGER.info("  stage C mix beta:      %.3f", float(selected_beta))
    LOGGER.info("  stage_a test_auc:     %.4f", stage_a_test_auc)
    LOGGER.info("  stage_c test_auc:     %.4f", stage_c_test_auc)
    LOGGER.info("  selected test_auc:    %.4f", test_auc)
    LOGGER.info("  summary_path:         %s", run_paths["summary_path"])
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    return summary


def main(dataset_name: str):
    """
    供各数据集 wrapper 调用的入口。
    """
    return run_tunev2_latentdiff_experiment(dataset_name)
