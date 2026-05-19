# -*- coding: utf-8 -*-
"""
BACE 专用并行方法改动版：
化学先验 + 并联 encoder + diffusion + Stage C residual adapter + residual mixed latent。

设计目标：
1. Stage A 与 Stage B 尽量复用当前稳定的协议修正版；
2. 只改 Stage C 对 synthetic latent 的利用方式；
3. 旧实现文件完全不覆盖，保持可独立运行。
"""

import json
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
    compute_dataset_auc,
    configure_logging,
    load_backbone,
    set_random_seed,
)
from chemprior_peak_latentdiff_stage1 import (  # noqa: E402
    apply_bace_qgate_if_needed,
    build_binary_diffusion_dataloaders,
    build_binary_train_class_counts,
    extract_fused_bank,
    generate_binary_synthetic_bank,
    set_requires_grad,
    train_diffusion_stage,
)
from chemprior_parallel_latentdiff_bace_sider_optimized import (  # noqa: E402
    OPTIMIZED_DEFAULTS,
    build_diffusion_model_with_steps,
    load_stage_a_best,
    log_run_header as base_log_run_header,
    resolve_run_paths as base_resolve_run_paths,
    run_stage_a_optimized,
)
from chemprior_parallel_latentdiff_general import (  # noqa: E402
    build_head as build_parallel_latentdiff_head,
    build_parser as build_base_parser,
    compute_selection_score_info,
    save_json,
    should_replace_stage_c_mix_candidate,
)
from heads_chemprior_bace_stagec_residual_mix import (  # noqa: E402
    BaceStageCResidualMixHead,
)
from train_bbbp_chemprior_parallel import should_replace_best  # noqa: E402


LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = {"bace"}
EXPERIMENT_SUFFIX = "chemprior_parallel_latentdiff_bace_stagec_residual_mix"


def build_parser(dataset_name: str):
    """
    在旧版并行 latentdiff parser 基础上扩展新的 Stage C 参数。
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"当前方法版仅支持 {sorted(SUPPORTED_DATASETS)}，收到 {dataset_name}")

    parser = build_base_parser(dataset_name)
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} chemprior parallel latentdiff stagec residual mix"
    parser.add_argument("--stage-a-patience", type=int, default=0)
    parser.add_argument(
        "--stage-c-residual-mix-alpha",
        type=float,
        default=0.10,
        help="Stage C 中 residual mixed latent 的固定混合系数 alpha",
    )
    parser.add_argument(
        "--stage-c-adapter-dropout",
        type=float,
        default=0.10,
        help="Stage C residual adapter 的 dropout",
    )
    parser.add_argument(
        "--stage-c-train-on-real-only",
        type=str,
        choices=["true", "false"],
        default="false",
        help="是否只用 real latent 训练 Stage C；默认 false，即使用 real + residual mix",
    )
    parser.add_argument(
        "--stage-c-selector-include-pure-routes",
        type=str,
        choices=["true", "false"],
        default="true",
        help="当启用 stage_c_logit_mix_grid 时，是否自动把 beta=0/1 两条纯路由也纳入最终验证选择",
    )
    parser.add_argument(
        "--stage-c-freeze-out-proj",
        type=str,
        choices=["true", "false"],
        default="false",
        help="是否冻结 Stage C 复制得到的 out_proj，仅训练 residual adapter",
    )
    parser.add_argument(
        "--stage-c-out-proj-lr-scale",
        type=float,
        default=1.0,
        help="Stage C 中 out_proj 相对 stage_c_lr 的缩放系数；1.0 表示同 LR",
    )

    parser.add_argument(
        "--stage-a-external-ckpt",
        type=str,
        default="",
        help="Optional external Stage A checkpoint path for fixed replay into Stage B/C",
    )
    parser.add_argument(
        "--skip-stage-a-train",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to skip Stage A training and directly replay an external Stage A checkpoint",
    )

    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        **OPTIMIZED_DEFAULTS[dataset_name],
    )
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    """
    为新方法版生成独立输出目录，避免与旧实现冲突。
    """
    run_paths = base_resolve_run_paths(args, dataset_name)
    run_name = f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed{args.seed}"
    run_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
    return {
        "root_output_dir": os.path.abspath(args.output_dir),
        "run_name": run_name,
        "run_dir": run_dir,
        "stage_a_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_a_best.pt"),
        "stage_b_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_b_diffusion_best.pt"),
        "stage_c_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_c_residual_head_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
        "base_run_paths_reference": run_paths,
    }


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _format_optional_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def parse_stage_c_logit_mix_grid(raw_value: str) -> List[float]:
    """
    解析 Stage A / Stage C logit 混合 beta 网格。
    留空时表示保持当前行为：仅使用 Stage C 头进行最终推理。
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


def extend_stage_c_logit_mix_grid(beta_grid: List[float], include_pure_routes: bool) -> List[float]:
    """
    在用户给定 grid 的基础上，按需补入纯 Stage A / 纯 Stage C 路由。
    这里不改默认关闭逻辑；只有显式提供 grid 时才会生效。
    """
    if not beta_grid:
        return []

    values = list(beta_grid)
    if include_pure_routes:
        values = [0.0] + values + [1.0]

    deduped: List[float] = []
    seen = set()
    for beta in values:
        key = f"{float(beta):.8f}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(beta))
    return deduped


def beta_to_stage_c_route_name(beta: float) -> str:
    beta = float(beta)
    if abs(beta) <= 1e-8:
        return "stage_a"
    if abs(beta - 1.0) <= 1e-8:
        return "stage_c"
    return "mixed"


def _load_stage_a_summary_sidecar(ckpt_path: str) -> Dict[str, Any]:
    summary_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "summary.json")
    if not os.path.isfile(summary_path):
        return {}

    try:
        with open(summary_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        LOGGER.warning("Failed to read Stage A sidecar summary from %s: %s", summary_path, exc)
        return {}

    stage_a = payload.get("stage_a")
    if not isinstance(stage_a, dict):
        stage_a = payload.get("best_state")
    if not isinstance(stage_a, dict):
        return {"summary_path": summary_path}

    return {
        "summary_path": summary_path,
        "best_epoch": stage_a.get("best_epoch"),
        "best_val_auc": stage_a.get("best_val_auc"),
        "best_val_loss": stage_a.get("best_val_loss"),
        "best_hybrid_score": stage_a.get("best_hybrid_score"),
    }


def load_external_stage_a_checkpoint(
    model: nn.Module,
    head: nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt_path = os.path.abspath(str(ckpt_path).strip())
    if not ckpt_path:
        raise ValueError("stage_a_external_ckpt 不能为空")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"External Stage A checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"External Stage A checkpoint is not a dict payload: {ckpt_path}")

    missing_keys = [key for key in ("backbone", "head") if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"External Stage A checkpoint missing required keys {missing_keys}: {ckpt_path}"
        )

    try:
        model.load_state_dict(checkpoint["backbone"])
    except RuntimeError as exc:
        raise RuntimeError(
            "External Stage A checkpoint backbone is incompatible with the current BACE residual_mix "
            f"pipeline. ckpt={ckpt_path}. Details: {exc}"
        ) from exc

    try:
        head.load_state_dict(checkpoint["head"])
    except RuntimeError as exc:
        raise RuntimeError(
            "External Stage A checkpoint head is incompatible with the current BACE residual_mix "
            "Stage A head. This usually means the source checkpoint is not from the same tunev2/parallel "
            f"head family. ckpt={ckpt_path}. Details: {exc}"
        ) from exc

    clamp_alpha = getattr(head, "clamp_alpha_", None)
    if callable(clamp_alpha):
        clamp_alpha()

    sidecar = _load_stage_a_summary_sidecar(ckpt_path)
    best_metric = str(checkpoint.get("best_metric", "")).strip().lower()
    best_metric_value = checkpoint.get("best_metric_value")
    best_epoch = sidecar.get("best_epoch", checkpoint.get("epoch"))
    best_val_auc = sidecar.get("best_val_auc")
    best_val_loss = sidecar.get("best_val_loss")
    best_hybrid_score = sidecar.get("best_hybrid_score")

    if best_val_auc is None and best_metric == "val_auc" and best_metric_value is not None:
        best_val_auc = float(best_metric_value)
    if best_val_loss is None and best_metric == "val_loss" and best_metric_value is not None:
        best_val_loss = float(best_metric_value)
    if best_hybrid_score is None and best_metric == "hybrid" and best_metric_value is not None:
        best_hybrid_score = float(best_metric_value)

    return {
        "source": "external_replay",
        "replayed": True,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_auc": None if best_val_auc is None else float(best_val_auc),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "best_hybrid_score": None if best_hybrid_score is None else float(best_hybrid_score),
        "best_ckpt_path": ckpt_path,
        "optimizer_groups": [],
        "history": [],
        "early_stopped": False,
        "stopped_epoch": int(best_epoch) if best_epoch is not None else 0,
        "source_best_metric": checkpoint.get("best_metric"),
        "source_best_metric_value": best_metric_value,
        "source_summary_path": sidecar.get("summary_path"),
    }


def log_run_header(args, spec: DatasetSpec, device: torch.device, run_paths: Dict[str, str]) -> None:
    """
    记录本方法版的关键信息，并复用旧版头部日志风格。
    """
    base_log_run_header(args=args, spec=spec, device=device, run_paths=run_paths)
    LOGGER.info("  experiment_suffix:         %s", EXPERIMENT_SUFFIX)
    LOGGER.info("  stage_c_residual_mix_alpha:%s", args.stage_c_residual_mix_alpha)
    LOGGER.info("  stage_c_adapter_dropout:   %s", args.stage_c_adapter_dropout)
    LOGGER.info("  stage_c_train_on_real_only:%s", args.stage_c_train_on_real_only)
    LOGGER.info("  stage_c_best_metric:      %s", args.stage_c_best_metric)
    LOGGER.info("  stage_c_best_score_alpha: %s", args.stage_c_best_score_alpha)
    LOGGER.info("  stage_c_mix_best_metric:  %s", args.stage_c_mix_best_metric)
    LOGGER.info("  stage_c_mix_best_alpha:   %s", args.stage_c_mix_best_score_alpha)
    LOGGER.info(
        "  stage_c_logit_mix_grid:   %s",
        args.stage_c_logit_mix_grid if str(args.stage_c_logit_mix_grid).strip() else "disabled",
    )
    LOGGER.info(
        "  stage_c_selector_include_pure_routes:%s",
        args.stage_c_selector_include_pure_routes,
    )
    LOGGER.info("  stage_c_freeze_out_proj:  %s", args.stage_c_freeze_out_proj)
    LOGGER.info("  stage_c_out_proj_lr_scale:%s", args.stage_c_out_proj_lr_scale)
    LOGGER.info("  skip_stage_a_train:       %s", args.skip_stage_a_train)
    LOGGER.info("  stage_a_external_ckpt:    %s", args.stage_a_external_ckpt or "n/a")
    LOGGER.info("=" * 72)


def build_stage_c_residual_head(head: nn.Module, args) -> BaceStageCResidualMixHead:
    """
    构建新的 Stage C residual adapter 头。
    """
    stage_c_head = BaceStageCResidualMixHead(
        base_head=head,
        adapter_dropout=float(args.stage_c_adapter_dropout),
    )
    return stage_c_head


def save_stage_c_best(
    head_stage_c: BaceStageCResidualMixHead,
    epoch: int,
    val_auc: float,
    run_paths: Dict[str, str],
) -> None:
    """
    保存新的 Stage C 最优权重。
    """
    torch.save(
        {
            "epoch": int(epoch),
            "val_auc": float(val_auc),
            "state_dict": head_stage_c.state_dict(),
        },
        run_paths["stage_c_ckpt_path"],
    )


def load_stage_c_best(
    head_stage_c: BaceStageCResidualMixHead,
    path: str,
    device: torch.device,
) -> None:
    """
    加载新的 Stage C 最优权重。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head_stage_c.load_state_dict(checkpoint["state_dict"])


def build_residual_mixed_latents(
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    生成 residual mixed latent。

    规则：
    1. 每个 synthetic latent 从同标签 real latent 中随机抽一个 anchor；
    2. 构造 z_mix = (1 - alpha) * z_real + alpha * z_syn；
    3. 标签直接继承 synthetic 对应标签。
    """
    import random

    if fused_syn_train.shape[0] == 0:
        empty_latent = fused_real_train.new_zeros((0, fused_real_train.shape[1]))
        empty_label = y_real_train.new_zeros((0, y_real_train.shape[1]))
        return empty_latent, empty_label, {
            "mix_count": 0,
            "alpha": float(alpha),
            "fallback_to_real_only": True,
        }

    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("stage_c_residual_mix_alpha 必须位于 [0, 1]")

    real_labels_flat = y_real_train.view(-1).long()
    syn_labels_flat = y_syn_train.view(-1).long()
    anchor_indices_list: List[int] = []
    missing_label_count = 0

    for syn_label in syn_labels_flat.tolist():
        candidate_indices = torch.nonzero(real_labels_flat == int(syn_label), as_tuple=False).view(-1)
        if candidate_indices.numel() == 0:
            missing_label_count += 1
            anchor_indices_list.append(random.randrange(int(fused_real_train.shape[0])))
            continue
        chosen_index = int(candidate_indices[random.randrange(int(candidate_indices.numel()))].item())
        anchor_indices_list.append(chosen_index)

    anchor_indices = torch.tensor(
        anchor_indices_list,
        dtype=torch.long,
        device=fused_real_train.device,
    )
    anchor_latents = fused_real_train.index_select(0, anchor_indices)
    mixed_latents = (1.0 - alpha) * anchor_latents + alpha * fused_syn_train
    mixed_targets = y_syn_train.clone()

    summary = {
        "mix_count": int(mixed_latents.shape[0]),
        "alpha": float(alpha),
        "missing_label_anchor_count": int(missing_label_count),
        "fallback_to_real_only": False,
    }
    return mixed_latents, mixed_targets, summary


def build_stage_c_train_bank(
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    构造新的 Stage C 训练数据：
    默认使用 [z_real, z_mix]；
    如显式指定 real_only 或 synthetic 为空，则退化为仅 real。
    """
    train_on_real_only = _str_to_bool(args.stage_c_train_on_real_only)
    if train_on_real_only or fused_syn_train.shape[0] == 0:
        return fused_real_train, y_real_train, {
            "train_on_real_only": True,
            "mix_count": 0,
            "alpha": float(args.stage_c_residual_mix_alpha),
            "fallback_to_real_only": True,
        }

    fused_mix, y_mix, mix_summary = build_residual_mixed_latents(
        fused_real_train=fused_real_train,
        y_real_train=y_real_train,
        fused_syn_train=fused_syn_train,
        y_syn_train=y_syn_train,
        alpha=float(args.stage_c_residual_mix_alpha),
    )
    if fused_mix.shape[0] == 0:
        return fused_real_train, y_real_train, {
            "train_on_real_only": True,
            "mix_count": 0,
            "alpha": float(args.stage_c_residual_mix_alpha),
            "fallback_to_real_only": True,
        }

    fused_train = torch.cat([fused_real_train, fused_mix], dim=0)
    y_train = torch.cat([y_real_train, y_mix], dim=0)
    summary = {
        "train_on_real_only": False,
        "mix_count": int(fused_mix.shape[0]),
        "alpha": float(args.stage_c_residual_mix_alpha),
        "fallback_to_real_only": False,
        **mix_summary,
    }
    return fused_train, y_train, summary


def evaluate_stage_c_latent_head(
    spec: DatasetSpec,
    head_stage_c: BaceStageCResidualMixHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    仅在 fused latent 上评估新的 Stage C 头。
    """
    head_stage_c.eval()
    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for step_idx, (fused_batch, targets_batch) in enumerate(loader, start=1):
            if max_batches > 0 and step_idx > max_batches:
                break
            fused_batch = fused_batch.to(device)
            targets_batch = targets_batch.to(device)
            logits = head_stage_c.forward_from_fused(fused_batch)
            loss, probs = _compute_loss_and_probs(spec, logits, targets_batch)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets_batch.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("Stage C 验证阶段没有执行任何 batch。")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def train_stage_c_residual_mix(
    spec: DatasetSpec,
    head_stage_c: BaceStageCResidualMixHead,
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
    训练新的 Stage C：
    1. 训练集默认是 [z_real, z_mix]；
    2. 验证集只用 real latent；
    3. 支持只训练 adapter，或降低 out_proj 的学习率。
    """
    fused_train, y_train, mix_summary = build_stage_c_train_bank(
        fused_real_train=fused_real_train,
        y_real_train=y_real_train,
        fused_syn_train=fused_syn_train,
        y_syn_train=y_syn_train,
        args=args,
    )

    train_loader = DataLoader(
        TensorDataset(fused_train.float().cpu(), y_train.float().cpu()),
        batch_size=args.stage_c_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float().cpu(), y_valid.float().cpu()),
        batch_size=args.stage_c_batch_size,
        shuffle=False,
    )

    freeze_out_proj = _str_to_bool(getattr(args, "stage_c_freeze_out_proj", "false"))
    out_proj_lr_scale = float(getattr(args, "stage_c_out_proj_lr_scale", 1.0))
    if out_proj_lr_scale < 0:
        raise ValueError("stage_c_out_proj_lr_scale 必须 >= 0")

    adapter_params = [param for param in head_stage_c.adapter.parameters() if param.requires_grad]
    out_proj_params = list(head_stage_c.out_proj.parameters())
    if freeze_out_proj:
        # 中文注释：BACE 小样本下先允许只学 residual adapter，避免复制来的分类边界被快速改坏。
        for param in out_proj_params:
            param.requires_grad_(False)
    out_proj_params = [param for param in out_proj_params if param.requires_grad]

    optimizer_param_groups = []
    optimizer_group_logs: List[Dict[str, Any]] = []

    if adapter_params:
        optimizer_param_groups.append(
            {
                "params": adapter_params,
                "lr": float(args.stage_c_lr),
                "weight_decay": float(args.stage_c_weight_decay),
            }
        )
        optimizer_group_logs.append(
            {
                "name": "adapter",
                "lr": float(args.stage_c_lr),
                "weight_decay": float(args.stage_c_weight_decay),
                "param_count": int(sum(param.numel() for param in adapter_params)),
            }
        )

    if out_proj_params:
        out_proj_lr = float(args.stage_c_lr) * out_proj_lr_scale
        optimizer_param_groups.append(
            {
                "params": out_proj_params,
                "lr": float(out_proj_lr),
                "weight_decay": float(args.stage_c_weight_decay),
            }
        )
        optimizer_group_logs.append(
            {
                "name": "out_proj",
                "lr": float(out_proj_lr),
                "weight_decay": float(args.stage_c_weight_decay),
                "param_count": int(sum(param.numel() for param in out_proj_params)),
            }
        )

    if not optimizer_param_groups:
        raise RuntimeError("Stage C 没有可训练参数，请检查 freeze_out_proj / lr 配置。")

    optimizer = AdamW(optimizer_param_groups)
    trainable_stage_c_params = [param for param in head_stage_c.parameters() if param.requires_grad]
    for group in optimizer_group_logs:
        LOGGER.info(
            "[Stage C ResidualMix] optimizer_group=%s lr=%.6g wd=%.6g params=%d",
            group["name"],
            group["lr"],
            group["weight_decay"],
            group["param_count"],
        )
    best_state = None
    history: List[Dict[str, Any]] = []
    stage_c_best_metric = str(getattr(args, "stage_c_best_metric", "val_auc")).lower()
    stage_c_best_alpha = float(getattr(args, "stage_c_best_score_alpha", 0.6))

    for epoch in range(1, args.stage_c_epochs + 1):
        head_stage_c.train()
        total_loss = 0.0
        batch_count = 0
        all_preds = []
        all_targets = []

        for step_idx, (fused_batch, targets_batch) in enumerate(train_loader, start=1):
            if args.max_stagec_train_batches > 0 and step_idx > args.max_stagec_train_batches:
                break
            fused_batch = fused_batch.to(device)
            targets_batch = targets_batch.to(device)

            logits = head_stage_c.forward_from_fused(fused_batch)
            loss, probs = _compute_loss_and_probs(spec, logits, targets_batch)

            optimizer.zero_grad()
            loss.backward()
            if args.stage_c_grad_clip is not None and args.stage_c_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_stage_c_params, args.stage_c_grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets_batch.detach().cpu().numpy())

        if batch_count == 0:
            raise RuntimeError("Stage C 训练阶段没有执行任何 batch。")

        train_preds = np.concatenate(all_preds, axis=0)
        train_targets = np.concatenate(all_targets, axis=0)
        train_loss = total_loss / batch_count
        train_auc = compute_dataset_auc(spec, train_preds, train_targets)

        val_loss, val_auc = evaluate_stage_c_latent_head(
            spec=spec,
            head_stage_c=head_stage_c,
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
            }
        )
        LOGGER.info(
            "[Stage C ResidualMix] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f score=%.4f",
            epoch,
            args.stage_c_epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
            float(score_info["score"]),
        )

        if epoch >= args.stage_c_min_best_epoch and should_replace_best(
            score_info=score_info,
            val_auc=val_auc,
            val_loss=val_loss,
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "best_score": float(score_info["score"]),
                "best_metric": str(stage_c_best_metric),
                "best_metric_value": float(score_info["metric_value"]),
                "best_hybrid_score": float(score_info["hybrid_score"]),
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
            }
            save_stage_c_best(
                head_stage_c=head_stage_c,
                epoch=epoch,
                val_auc=val_auc,
                run_paths=run_paths,
            )
            LOGGER.info(
                "[Stage C ResidualMix] new best metric=%s score=%.4f val_auc=%.4f val_loss=%.4f at epoch %d",
                stage_c_best_metric,
                float(score_info["score"]),
                float(val_auc),
                float(val_loss),
                int(epoch),
            )

    if best_state is None:
        raise RuntimeError("Stage C 未产生有效 best checkpoint，请检查训练或选择配置。")
    load_stage_c_best(head_stage_c=head_stage_c, path=run_paths["stage_c_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_metric": str(best_state["best_metric"]),
        "best_metric_value": float(best_state["best_metric_value"]),
        "best_hybrid_score": float(best_state["best_hybrid_score"]),
        "best_ckpt_path": run_paths["stage_c_ckpt_path"],
        "history": history,
        "train_size_real": int(fused_real_train.shape[0]),
        "train_size_synth_after_qgate": int(fused_syn_train.shape[0]),
        "train_size_total": int(fused_train.shape[0]),
        "mix_summary": mix_summary,
        "optimizer_groups": optimizer_group_logs,
        "freeze_out_proj": bool(freeze_out_proj),
        "out_proj_lr_scale": float(out_proj_lr_scale),
    }


def evaluate_stage_c_logit_mix(
    spec: DatasetSpec,
    head_stage_a: nn.Module,
    head_stage_c: BaceStageCResidualMixHead,
    fused_eval: torch.Tensor,
    y_eval: torch.Tensor,
    beta: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    在验证 latent 上评估 Stage A / Stage C 的 logit 混合。
    beta=1 等价于当前默认行为，只使用 Stage C。
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
            logits = (1.0 - beta) * logits_stage_a + beta * logits_stage_c
            loss, probs = _compute_loss_and_probs(spec, logits, targets_batch)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets_batch.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("Stage C logit mix 验证阶段没有执行任何 batch。")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return {
        "beta": float(beta),
        "val_loss": float(total_loss / batch_count),
        "val_auc": float(auc),
    }


def select_stage_c_logit_mix_beta(
    spec: DatasetSpec,
    head_stage_a: nn.Module,
    head_stage_c: BaceStageCResidualMixHead,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    args,
    device: torch.device,
) -> Dict[str, Any]:
    """
    在验证集上为 Stage A / Stage C logit 混合选择 beta。
    默认关闭；未提供 grid 时保持 beta=1.0，即当前主线行为。
    """
    requested_beta_grid = parse_stage_c_logit_mix_grid(getattr(args, "stage_c_logit_mix_grid", ""))
    include_pure_routes = _str_to_bool(
        getattr(args, "stage_c_selector_include_pure_routes", "true")
    )
    mix_metric_name = str(getattr(args, "stage_c_mix_best_metric", "val_auc")).lower()
    mix_metric_alpha = float(getattr(args, "stage_c_mix_best_score_alpha", 0.6))
    beta_grid = extend_stage_c_logit_mix_grid(requested_beta_grid, include_pure_routes)
    if not beta_grid:
        return {
            "enabled": False,
            "selected_beta": 1.0,
            "selected_route": "stage_c",
            "grid": [],
            "requested_grid": [],
            "include_pure_routes": bool(include_pure_routes),
            "selection_metric": str(mix_metric_name),
            "selection_alpha": float(mix_metric_alpha),
            "selected_score": None,
            "best_val_auc": None,
            "best_val_loss": None,
            "history": [],
        }

    best_state = None
    history: List[Dict[str, float]] = []
    eval_batch_size = max(1, int(args.stage_c_batch_size))

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
            "val_auc": float(metrics["val_auc"]),
            "val_loss": float(metrics["val_loss"]),
            "score": float(score_info["score"]),
            "hybrid_score": float(score_info["hybrid_score"]),
            "route": beta_to_stage_c_route_name(float(beta)),
        }
        history.append(row)
        LOGGER.info(
            "[Stage C Mix] beta=%.3f route=%s val_loss=%.4f val_auc=%.4f score=%.4f",
            float(beta),
            row["route"],
            float(metrics["val_loss"]),
            float(metrics["val_auc"]),
            float(score_info["score"]),
        )

        if should_replace_stage_c_mix_candidate(
            beta=float(beta),
            score_info=score_info,
            val_auc=float(metrics["val_auc"]),
            val_loss=float(metrics["val_loss"]),
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "beta": float(beta),
                "best_score": float(score_info["score"]),
                "selection_metric": str(mix_metric_name),
                "selection_alpha": float(mix_metric_alpha),
                "best_val_auc": float(metrics["val_auc"]),
                "best_val_loss": float(metrics["val_loss"]),
            }

    if best_state is None:
        raise RuntimeError("Stage C logit mix 选择阶段未生成有效 beta。")

    LOGGER.info(
        "[Stage C Mix] selected beta=%.3f route=%s val_loss=%.4f val_auc=%.4f score=%.4f",
        float(best_state["beta"]),
        beta_to_stage_c_route_name(float(best_state["beta"])),
        float(best_state["best_val_loss"]),
        float(best_state["best_val_auc"]),
        float(best_state["best_score"]),
    )
    return {
        "enabled": True,
        "selected_beta": float(best_state["beta"]),
        "selected_route": beta_to_stage_c_route_name(float(best_state["beta"])),
        "grid": [float(beta) for beta in beta_grid],
        "requested_grid": [float(beta) for beta in requested_beta_grid],
        "include_pure_routes": bool(include_pure_routes),
        "selection_metric": str(best_state["selection_metric"]),
        "selection_alpha": float(best_state["selection_alpha"]),
        "selected_score": float(best_state["best_score"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "history": history,
    }


def evaluate_full_model_with_stage_c_residual(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: nn.Module,
    head_stage_c: BaceStageCResidualMixHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    完整推理链评估：
    backbone -> fused real latent -> residual adapter -> out_proj
    """
    model.eval()
    head_for_feature.eval()
    head_stage_c.eval()

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

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
            fused = head_for_feature.extract_fused_latent(
                hidden_states,
                smiles=smiles_batch,
                detach=False,
            )
            logits = head_stage_c.forward_from_fused(fused)
            loss, probs = _compute_loss_and_probs(spec, logits, targets)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("测试阶段没有执行任何 batch。")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def evaluate_full_model_with_stage_c_mix(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: nn.Module,
    head_stage_c: BaceStageCResidualMixHead,
    loader,
    device: torch.device,
    mix_beta: float = 1.0,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    完整推理链评估，并支持 Stage A / Stage C 的 logit 混合。
    mix_beta=1 时完全等价于当前默认行为。
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
            fused = head_for_feature.extract_fused_latent(
                hidden_states,
                smiles=smiles_batch,
                detach=False,
            )
            logits_stage_a = head_for_feature.forward_from_fused(fused)
            logits_stage_c = head_stage_c.forward_from_fused(fused)
            logits = (1.0 - mix_beta) * logits_stage_a + mix_beta * logits_stage_c
            loss, probs = _compute_loss_and_probs(spec, logits, targets)

            total_loss += float(loss.item())
            batch_count += 1
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("测试阶段没有执行任何 batch。")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def evaluate_stage_c_test_routes(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: nn.Module,
    head_stage_c: BaceStageCResidualMixHead,
    loader,
    device: torch.device,
    selected_beta: float,
    max_batches: int = 0,
) -> Dict[str, Any]:
    """
    统一输出 Stage A / Stage C / 最终选中路由三份测试结果，避免日志里只剩一个 test_auc。
    """

    def _run_route(beta: float, route_name: str) -> Dict[str, Any]:
        test_loss, test_auc = evaluate_full_model_with_stage_c_mix(
            spec=spec,
            model=model,
            head_for_feature=head_for_feature,
            head_stage_c=head_stage_c,
            loader=loader,
            device=device,
            mix_beta=float(beta),
            max_batches=max_batches,
        )
        return {
            "route": route_name,
            "beta": float(beta),
            "test_loss": float(test_loss),
            "test_auc": float(test_auc),
        }

    stage_a_metrics = _run_route(beta=0.0, route_name="stage_a")
    stage_c_metrics = _run_route(beta=1.0, route_name="stage_c")
    selected_route = beta_to_stage_c_route_name(float(selected_beta))

    if selected_route == "stage_a":
        selected_metrics = dict(stage_a_metrics)
    elif selected_route == "stage_c":
        selected_metrics = dict(stage_c_metrics)
    else:
        selected_metrics = _run_route(beta=float(selected_beta), route_name="mixed")

    return {
        "stage_a": stage_a_metrics,
        "stage_c": stage_c_metrics,
        "selected": selected_metrics,
        "selected_beta": float(selected_beta),
        "selected_route": selected_route,
    }


def run_bace_stagec_residual_mix_experiment(dataset_name: str):
    """
    运行 BACE 方法改动版实验。
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"当前方法版仅支持 {sorted(SUPPORTED_DATASETS)}，收到 {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()
    stage_a_external_ckpt = str(getattr(args, "stage_a_external_ckpt", "")).strip()
    skip_stage_a_train = _str_to_bool(getattr(args, "skip_stage_a_train", "false"))
    if stage_a_external_ckpt:
        skip_stage_a_train = True
    args.stage_a_external_ckpt = stage_a_external_ckpt
    args.skip_stage_a_train = "true" if skip_stage_a_train else "false"

    if int(args.diffusion_epochs) <= 0:
        raise ValueError("diffusion_epochs 必须大于 0")
    if int(args.stage_c_epochs) <= 0:
        raise ValueError("stage_c_epochs 必须大于 0")

    if skip_stage_a_train and not stage_a_external_ckpt:
        raise ValueError(
            "skip_stage_a_train=true requires --stage-a-external-ckpt in this BACE residual_mix validation entry"
        )

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

    LOGGER.info("[2] Building BACE residual-mix latentdiff head...")
    head = build_parallel_latentdiff_head(args=args, spec=spec, embed_dim=embed_dim, device=device)
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

    if skip_stage_a_train:
        LOGGER.info("[5] Stage A external replay enabled; skipping Stage A training...")
        stage_a = load_external_stage_a_checkpoint(
            model=model,
            head=head,
            ckpt_path=stage_a_external_ckpt,
            device=device,
        )
        LOGGER.info("[5] Loaded external Stage A ckpt: %s", stage_a["best_ckpt_path"])
    else:
        LOGGER.info("[5] Stage A training...")
        stage_a = run_stage_a_optimized(
            spec=spec,
            model=model,
            head=head,
            train_loader=train_loader,
            valid_loader=valid_loader,
            args=args,
            device=device,
            run_paths=run_paths,
        )
        load_stage_a_best(model=model, head=head, path=run_paths["stage_a_ckpt_path"], device=device)

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

    LOGGER.info("[7] Stage B binary-conditioned diffusion for BACE...")
    y_train_binary = y_train.view(-1).long()
    y_valid_binary = y_valid.view(-1).long()
    train_class_counts = build_binary_train_class_counts(y_train_binary, num_classes=2)
    LOGGER.info(
        "  train_class_counts: class0=%d class1=%d",
        int(train_class_counts[0].item()),
        int(train_class_counts[1].item()),
    )

    diffusion = build_diffusion_model_with_steps(
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

    LOGGER.info("[8] Stage C residual-mix classifier training...")
    head_stage_c = build_stage_c_residual_head(head=head, args=args).to(device)
    stage_c = train_stage_c_residual_mix(
        spec=spec,
        head_stage_c=head_stage_c,
        fused_real_train=fused_train.float().to(device),
        y_real_train=y_train.float().to(device),
        fused_syn_train=fused_syn.float().to(device),
        y_syn_train=y_syn.float().to(device),
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

    LOGGER.info("[9] Evaluating full inference chain on test split...")
    test_routes = evaluate_stage_c_test_routes(
        spec=spec,
        model=model,
        head_for_feature=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
        selected_beta=float(stage_c_logit_mix["selected_beta"]),
        max_batches=args.max_test_batches,
    )
    test_loss = float(test_routes["selected"]["test_loss"])
    test_auc = float(test_routes["selected"]["test_auc"])

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
        "qgate": qgate_summary,
        "synthetic_total_after_qgate": int(fused_syn.shape[0]),
        "stage_a_test_loss": float(test_routes["stage_a"]["test_loss"]),
        "stage_a_test_auc": float(test_routes["stage_a"]["test_auc"]),
        "stage_c_test_loss": float(test_routes["stage_c"]["test_loss"]),
        "stage_c_test_auc": float(test_routes["stage_c"]["test_auc"]),
        "mixed_test_loss": float(test_routes["selected"]["test_loss"]),
        "mixed_test_auc": float(test_routes["selected"]["test_auc"]),
        "selected_test_route": test_routes["selected_route"],
        "selected_test_beta": float(test_routes["selected_beta"]),
        "test_routes": test_routes,
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "args": vars(args),
        "run_paths": run_paths,
    }
    save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s stagec residual mix final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  stage A source:        %s", stage_a.get("source", "trained"))
    LOGGER.info("  best stage A val_auc: %s", _format_optional_metric(stage_a.get("best_val_auc")))
    LOGGER.info("  best stage C val_auc: %.4f", stage_c["best_val_auc"])
    LOGGER.info("  stage C mix beta:     %.3f", float(stage_c_logit_mix["selected_beta"]))
    LOGGER.info("  selected test route:  %s", test_routes["selected_route"])
    LOGGER.info("  stage_a_test_auc:     %.4f", float(test_routes["stage_a"]["test_auc"]))
    LOGGER.info("  stage_c_test_auc:     %.4f", float(test_routes["stage_c"]["test_auc"]))
    LOGGER.info("  mixed_test_auc:       %.4f", float(test_routes["selected"]["test_auc"]))
    LOGGER.info("  test protocol:        real latent + route-aware final selector")
    LOGGER.info("  test_auc:             %.4f", test_auc)
    LOGGER.info("  summary_path:         %s", run_paths["summary_path"])
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    return summary


def main(dataset_name: str):
    """
    BACE Stage C residual mix 方法版入口。
    """
    return run_bace_stagec_residual_mix_experiment(dataset_name)
