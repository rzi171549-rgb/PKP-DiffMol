# -*- coding: utf-8 -*-
"""
六个 MoleculeNet 目标数据集的第一阶段 latent diffusion 迁移入口。

实现原则：
1. 不修改任何旧 baseline 文件；
2. 最大化复用 chemprior_peak_general.py 的公共训练骨架；
3. BACE 复用 BBBP 已验证的二分类 conditioned diffusion + class-wise QGate；
4. ClinTox/Tox21/SIDER/ToxCast/MUV 采用第一阶段保守版：
   - Stage A/Stage C 继续使用 masked BCE / multitask AUC；
   - Stage B 采用全局无条件 fused latent diffusion；
   - QGate 采用全局无标签版本；
   - synthetic 标签只允许从 train split 真实样本继承。
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

try:
    from chemprior_peak_general import (
        DATASET_SPECS,
        DatasetSpec,
        _compute_loss_and_probs,
        build_dataloaders,
        build_optimizer_and_scheduler,
        build_parser as build_general_parser,
        compute_dataset_auc,
        configure_logging,
        load_backbone,
        run_epoch,
        set_random_seed,
    )
    from heads_chemprior_peak_latentdiff_stage1 import BaselineEquivalentChemPriorLatentDiffHead
    from latent_diffusion_core import ContinuousLatentDiffusion
    from latent_diffusion_denoiser import LatentDiffusionMLPDenoiser
    from latent_diffusion_denoiser_stage1 import GlobalLatentDiffusionMLPDenoiser
    from latent_qgate_stage1 import (
        apply_global_qgate,
        apply_qgate,
        compute_class_stats_for_qgate,
        compute_global_stats_for_qgate,
    )
except ImportError:  # pragma: no cover
    from .chemprior_peak_general import (
        DATASET_SPECS,
        DatasetSpec,
        _compute_loss_and_probs,
        build_dataloaders,
        build_optimizer_and_scheduler,
        build_parser as build_general_parser,
        compute_dataset_auc,
        configure_logging,
        load_backbone,
        run_epoch,
        set_random_seed,
    )
    from .heads_chemprior_peak_latentdiff_stage1 import (
        BaselineEquivalentChemPriorLatentDiffHead,
    )
    from .latent_diffusion_core import ContinuousLatentDiffusion
    from .latent_diffusion_denoiser import LatentDiffusionMLPDenoiser
    from .latent_diffusion_denoiser_stage1 import GlobalLatentDiffusionMLPDenoiser
    from .latent_qgate_stage1 import (
        apply_global_qgate,
        apply_qgate,
        compute_class_stats_for_qgate,
        compute_global_stats_for_qgate,
    )


LOGGER = logging.getLogger(__name__)
EXPERIMENT_SUFFIX = "baselineeq_chemprior_peak_latentdiff_stage1"
STAGE_B_NUM_TIMESTEPS = 50


def resolve_diffusion_num_timesteps(args) -> int:
    value = int(getattr(args, "diffusion_num_timesteps", STAGE_B_NUM_TIMESTEPS))
    if value <= 0:
        raise ValueError("diffusion_num_timesteps must be greater than 0")
    return value


def build_parser(dataset_name: str):
    """
    在通用 parser 上补充第一阶段 latent diffusion 相关参数。
    """
    parser = build_general_parser(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    tasks_dir = os.path.dirname(os.path.dirname(__file__))

    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} latent diffusion stage1 migration"
    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        epochs=40,
        batch_size=64,
        lr=spec.lr,
        warmup_ratio=spec.warmup_ratio,
        pooler_dropout=spec.pooler_dropout,
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

    parser.add_argument("--synthetic-rho", type=float, default=0.05)
    parser.add_argument("--use-qgate", dest="use_qgate", action="store_true")
    parser.add_argument("--no-qgate", dest="use_qgate", action="store_false")
    parser.set_defaults(use_qgate=True)
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
    parser.add_argument(
        "--stage-c-logit-mix-grid",
        type=str,
        default="",
        help="可选的 Stage A / Stage C logit 混合 beta 网格，例如 0,0.1,0.2,0.35,0.5,0.65,0.8,1",
    )
    parser.add_argument("--stage-c-mix-prefer-lower-beta-on-auc-tie", dest="stage_c_mix_prefer_lower_beta_on_auc_tie", action="store_true")
    parser.add_argument("--no-stage-c-mix-prefer-lower-beta-on-auc-tie", dest="stage_c_mix_prefer_lower_beta_on_auc_tie", action="store_false")
    parser.set_defaults(stage_c_mix_prefer_lower_beta_on_auc_tie=False)
    parser.add_argument(
        "--stage-c-mix-tie-auc-eps",
        type=float,
        default=5e-4,
        help="中文注释：Stage C mix 选择时，把验证 AUC 视为近似持平的阈值；阈值内优先更保守的小 beta。",
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
            diffusion_epochs=40,
            diffusion_batch_size=32,
            stage_c_epochs=40,
            stage_c_batch_size=32,
            qgate_quantile=0.99,
            stage_c_logit_mix_grid="0,0.1,0.2,0.35,0.5,0.65,0.8,1",
            stage_c_mix_prefer_lower_beta_on_auc_tie=True,
            stage_c_mix_tie_auc_eps=8e-4,
        )
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    """
    为第一阶段迁移实验生成独立输出路径。
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
        "stage_c_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_c_head_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.pt"),
    }


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """
    批量控制模块参数是否参与训练。
    """
    for parameter in module.parameters():
        parameter.requires_grad = bool(flag)


def log_run_header(args, spec: DatasetSpec, device: torch.device, run_paths: Dict[str, str]) -> None:
    """
    记录第一阶段迁移实验头信息。
    """
    LOGGER.info("=" * 72)
    LOGGER.info("%s latent diffusion stage1", spec.name.upper())
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
    LOGGER.info(
        "  stage_c_logit_mix_grid:    %s",
        args.stage_c_logit_mix_grid if str(args.stage_c_logit_mix_grid).strip() else "disabled",
    )
    LOGGER.info("  diffusion_num_timesteps:   %s", STAGE_B_NUM_TIMESTEPS)
    LOGGER.info("=" * 72)


def build_head(args, spec: DatasetSpec, embed_dim: int, device: torch.device):
    """
    构建第一阶段并行 head。
    """
    return BaselineEquivalentChemPriorLatentDiffHead(
        in_dim=embed_dim,
        out_dim=spec.output_dim,
        desc_hidden=args.desc_hidden,
        dropout=args.pooler_dropout,
        desc_dropout=args.desc_dropout,
        fusion_dropout=args.fusion_dropout,
        use_desc_layernorm=args.use_desc_layernorm,
    ).to(device)


def build_stage_c_head(head: BaselineEquivalentChemPriorLatentDiffHead):
    """
    为 Stage C 复制一个结构相同的新 head，只训练 out_proj。
    """
    head_stage_c = copy.deepcopy(head)
    set_requires_grad(head_stage_c, False)
    set_requires_grad(head_stage_c.out_proj, True)
    return head_stage_c


def load_stage_a_best(
    model: nn.Module,
    head: BaselineEquivalentChemPriorLatentDiffHead,
    path: str,
    device: torch.device,
) -> None:
    """
    重新加载 Stage A 最优参数。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])


def load_stage_c_best(
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
    path: str,
    device: torch.device,
) -> None:
    """
    重新加载 Stage C 最优 out_proj。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head_stage_c.out_proj.load_state_dict(checkpoint["out_proj"])


def parse_stage_c_logit_mix_grid(raw_value: str) -> List[float]:
    """
    解析 Stage A / Stage C logit 混合 beta 网格。
    """
    raw_text = str(raw_value).strip()
    if not raw_text:
        return []

    values: List[float] = []
    for part in raw_text.split(","):
        token = part.strip()
        if not token:
            continue
        beta = float(token)
        if beta < 0.0 or beta > 1.0:
            raise ValueError("stage_c_logit_mix_grid 中的 beta 必须位于 [0, 1]")
        values.append(float(beta))

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
    中文注释：把最终 beta 映射成可审计模式，便于 ClinTox 复盘最终到底用了哪条头链。
    """
    beta = float(beta)
    if abs(beta) <= 1e-8:
        return "stage_a"
    if abs(beta - 1.0) <= 1e-8:
        return "stage_c"
    return "mix"


def should_replace_stage_c_mix_candidate(
    *,
    beta: float,
    val_auc: float,
    val_loss: float,
    best_state: Optional[Dict[str, float]],
    args,
) -> bool:
    """
    Stage C mix 固定按验证 AUC 优先；如 AUC 基本持平，则优先更保守的较小 beta。
    """
    if best_state is None:
        return True

    auc_eps = float(getattr(args, "stage_c_mix_tie_auc_eps", 5e-4))
    loss_eps = 1e-4
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


def run_stage_a_baseline(
    spec: DatasetSpec,
    model: nn.Module,
    head: BaselineEquivalentChemPriorLatentDiffHead,
    train_loader,
    valid_loader,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage A：完全复用 general 骨架的训练逻辑。
    """
    if args.freeze_backbone:
        set_requires_grad(model, False)

    optimizer, scheduler, num_training_steps, num_warmup_steps = build_optimizer_and_scheduler(
        args, model, head, train_loader
    )
    best_val_auc = float("-inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    LOGGER.info("[Stage A] total steps:  %s", num_training_steps)
    LOGGER.info("[Stage A] warmup steps: %s", num_warmup_steps)

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
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
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
        if epoch >= args.min_best_epoch and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            torch.save(
                {
                    "dataset": spec.name,
                    "epoch": int(epoch),
                    "val_auc": float(val_auc),
                    "backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "args": vars(args),
                },
                run_paths["stage_a_ckpt_path"],
            )
            LOGGER.info("[Stage A] new best val_auc=%.4f at epoch %d", val_auc, epoch)

    load_stage_a_best(model=model, head=head, path=run_paths["stage_a_ckpt_path"], device=device)
    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_ckpt_path": run_paths["stage_a_ckpt_path"],
        "history": history,
    }


@torch.no_grad()
def extract_fused_bank(
    model: nn.Module,
    head: BaselineEquivalentChemPriorLatentDiffHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从真实 split 中抽取 fused latent 和完整标签张量。
    """
    model.eval()
    head.eval()
    fused_list = []
    target_list = []

    for step_idx, (tokens, targets, smiles_batch) in enumerate(loader, start=1):
        if max_batches > 0 and step_idx > max_batches:
            break
        tokens = tokens.to(device)
        hidden_states, _ = model(
            src_tokens=tokens,
            src_lengths=None,
            features_only=True,
            levenshtein=False,
        )
        fused = head.extract_fused_latent(hidden_states, smiles=smiles_batch, detach=True)
        fused_list.append(fused.cpu())
        target_list.append(targets.cpu())

    if not fused_list:
        raise RuntimeError("未能从 loader 中抽取任何 fused latent。")

    return torch.cat(fused_list, dim=0), torch.cat(target_list, dim=0)


def build_binary_train_class_counts(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    仅用于 BACE 的 train split 类别计数。
    """
    labels = labels.view(-1).long()
    return torch.bincount(labels, minlength=num_classes).long()


def build_synthetic_count(total_real: int, rho: float) -> int:
    """
    根据 rho 计算 synthetic 总量。
    """
    return max(0, int(round(float(total_real) * float(rho))))


def build_binary_synthetic_class_counts(train_class_counts: torch.Tensor, rho: float) -> torch.Tensor:
    """
    根据 BACE train 类别分布生成 synthetic 类别配额。
    """
    total_train = int(train_class_counts.sum().item())
    total_synth = build_synthetic_count(total_train, rho)
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


def build_diffusion_model(args, latent_dim: int, is_binary_conditioned: bool, device: torch.device):
    """
    构建 Stage B diffusion。
    """
    if is_binary_conditioned:
        denoiser = LatentDiffusionMLPDenoiser(
            latent_dim=latent_dim,
            num_classes=2,
            time_embed_dim=args.diffusion_time_embed_dim,
            cond_dim=args.diffusion_cond_dim,
            hidden_dim=args.diffusion_hidden_dim,
            num_blocks=args.diffusion_num_blocks,
            dropout=args.diffusion_dropout,
        )
    else:
        denoiser = GlobalLatentDiffusionMLPDenoiser(
            latent_dim=latent_dim,
            time_embed_dim=args.diffusion_time_embed_dim,
            cond_dim=args.diffusion_cond_dim,
            hidden_dim=args.diffusion_hidden_dim,
            num_blocks=args.diffusion_num_blocks,
            dropout=args.diffusion_dropout,
        )

    diffusion = ContinuousLatentDiffusion(
        denoiser=denoiser,
        latent_dim=latent_dim,
        num_timesteps=resolve_diffusion_num_timesteps(args),
        beta_schedule=args.diffusion_beta_schedule,
        prediction_type=args.diffusion_prediction_type,
    ).to(device)
    return diffusion


def _iter_diffusion_loader(loader, max_batches: int):
    """
    截断 diffusion 训练/验证 batch 数。
    """
    for step_idx, batch in enumerate(loader, start=1):
        if max_batches > 0 and step_idx > max_batches:
            break
        yield batch


def train_diffusion_stage(
    diffusion: ContinuousLatentDiffusion,
    train_loader,
    valid_loader,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage B：训练 latent diffusion。
    """
    optimizer = AdamW(
        diffusion.parameters(),
        lr=args.diffusion_lr,
        weight_decay=args.diffusion_weight_decay,
    )
    best_val_loss: Optional[float] = None
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.diffusion_epochs + 1):
        diffusion.train()
        train_losses = []
        for z0, cond in _iter_diffusion_loader(train_loader, args.max_diffusion_train_batches):
            z0 = z0.to(device)
            cond = cond.to(device)
            loss_dict = diffusion.training_loss(z0=z0, c=cond)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.diffusion_grad_clip is not None and args.diffusion_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.diffusion_grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        diffusion.eval()
        valid_losses = []
        with torch.no_grad():
            for z0, cond in _iter_diffusion_loader(valid_loader, args.max_diffusion_valid_batches):
                z0 = z0.to(device)
                cond = cond.to(device)
                loss_dict = diffusion.training_loss(z0=z0, c=cond)
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
        LOGGER.info(
            "[Stage B] epoch=%d/%d train_loss=%.6f val_loss=%.6f",
            epoch,
            args.diffusion_epochs,
            train_loss,
            val_loss,
        )

        if epoch >= args.diffusion_min_best_epoch and (
            best_val_loss is None or val_loss < best_val_loss
        ):
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            torch.save(
                {
                    "diffusion": diffusion.state_dict(),
                    "epoch": int(epoch),
                    "best_val_loss": float(val_loss),
                },
                run_paths["stage_b_ckpt_path"],
            )
            LOGGER.info("[Stage B] new best val_loss=%.6f at epoch %d", val_loss, epoch)

    checkpoint = torch.load(run_paths["stage_b_ckpt_path"], map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint["diffusion"])
    return {
        "diffusion": diffusion,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_ckpt_path": run_paths["stage_b_ckpt_path"],
        "history": history,
    }


def build_binary_diffusion_dataloaders(
    fused_train: torch.Tensor,
    y_train: torch.Tensor,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    batch_size: int,
):
    """
    BACE 的 conditioned diffusion dataloader。
    """
    train_loader = DataLoader(
        TensorDataset(fused_train.float(), y_train.view(-1).long()),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float(), y_valid.view(-1).long()),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader


def build_multitask_diffusion_dataloaders(
    fused_train: torch.Tensor,
    fused_valid: torch.Tensor,
    batch_size: int,
):
    """
    多任务保守版 diffusion dataloader。
    """
    train_cond = torch.zeros((fused_train.shape[0],), dtype=torch.long)
    valid_cond = torch.zeros((fused_valid.shape[0],), dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(fused_train.float(), train_cond),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(fused_valid.float(), valid_cond),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader


def generate_binary_synthetic_bank(
    diffusion: ContinuousLatentDiffusion,
    train_class_counts: torch.Tensor,
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BACE：按照 train 类别分布生成 synthetic fused latent。
    """
    synth_class_counts = build_binary_synthetic_class_counts(train_class_counts, args.synthetic_rho)
    total_synth = int(synth_class_counts.sum().item())
    if total_synth <= 0:
        return (
            torch.zeros((0, diffusion.latent_dim), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            synth_class_counts,
        )

    label_chunks = []
    for class_id, class_count in enumerate(synth_class_counts.tolist()):
        if class_count > 0:
            label_chunks.append(torch.full((class_count,), class_id, dtype=torch.long))
    synthetic_labels = torch.cat(label_chunks, dim=0)
    synthetic_labels = synthetic_labels[torch.randperm(synthetic_labels.shape[0])]

    synthetic_latents = []
    for start in range(0, total_synth, args.diffusion_batch_size):
        end = min(start + args.diffusion_batch_size, total_synth)
        batch_labels = synthetic_labels[start:end].to(device)
        batch_latents = diffusion.sample(class_labels=batch_labels, device=device)
        synthetic_latents.append(batch_latents.cpu())

    return torch.cat(synthetic_latents, dim=0), synthetic_labels, synth_class_counts


def generate_multitask_synthetic_bank(
    diffusion: ContinuousLatentDiffusion,
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    多任务保守版 synthetic 生成。
    """
    total_synth = build_synthetic_count(int(fused_real_train.shape[0]), args.synthetic_rho)
    if total_synth <= 0:
        return (
            torch.zeros((0, fused_real_train.shape[1]), dtype=torch.float32),
            torch.zeros((0, y_real_train.shape[1]), dtype=torch.float32),
        )

    anchor_indices = torch.randint(
        low=0,
        high=int(fused_real_train.shape[0]),
        size=(total_synth,),
        dtype=torch.long,
    )
    anchor_latents = fused_real_train[anchor_indices].float().to(device)
    anchor_targets = y_real_train[anchor_indices].float().clone()

    t = torch.randint(
        low=0,
        high=diffusion.num_timesteps,
        size=(total_synth,),
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(anchor_latents)
    z_t = diffusion.q_sample(z0=anchor_latents, t=t, noise=noise)
    cond = torch.zeros((total_synth,), device=device, dtype=torch.long)
    outputs = diffusion.p_mean_variance(z_t=z_t, t=t, c=cond)
    synthetic_latents = outputs["pred_x0"].detach().cpu()

    return synthetic_latents, anchor_targets


def apply_bace_qgate_if_needed(
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn: torch.Tensor,
    y_syn: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    BACE：class-wise QGate。
    """
    if getattr(args, "no_qgate", False):
        before = (
            build_binary_train_class_counts(y_syn, num_classes=2)
            if y_syn.numel() > 0
            else torch.zeros(2, dtype=torch.long)
        )
        n_total = int(fused_syn.shape[0])
        summary = {
            "before": n_total,
            "after": n_total,
            "kept_ratio": 1.0,
            "rejection_rate": 0.0,
            "before_count": n_total,
            "after_count": n_total,
            "keep_ratio": 1.0,
            "per_class_before": {0: int(before[0].item()), 1: int(before[1].item())},
            "per_class_after": {0: int(before[0].item()), 1: int(before[1].item())},
        }
        print(
            f"[NoQC-Binary] QGate disabled. "
            f"All {n_total} synthetic latents retained."
        )
        return fused_syn, y_syn, summary

    if not args.use_qgate or fused_syn.shape[0] == 0:
        before = (
            build_binary_train_class_counts(y_syn, num_classes=2)
            if y_syn.numel() > 0
            else torch.zeros(2, dtype=torch.long)
        )
        return fused_syn, y_syn, {
            "before_count": int(fused_syn.shape[0]),
            "after_count": int(fused_syn.shape[0]),
            "keep_ratio": 1.0 if fused_syn.shape[0] > 0 else 0.0,
            "per_class_before": {0: int(before[0].item()), 1: int(before[1].item())},
            "per_class_after": {0: int(before[0].item()), 1: int(before[1].item())},
        }

    class_stats = compute_class_stats_for_qgate(
        fused_real_train=fused_real_train,
        y_real_train=y_real_train.view(-1),
        num_classes=2,
        quantile=args.qgate_quantile,
        eps=args.qgate_cov_eps,
    )
    fused_syn, y_syn, summary = apply_qgate(fused_syn=fused_syn, y_syn=y_syn, class_stats=class_stats)
    n_before = int(summary["before_count"])
    n_after = int(summary["after_count"])
    rejection_rate = 1.0 - n_after / max(n_before, 1)
    summary["before"] = n_before
    summary["after"] = n_after
    summary["kept_ratio"] = float(summary.get("keep_ratio", 0.0))
    summary["rejection_rate"] = rejection_rate
    print(
        f"[QGate-Binary] before={n_before}, "
        f"after={n_after}, "
        f"rejection_rate={rejection_rate:.2%}"
    )
    return fused_syn, y_syn, summary


def apply_multitask_qgate_if_needed(
    fused_real_train: torch.Tensor,
    fused_syn: torch.Tensor,
    y_syn: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    多任务：global QGate。
    """
    if getattr(args, "no_qgate", False):
        n_total = int(fused_syn.shape[0])
        summary = {
            "before": n_total,
            "after": n_total,
            "kept_ratio": 1.0,
            "rejection_rate": 0.0,
            "before_count": n_total,
            "after_count": n_total,
            "keep_ratio": 1.0,
        }
        print(
            f"[NoQC-Multitask] QGate disabled. "
            f"All {n_total} synthetic latents retained."
        )
        return fused_syn, y_syn, summary

    if not args.use_qgate or fused_syn.shape[0] == 0:
        return fused_syn, y_syn, {
            "before_count": int(fused_syn.shape[0]),
            "after_count": int(fused_syn.shape[0]),
            "keep_ratio": 1.0 if fused_syn.shape[0] > 0 else 0.0,
        }

    global_stats = compute_global_stats_for_qgate(
        fused_real_train=fused_real_train,
        quantile=args.qgate_quantile,
        eps=args.qgate_cov_eps,
    )
    fused_kept, keep_mask, summary = apply_global_qgate(fused_syn=fused_syn, global_stats=global_stats)
    y_kept = y_syn[keep_mask.cpu()]
    n_before = int(summary["before_count"])
    n_after = int(summary["after_count"])
    rejection_rate = 1.0 - n_after / max(n_before, 1)
    summary["before"] = n_before
    summary["after"] = n_after
    summary["kept_ratio"] = float(summary.get("keep_ratio", 0.0))
    summary["rejection_rate"] = rejection_rate
    print(
        f"[QGate-Multitask] before={n_before}, "
        f"after={n_after}, "
        f"rejection_rate={rejection_rate:.2%}"
    )
    return fused_kept, y_kept, summary


def evaluate_latent_head(
    spec: DatasetSpec,
    head: BaselineEquivalentChemPriorLatentDiffHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    只在 fused latent 上评估 Stage C 头。
    """
    head.eval()
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
            logits = head.forward_from_fused(fused_batch)
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


def train_classifier_stage(
    spec: DatasetSpec,
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
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
    Stage C：冻结 backbone，仅训练 out_proj。
    """
    if fused_syn_train.shape[0] > 0:
        fused_train = torch.cat([fused_real_train, fused_syn_train], dim=0)
        y_train = torch.cat([y_real_train, y_syn_train], dim=0)
    else:
        fused_train = fused_real_train
        y_train = y_real_train

    train_loader = DataLoader(
        TensorDataset(fused_train.float(), y_train.float()),
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
    best_val_auc = float("-inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

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
                torch.nn.utils.clip_grad_norm_(head_stage_c.out_proj.parameters(), args.stage_c_grad_clip)
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

        val_loss, val_auc = evaluate_latent_head(
            spec=spec,
            head=head_stage_c,
            loader=valid_loader,
            device=device,
            max_batches=args.max_stagec_valid_batches,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
            }
        )
        LOGGER.info(
            "[Stage C] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            epoch,
            args.stage_c_epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
        )
        if epoch >= args.stage_c_min_best_epoch and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            torch.save(
                {
                    "epoch": int(epoch),
                    "val_auc": float(val_auc),
                    "out_proj": head_stage_c.out_proj.state_dict(),
                },
                run_paths["stage_c_ckpt_path"],
            )
            LOGGER.info("[Stage C] new best val_auc=%.4f at epoch %d", val_auc, epoch)

    load_stage_c_best(head_stage_c=head_stage_c, path=run_paths["stage_c_ckpt_path"], device=device)
    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_ckpt_path": run_paths["stage_c_ckpt_path"],
        "history": history,
        "train_size_real": int(fused_real_train.shape[0]),
        "train_size_synth": int(fused_syn_train.shape[0]),
        "train_size_total": int(fused_train.shape[0]),
    }


def evaluate_stage_c_logit_mix(
    spec: DatasetSpec,
    head_stage_a: BaselineEquivalentChemPriorLatentDiffHead,
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
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
    head_stage_a: BaselineEquivalentChemPriorLatentDiffHead,
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
    fused_valid: torch.Tensor,
    y_valid: torch.Tensor,
    args,
    device: torch.device,
) -> Dict[str, Any]:
    """
    在验证集上选择 Stage A / Stage C 的最优 logit 混合 beta。
    """
    beta_grid = parse_stage_c_logit_mix_grid(getattr(args, "stage_c_logit_mix_grid", ""))
    if not beta_grid:
        return {
            "enabled": False,
            "selected_beta": 1.0,
            "grid": [],
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
        row = {
            "beta": float(beta),
            "val_auc": float(metrics["val_auc"]),
            "val_loss": float(metrics["val_loss"]),
        }
        history.append(row)
        LOGGER.info(
            "[Stage C Mix] beta=%.3f val_loss=%.4f val_auc=%.4f",
            beta,
            metrics["val_loss"],
            metrics["val_auc"],
        )

        if should_replace_stage_c_mix_candidate(
            beta=beta,
            val_auc=metrics["val_auc"],
            val_loss=metrics["val_loss"],
            best_state=best_state,
            args=args,
        ):
            best_state = {
                "beta": float(beta),
                "best_val_auc": float(metrics["val_auc"]),
                "best_val_loss": float(metrics["val_loss"]),
            }

    if best_state is None:
        raise RuntimeError("Stage C logit mix 选择阶段未生成有效 beta。")

    LOGGER.info(
        "[Stage C Mix] selected beta=%.3f val_loss=%.4f val_auc=%.4f",
        best_state["beta"],
        best_state["best_val_loss"],
        best_state["best_val_auc"],
    )
    return {
        "enabled": True,
        "selected_beta": float(best_state["beta"]),
        "selected_mode": infer_eval_mode_from_beta(float(best_state["beta"])),
        "grid": [float(beta) for beta in beta_grid],
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "history": history,
    }


def evaluate_full_model_with_stage_c_mix(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: BaselineEquivalentChemPriorLatentDiffHead,
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
    loader,
    device: torch.device,
    mix_beta: float = 1.0,
    max_batches: int = 0,
) -> Tuple[float, float]:
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
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("测试阶段没有执行任何 batch。")

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, auc


def evaluate_full_model(
    spec: DatasetSpec,
    model: nn.Module,
    head_for_feature: BaselineEquivalentChemPriorLatentDiffHead,
    head_stage_c: BaselineEquivalentChemPriorLatentDiffHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """
    使用完整推理链评估：backbone -> fused -> Stage C out_proj。
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
            fused = head_for_feature.extract_fused_latent(hidden_states, smiles=smiles_batch, detach=False)
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


def run_dataset_stage1_experiment(dataset_name: str):
    """
    运行单个数据集的第一阶段迁移实验。
    """
    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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

    LOGGER.info("[2] Building latentdiff stage1 head...")
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
    train_smiles = train_dataset.collect_smiles()
    head.fit_normalizer(train_smiles, device)

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
        fused_real_train=fused_train,
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

    LOGGER.info("[9] Evaluating stage_a / stage_c / selected blend on test split...")
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
        },
        "stage_c": stage_c,
        "stage_c_logit_mix": stage_c_logit_mix,
        "final_eval_mode": final_eval_mode,
        "final_eval": {
            "mode": final_eval_mode,
            "selected_beta": float(selected_beta),
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
    torch.save(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s stage1 final results", dataset_name.upper())
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
    薄入口，供各数据集脚本调用。
    """
    return run_dataset_stage1_experiment(dataset_name)
