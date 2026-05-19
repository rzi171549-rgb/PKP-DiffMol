# -*- coding: utf-8 -*-
"""
MUV formal4 protocol fix v2.

Design goals:
1. Keep formal4 architecture unchanged: parallel encoder + chemprior + semantic tunev2 + diffusion.
2. Only repair protocol-level instability for MUV in a new parallel file.
3. Prefer Stage A selection by pure val_auc.
4. Keep Stage C as an auxiliary adaptation and support conservative final evaluation by blend.
"""

import copy
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    configure_logging,
    load_backbone,
    set_random_seed,
    split_decay_param_groups,
)
from chemprior_peak_latentdiff_stage1 import (  # noqa: E402
    STAGE_B_NUM_TIMESTEPS,
    apply_multitask_qgate_if_needed,
    build_binary_diffusion_dataloaders,
    build_binary_train_class_counts,
    build_diffusion_model,
    build_multitask_diffusion_dataloaders,
    extract_fused_bank,
    generate_binary_synthetic_bank,
    generate_multitask_synthetic_bank,
    set_requires_grad,
    train_diffusion_stage,
)
from heads_chemprior_peak_qsarprompt_semantic_tunev2_latentdiff import (  # noqa: E402
    PoolPoolerBaseTuneV2LatentDiffHead,
)
from latent_qgate_stage1 import apply_qgate, compute_class_stats_for_qgate  # noqa: E402
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (  # noqa: E402
    build_optimizer_param_groups,
    post_optimizer_step,
    prepare_epoch_training_policy,
    str2bool,
    summarize_optimizer_groups,
)
from train_bbbp_chemprior_parallel import build_scheduler  # noqa: E402


LOGGER = logging.getLogger(__name__)
EXPERIMENT_SUFFIX = "baselineeq_chemprior_peak_tunev2_latentdiff_muvformal4fix_v2"
DEFAULT_BLEND_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def save_json(payload: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_parser(dataset_name: str):
    parser = build_general_parser(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} chemprior parallel latentdiff muv formal4 fix v2"
    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        lr=spec.lr,
        warmup_ratio=spec.warmup_ratio,
        pooler_dropout=spec.pooler_dropout,
        best_metric="val_auc",
    )

    parser.add_argument("--text-lr", "--text_lr", type=float, default=1e-5)
    parser.add_argument("--semantic-proj-lr", "--semantic_proj_lr", type=float, default=5e-5)
    parser.add_argument("--alpha-lr", "--alpha_lr", type=float, default=1e-5)
    parser.add_argument("--text-weight-decay", "--text_weight_decay", type=float, default=None)
    parser.add_argument("--semantic-proj-weight-decay", "--semantic_proj_weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha-weight-decay", "--alpha_weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument(
        "--best-metric",
        type=str,
        choices=["val_auc", "val_loss", "hybrid"],
        default="val_auc",
    )
    parser.add_argument("--hybrid-best-score-alpha", "--hybrid_best_score_alpha", type=float, default=0.6)
    parser.add_argument("--checkpoint-score-eps", type=float, default=1e-4)
    parser.add_argument("--checkpoint-tie-auc-eps", type=float, default=5e-4)
    parser.add_argument("--checkpoint-tie-loss-eps", type=float, default=1e-4)
    parser.add_argument("--desc-text-fusion-dropout", "--desc_text_fusion_dropout", type=float, default=0.1)
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

    parser.add_argument("--synthetic-rho", type=float, default=0.03)
    parser.add_argument(
        "--use-qgate",
        "--use_qgate",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--qgate-quantile", type=float, default=0.90)
    parser.add_argument("--qgate-cov-eps", type=float, default=1e-6)

    parser.add_argument("--stage-c-epochs", type=int, default=15)
    parser.add_argument("--stage-c-batch-size", type=int, default=256)
    parser.add_argument("--stage-c-lr", type=float, default=3e-4)
    parser.add_argument("--stage-c-weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-c-grad-clip", type=float, default=1.0)
    parser.add_argument("--stage-c-min-best-epoch", type=int, default=1)
    parser.add_argument(
        "--stage-c-train-scope",
        type=str,
        choices=["out_proj", "head_light", "full_head"],
        default="head_light",
    )
    parser.add_argument(
        "--final-eval-mode",
        type=str,
        choices=["stage_a_only", "stage_c_only", "val_tuned_blend"],
        default="val_tuned_blend",
    )
    parser.add_argument(
        "--blend-beta-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_BLEND_GRID),
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
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    root_output_dir = os.path.abspath(args.output_dir)
    run_name = f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed{args.seed}"
    run_dir = os.path.join(root_output_dir, run_name)
    return {
        "root_output_dir": root_output_dir,
        "run_name": run_name,
        "run_dir": run_dir,
        "log_file": os.path.join(run_dir, f"{run_name}.log"),
        "stage_a_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_a_best.pt"),
        "stage_b_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_b_diffusion_best.pt"),
        "stage_c_ckpt_path": os.path.join(run_dir, f"{run_name}_stage_c_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def build_head(args, spec: DatasetSpec, embed_dim: int, device: torch.device):
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
            f"head output dim mismatch: expected {spec.output_dim}, got {head.out_proj.out_features}"
        )
    return head


def _stage_c_trainable_names(head: PoolPoolerBaseTuneV2LatentDiffHead, scope: str) -> List[str]:
    if scope == "full_head":
        return [name for name, _ in head.named_parameters()]

    base_names = ["out_proj"]
    if scope == "head_light":
        base_names.extend(
            [
                "pooler_layernorm",
                "post_fusion_desc_layernorm",
                "fused_layernorm",
                "text_to_desc_proj",
                "desc_text_gate",
                "fusion_alpha",
            ]
        )

    names: List[str] = []
    for name, _ in head.named_parameters():
        for prefix in base_names:
            if name == prefix or name.startswith(prefix + "."):
                names.append(name)
                break
    return names


def build_stage_c_head(
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    scope: str,
) -> Tuple[PoolPoolerBaseTuneV2LatentDiffHead, List[str]]:
    head_stage_c = copy.deepcopy(head)
    set_requires_grad(head_stage_c, False)
    trainable_names = _stage_c_trainable_names(head_stage_c, scope)
    trainable_name_set = set(trainable_names)
    for name, parameter in head_stage_c.named_parameters():
        parameter.requires_grad = name in trainable_name_set
    if not trainable_names:
        raise RuntimeError(f"Stage C train scope produced no trainable params: {scope}")
    return head_stage_c, sorted(trainable_names)


def load_stage_a_best(
    model: nn.Module,
    head: PoolPoolerBaseTuneV2LatentDiffHead,
    path: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    post_optimizer_step(head)
    return checkpoint


def load_stage_c_best(
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    path: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head_stage_c.load_state_dict(checkpoint["head"])
    post_optimizer_step(head_stage_c)
    return checkpoint


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
        raise RuntimeError("Stage A ran zero batches")

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
    if args.freeze_backbone:
        set_requires_grad(model, False)

    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("Stage A found no trainable param groups")

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

    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    effective_min_best_epoch = min(max(1, int(args.min_best_epoch)), int(args.epochs))

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
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
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

        if epoch < effective_min_best_epoch:
            continue
        if best_state is None or float(val_auc) > float(best_state["best_val_auc"]):
            best_state = {
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
                "best_metric_mode": "val_auc",
            }
            torch.save(
                {
                    "dataset": spec.name,
                    "epoch": int(epoch),
                    "best_metric": "val_auc",
                    "best_score": float(val_auc),
                    "best_metric_value": float(val_auc),
                    "backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "args": vars(args),
                },
                run_paths["stage_a_ckpt_path"],
            )
            LOGGER.info("[Stage A] new best checkpoint epoch=%d val_auc=%.4f", epoch, val_auc)

    if best_state is None:
        raise RuntimeError("Stage A ended without a best checkpoint")

    load_stage_a_best(model=model, head=head, path=run_paths["stage_a_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_metric_mode": str(best_state["best_metric_mode"]),
        "best_ckpt_path": run_paths["stage_a_ckpt_path"],
        "optimizer_groups": optimizer_group_summary,
        "history": history,
    }


def log_run_header(args, spec: DatasetSpec, device: torch.device, run_paths: Dict[str, str]) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info("%s chemprior parallel latentdiff muv formal4 fix v2", spec.name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  device:                    %s", device)
    LOGGER.info("  task_type:                 %s", spec.task_type)
    LOGGER.info("  output_dim:                %s", spec.output_dim)
    LOGGER.info("  run_dir:                   %s", run_paths["run_dir"])
    LOGGER.info("  log_file:                  %s", run_paths["log_file"])
    LOGGER.info("  stage_a_ckpt:              %s", run_paths["stage_a_ckpt_path"])
    LOGGER.info("  stage_b_ckpt:              %s", run_paths["stage_b_ckpt_path"])
    LOGGER.info("  stage_c_ckpt:              %s", run_paths["stage_c_ckpt_path"])
    LOGGER.info("  summary_path:              %s", run_paths["summary_path"])
    LOGGER.info("  epochs:                    %s", args.epochs)
    LOGGER.info("  batch_size:                %s", args.batch_size)
    LOGGER.info("  best_metric:               %s", args.best_metric)
    LOGGER.info("  scheduler:                 %s", args.scheduler)
    LOGGER.info("  warmup_ratio:              %s", args.warmup_ratio)
    LOGGER.info("  synthetic_rho:             %s", args.synthetic_rho)
    LOGGER.info("  use_qgate:                 %s", args.use_qgate)
    LOGGER.info("  qgate_quantile:            %s", args.qgate_quantile)
    LOGGER.info("  stage_c_train_scope:       %s", args.stage_c_train_scope)
    LOGGER.info("  final_eval_mode:           %s", args.final_eval_mode)
    LOGGER.info("  diffusion_num_timesteps:   %s", args.diffusion_num_timesteps)
    LOGGER.info("=" * 72)


def apply_bace_qgate_if_needed(
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    fused_syn: torch.Tensor,
    y_syn: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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
    return fused_syn, y_syn, summary


def _make_stage_c_optimizer(
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    args,
) -> AdamW:
    decay_params, no_decay_params = split_decay_param_groups(head_stage_c)
    param_groups: List[Dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(args.stage_c_lr),
                "weight_decay": float(args.stage_c_weight_decay),
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": float(args.stage_c_lr),
                "weight_decay": 0.0,
            }
        )
    if not param_groups:
        raise RuntimeError("Stage C optimizer received no trainable params")
    return AdamW(param_groups, lr=args.stage_c_lr, weight_decay=args.stage_c_weight_decay)


def _make_synth_loader(
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    batch_size: int,
) -> Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]]:
    if fused_syn_train.shape[0] <= 0:
        return None
    return DataLoader(
        TensorDataset(fused_syn_train.float(), y_syn_train.float()),
        batch_size=batch_size,
        shuffle=True,
    )


def _grad_params(module: nn.Module) -> List[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


@torch.no_grad()
def evaluate_stage_c_full_model(
    spec: DatasetSpec,
    model: nn.Module,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.eval()
    head_stage_c.eval()

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []
    for step_idx, (tokens, targets_batch, smiles_batch) in enumerate(loader, start=1):
        if max_batches > 0 and step_idx > max_batches:
            break
        tokens = tokens.to(device)
        targets_batch = targets_batch.to(device)
        hidden_states, _ = model(
            src_tokens=tokens,
            src_lengths=None,
            features_only=True,
            levenshtein=False,
        )
        logits = head_stage_c(hidden_states, smiles_batch)
        loss, probs = _compute_loss_and_probs(spec, logits, targets_batch)
        total_loss += float(loss.item())
        batch_count += 1
        all_preds.append(probs.detach().cpu().numpy())
        all_targets.append(targets_batch.detach().cpu().numpy())

    if batch_count == 0:
        raise RuntimeError("Stage C validation ran zero batches")
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    auc = compute_dataset_auc(spec, preds, targets)
    return total_loss / batch_count, float(auc)


@torch.no_grad()
def collect_logits_and_targets(
    spec: DatasetSpec,
    model: nn.Module,
    head_stage_a: PoolPoolerBaseTuneV2LatentDiffHead,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, np.ndarray]:
    model.eval()
    head_stage_a.eval()
    head_stage_c.eval()

    logits_a_list: List[torch.Tensor] = []
    logits_c_list: List[torch.Tensor] = []
    probs_a_list: List[torch.Tensor] = []
    probs_c_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

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
        logits_a = head_stage_a(hidden_states, smiles_batch)
        logits_c = head_stage_c(hidden_states, smiles_batch)
        _, probs_a = _compute_loss_and_probs(spec, logits_a, targets)
        _, probs_c = _compute_loss_and_probs(spec, logits_c, targets)

        logits_a_list.append(logits_a.detach().cpu())
        logits_c_list.append(logits_c.detach().cpu())
        probs_a_list.append(probs_a.detach().cpu())
        probs_c_list.append(probs_c.detach().cpu())
        targets_list.append(targets.detach().cpu())

    if not targets_list:
        raise RuntimeError("No batches available for logits collection")

    logits_a_cat = torch.cat(logits_a_list, dim=0)
    logits_c_cat = torch.cat(logits_c_list, dim=0)
    probs_a_cat = torch.cat(probs_a_list, dim=0)
    probs_c_cat = torch.cat(probs_c_list, dim=0)
    targets_cat = torch.cat(targets_list, dim=0)
    return {
        "logits_a": logits_a_cat.numpy(),
        "logits_c": logits_c_cat.numpy(),
        "probs_a": probs_a_cat.numpy(),
        "probs_c": probs_c_cat.numpy(),
        "targets": targets_cat.numpy(),
    }


def _probs_from_logits(spec: DatasetSpec, logits: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(logits)
    if spec.is_multitask:
        return torch.sigmoid(tensor).numpy()
    return torch.sigmoid(tensor.view(-1)).numpy()


def search_blend_beta(
    spec: DatasetSpec,
    val_outputs: Dict[str, np.ndarray],
    beta_grid: Sequence[float],
) -> Dict[str, float]:
    best_beta = 0.0
    best_auc = float("-inf")
    targets = val_outputs["targets"]
    logits_a = val_outputs["logits_a"]
    logits_c = val_outputs["logits_c"]

    for beta in beta_grid:
        beta = float(beta)
        blended_logits = (1.0 - beta) * logits_a + beta * logits_c
        blended_probs = _probs_from_logits(spec, blended_logits)
        auc = compute_dataset_auc(spec, blended_probs, targets)
        if auc > best_auc:
            best_auc = float(auc)
            best_beta = beta
    return {"beta": float(best_beta), "val_auc": float(best_auc)}


def evaluate_mode_auc(
    spec: DatasetSpec,
    outputs: Dict[str, np.ndarray],
    mode: str,
    blend_beta: float = 0.0,
) -> float:
    targets = outputs["targets"]
    if mode == "stage_a_only":
        return float(compute_dataset_auc(spec, outputs["probs_a"], targets))
    if mode == "stage_c_only":
        return float(compute_dataset_auc(spec, outputs["probs_c"], targets))
    if mode == "val_tuned_blend":
        blended_logits = (1.0 - blend_beta) * outputs["logits_a"] + blend_beta * outputs["logits_c"]
        blended_probs = _probs_from_logits(spec, blended_logits)
        return float(compute_dataset_auc(spec, blended_probs, targets))
    raise ValueError(f"Unsupported eval mode: {mode}")


def train_classifier_stage_protocolfix(
    spec: DatasetSpec,
    model: nn.Module,
    head_stage_c: PoolPoolerBaseTuneV2LatentDiffHead,
    train_loader,
    valid_loader,
    fused_syn_train: torch.Tensor,
    y_syn_train: torch.Tensor,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
    trainable_names: Sequence[str],
) -> Dict[str, Any]:
    set_requires_grad(model, False)
    optimizer = _make_stage_c_optimizer(head_stage_c=head_stage_c, args=args)
    grad_params = _grad_params(head_stage_c)
    synth_loader = _make_synth_loader(
        fused_syn_train=fused_syn_train,
        y_syn_train=y_syn_train,
        batch_size=args.stage_c_batch_size,
    )

    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.stage_c_epochs + 1):
        head_stage_c.train()
        total_loss = 0.0
        real_loss_total = 0.0
        synth_loss_total = 0.0
        batch_count = 0
        real_batch_count = 0
        synth_batch_count = 0
        all_preds = []
        all_targets = []

        synth_iter = iter(synth_loader) if synth_loader is not None else None

        for step_idx, (tokens, targets, smiles_batch) in enumerate(train_loader, start=1):
            if args.max_stagec_train_batches > 0 and step_idx > args.max_stagec_train_batches:
                break
            tokens = tokens.to(device)
            targets = targets.to(device)

            hidden_states, _ = model(
                src_tokens=tokens,
                src_lengths=None,
                features_only=True,
                levenshtein=False,
            )
            logits_real = head_stage_c(hidden_states, smiles_batch)
            real_loss, probs_real = _compute_loss_and_probs(spec, logits_real, targets)
            loss = real_loss
            synth_loss_value = None

            if synth_iter is not None:
                try:
                    fused_syn_batch, targets_syn_batch = next(synth_iter)
                except StopIteration:
                    synth_iter = iter(synth_loader)
                    fused_syn_batch, targets_syn_batch = next(synth_iter)

                fused_syn_batch = fused_syn_batch.to(device)
                targets_syn_batch = targets_syn_batch.to(device)
                logits_syn = head_stage_c.forward_from_fused(fused_syn_batch)
                synth_loss_value, _ = _compute_loss_and_probs(spec, logits_syn, targets_syn_batch)
                loss = real_loss + synth_loss_value

            optimizer.zero_grad()
            loss.backward()
            if args.stage_c_grad_clip is not None and args.stage_c_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=args.stage_c_grad_clip)
            optimizer.step()
            post_optimizer_step(head_stage_c)

            total_loss += float(loss.item())
            real_loss_total += float(real_loss.item())
            batch_count += 1
            real_batch_count += 1
            all_preds.append(probs_real.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            if synth_loss_value is not None:
                synth_loss_total += float(synth_loss_value.item())
                synth_batch_count += 1

        if batch_count == 0:
            raise RuntimeError("Stage C training ran zero batches")

        train_preds = np.concatenate(all_preds, axis=0)
        train_targets = np.concatenate(all_targets, axis=0)
        train_auc = compute_dataset_auc(spec, train_preds, train_targets)
        train_loss = total_loss / batch_count
        real_loss_mean = real_loss_total / max(1, real_batch_count)
        synth_loss_mean = synth_loss_total / max(1, synth_batch_count) if synth_batch_count > 0 else 0.0

        val_loss, val_auc = evaluate_stage_c_full_model(
            spec=spec,
            model=model,
            head_stage_c=head_stage_c,
            loader=valid_loader,
            device=device,
            max_batches=args.max_stagec_valid_batches,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "train_real_loss": float(real_loss_mean),
                "train_synth_loss": float(synth_loss_mean),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "used_synthetic": bool(synth_iter is not None),
            }
        )
        LOGGER.info(
            "[Stage C] epoch=%d/%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f use_synth=%s",
            epoch,
            args.stage_c_epochs,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
            synth_iter is not None,
        )

        if epoch < int(args.stage_c_min_best_epoch):
            continue
        if best_state is None or float(val_auc) > float(best_state["best_val_auc"]):
            best_state = {
                "best_epoch": int(epoch),
                "best_val_auc": float(val_auc),
                "best_val_loss": float(val_loss),
            }
            torch.save(
                {
                    "epoch": int(epoch),
                    "val_auc": float(val_auc),
                    "val_loss": float(val_loss),
                    "head": head_stage_c.state_dict(),
                    "train_scope": str(args.stage_c_train_scope),
                    "trainable_names": list(trainable_names),
                    "args": vars(args),
                },
                run_paths["stage_c_ckpt_path"],
            )
            LOGGER.info("[Stage C] new best checkpoint epoch=%d val_auc=%.4f", epoch, val_auc)

    if best_state is None:
        raise RuntimeError("Stage C ended without a best checkpoint")

    load_stage_c_best(head_stage_c=head_stage_c, path=run_paths["stage_c_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_ckpt_path": run_paths["stage_c_ckpt_path"],
        "history": history,
        "train_size_real": int(len(train_loader.dataset)),
        "train_size_synth": int(fused_syn_train.shape[0]),
        "train_size_total": int(len(train_loader.dataset) + fused_syn_train.shape[0]),
        "train_scope": str(args.stage_c_train_scope),
        "trainable_names": list(trainable_names),
    }


def run_tunev2_latentdiff_muvformal4fix_v2_experiment(dataset_name: str):
    spec = DATASET_SPECS[dataset_name]
    if dataset_name != "muv":
        raise ValueError("muvformal4fix_v2 is only intended for the MUV route")

    parser = build_parser(dataset_name)
    args = parser.parse_args()

    run_paths = resolve_run_paths(args, dataset_name)
    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)
    args.log_file = run_paths["log_file"]
    configure_logging(args.log_file)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            f"extract_fused_latent dim mismatch: expected {head.final_fused_dim}, got {fused_train.shape[1]}"
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
        synthetic_total_before_qgate = int(fused_syn.shape[0])
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
        synthetic_total_before_qgate = int(fused_syn.shape[0])
        fused_syn, y_syn, qgate_summary = apply_bace_qgate_if_needed(
            fused_real_train=fused_train,
            y_real_train=y_train_binary,
            fused_syn=fused_syn,
            y_syn=y_syn,
            args=args,
        )
        y_syn = y_syn.view(-1, 1).float()

    synthetic_total_after_qgate = int(fused_syn.shape[0])
    LOGGER.info(
        "[QGate] synthetic before/after: %d -> %d",
        synthetic_total_before_qgate,
        synthetic_total_after_qgate,
    )

    LOGGER.info("[8] Stage C protocol-fix auxiliary training...")
    head_stage_c, trainable_names = build_stage_c_head(head=head, scope=args.stage_c_train_scope)
    head_stage_c = head_stage_c.to(device)
    stage_c = train_classifier_stage_protocolfix(
        spec=spec,
        model=model,
        head_stage_c=head_stage_c,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fused_syn_train=fused_syn.float(),
        y_syn_train=y_syn.float(),
        args=args,
        device=device,
        run_paths=run_paths,
        trainable_names=trainable_names,
    )

    LOGGER.info("[9] Collecting validation/test logits for final evaluation...")
    valid_outputs = collect_logits_and_targets(
        spec=spec,
        model=model,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        loader=valid_loader,
        device=device,
        max_batches=args.max_valid_batches,
    )
    test_outputs = collect_logits_and_targets(
        spec=spec,
        model=model,
        head_stage_a=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
        max_batches=args.max_test_batches,
    )

    stage_a_val_auc = evaluate_mode_auc(spec=spec, outputs=valid_outputs, mode="stage_a_only")
    stage_c_val_auc = evaluate_mode_auc(spec=spec, outputs=valid_outputs, mode="stage_c_only")
    stage_a_test_auc = evaluate_mode_auc(spec=spec, outputs=test_outputs, mode="stage_a_only")
    stage_c_test_auc = evaluate_mode_auc(spec=spec, outputs=test_outputs, mode="stage_c_only")

    blend_search = search_blend_beta(
        spec=spec,
        val_outputs=valid_outputs,
        beta_grid=args.blend_beta_grid,
    )
    blend_beta = float(blend_search["beta"])
    blend_val_auc = float(blend_search["val_auc"])
    blend_test_auc = evaluate_mode_auc(
        spec=spec,
        outputs=test_outputs,
        mode="val_tuned_blend",
        blend_beta=blend_beta,
    )

    if args.final_eval_mode == "stage_a_only":
        final_test_auc = stage_a_test_auc
    elif args.final_eval_mode == "stage_c_only":
        final_test_auc = stage_c_test_auc
    else:
        final_test_auc = blend_test_auc

    summary = {
        "dataset": dataset_name,
        "experiment": EXPERIMENT_SUFFIX,
        "task_type": spec.task_type,
        "output_dim": spec.output_dim,
        "final_eval_mode": str(args.final_eval_mode),
        "test_auc": float(final_test_auc),
        "stage_a_test_auc": float(stage_a_test_auc),
        "stage_c_test_auc": float(stage_c_test_auc),
        "blend_test_auc": float(blend_test_auc),
        "blend_beta": float(blend_beta),
        "blend_val_auc": float(blend_val_auc),
        "stage_a_val_auc": float(stage_a_val_auc),
        "stage_c_val_auc": float(stage_c_val_auc),
        "stage_a_best_val_auc": float(stage_a["best_val_auc"]),
        "stage_a_best_epoch": int(stage_a["best_epoch"]),
        "stage_a_best_metric_mode": str(stage_a["best_metric_mode"]),
        "synthetic_total_before_qgate": int(synthetic_total_before_qgate),
        "synthetic_total_after_qgate": int(synthetic_total_after_qgate),
        "stage_c_train_scope": str(args.stage_c_train_scope),
        "stage_a": stage_a,
        "stage_b": {
            "best_epoch": stage_b["best_epoch"],
            "best_val_loss": stage_b["best_val_loss"],
            "best_ckpt_path": stage_b["best_ckpt_path"],
            "history": stage_b["history"],
            "diffusion_num_timesteps": int(args.diffusion_num_timesteps),
        },
        "stage_c": stage_c,
        "qgate": qgate_summary,
        "args": vars(args),
        "run_paths": run_paths,
    }
    save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s muv formal4 fix v2 final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  stage_a best val_auc:      %.4f", stage_a["best_val_auc"])
    LOGGER.info("  stage_c best val_auc:      %.4f", stage_c["best_val_auc"])
    LOGGER.info("  stage_a test_auc:          %.4f", stage_a_test_auc)
    LOGGER.info("  stage_c test_auc:          %.4f", stage_c_test_auc)
    LOGGER.info("  blend_beta:                %.2f", blend_beta)
    LOGGER.info("  blend_test_auc:            %.4f", blend_test_auc)
    LOGGER.info("  final_eval_mode:           %s", args.final_eval_mode)
    LOGGER.info("  test_auc:                  %.4f", final_test_auc)
    LOGGER.info("  summary_path:              %s", run_paths["summary_path"])
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    return summary


def main(dataset_name: str):
    return run_tunev2_latentdiff_muvformal4fix_v2_experiment(dataset_name)
