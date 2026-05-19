# -*- coding: utf-8 -*-
"""
BACE / SIDER 第4组优化版通用入口。

设计原则：
1. 不修改现有第4组主线 general、head、stage1 文件。
2. 仅为 BACE / SIDER 提供更稳的默认超参与更合理的最终测试头选择。
3. Stage A / B / C 仍然复用现有三阶段框架，只在必要处补充可调 diffusion 步数与早停。
"""

import logging
import os
import sys
from typing import Any, Dict, List, Tuple

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

from chemprior_peak_general import (  # noqa: E402
    DATASET_SPECS,
    DatasetSpec,
    build_dataloaders,
    configure_logging,
    load_backbone,
    set_random_seed,
)
from chemprior_peak_latentdiff_stage1 import (  # noqa: E402
    STAGE_B_NUM_TIMESTEPS,
    apply_bace_qgate_if_needed,
    apply_multitask_qgate_if_needed,
    build_binary_diffusion_dataloaders,
    build_binary_train_class_counts,
    build_multitask_diffusion_dataloaders,
    evaluate_full_model,
    extract_fused_bank,
    generate_binary_synthetic_bank,
    generate_multitask_synthetic_bank,
    set_requires_grad,
    train_classifier_stage,
    train_diffusion_stage,
)
from chemprior_parallel_latentdiff_general import (  # noqa: E402
    build_head as build_parallel_latentdiff_head,
    build_parser as build_base_parser,
    build_stage_c_head,
    load_stage_a_best,
    run_epoch_tunev2_latentdiff,
    save_json,
)
from latent_diffusion_core import ContinuousLatentDiffusion  # noqa: E402
from latent_diffusion_denoiser import LatentDiffusionMLPDenoiser  # noqa: E402
from latent_diffusion_denoiser_stage1 import GlobalLatentDiffusionMLPDenoiser  # noqa: E402
from train_bbbp_baselineeq_chemprior_peak_qsarprompt_semantic_v2 import (  # noqa: E402
    build_optimizer_param_groups,
    compute_best_score,
    post_optimizer_step,
    prepare_epoch_training_policy,
    summarize_optimizer_groups,
)
from train_bbbp_chemprior_parallel import build_scheduler, should_replace_best  # noqa: E402


LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = {"bace", "sider"}
EXPERIMENT_SUFFIX = "chemprior_parallel_latentdiff_optimized"

OPTIMIZED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "bace": {
        "epochs": 18,
        "batch_size": 32,
        "lr": 8e-5,
        "backbone_lr": 6e-5,
        "head_lr": 1e-4,
        "semantic_proj_lr": 4e-5,
        "alpha_lr": 1e-5,
        "warmup_ratio": 0.08,
        "weight_decay": 0.01,
        "head_weight_decay": 0.02,
        "pooler_dropout": 0.30,
        "desc_dropout": 0.10,
        "fusion_dropout": 0.15,
        "desc_text_fusion_dropout": 0.10,
        "grad_clip": 0.8,
        "best_metric": "val_auc",
        "min_best_epoch": 3,
        "freeze_semantic_proj_epochs": 1,
        "alpha_warmup_epochs": 4,
        "diffusion_epochs": 10,
        "diffusion_batch_size": 64,
        "diffusion_lr": 6e-4,
        "diffusion_weight_decay": 1e-4,
        "diffusion_num_timesteps": 30,
        "synthetic_rho": 0.03,
        "use_qgate": False,
        "stage_c_epochs": 12,
        "stage_c_batch_size": 64,
        "stage_c_lr": 1.5e-4,
        "stage_c_weight_decay": 0.01,
        "stage_a_patience": 6,
    },
    "sider": {
        "epochs": 40,
        "batch_size": 8,
        "lr": 8e-5,
        "backbone_lr": 6e-5,
        "head_lr": 1e-4,
        "semantic_proj_lr": 3e-5,
        "alpha_lr": 2e-5,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "pooler_dropout": 0.20,
        "desc_dropout": 0.20,
        "fusion_dropout": 0.20,
        "desc_text_fusion_dropout": 0.20,
        "grad_clip": 0.8,
        "best_metric": "val_auc",
        "min_best_epoch": 6,
        "freeze_semantic_proj_epochs": 2,
        "alpha_warmup_epochs": 6,
        "diffusion_epochs": 18,
        "diffusion_batch_size": 128,
        "diffusion_lr": 8e-4,
        "diffusion_weight_decay": 1e-4,
        "diffusion_num_timesteps": 50,
        "synthetic_rho": 0.08,
        "use_qgate": True,
        "stage_c_epochs": 15,
        "stage_c_batch_size": 128,
        "stage_c_lr": 2e-4,
        "stage_c_weight_decay": 0.01,
        "stage_a_patience": 10,
    },
}


def build_parser(dataset_name: str):
    """
    在现有第4组 parser 上叠加 BACE / SIDER 的优化版默认参数。
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"优化版目前只支持 {sorted(SUPPORTED_DATASETS)}，收到 {dataset_name}")

    parser = build_base_parser(dataset_name)
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} chemprior parallel latentdiff optimized training"
    parser.add_argument("--diffusion-num-timesteps", type=int, default=STAGE_B_NUM_TIMESTEPS)
    parser.add_argument("--stage-a-patience", type=int, default=0)

    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        **OPTIMIZED_DEFAULTS[dataset_name],
    )
    return parser


def resolve_run_paths(args, dataset_name: str) -> Dict[str, str]:
    """
    为优化版实验生成独立输出目录。
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


def build_diffusion_model_with_steps(
    args,
    latent_dim: int,
    is_binary_conditioned: bool,
    device: torch.device,
):
    """
    复用现有 denoiser 结构，只把 diffusion 步数改成可配。
    """
    if int(args.diffusion_num_timesteps) <= 0:
        raise ValueError("diffusion_num_timesteps 必须大于 0")

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

    return ContinuousLatentDiffusion(
        denoiser=denoiser,
        latent_dim=latent_dim,
        num_timesteps=int(args.diffusion_num_timesteps),
        beta_schedule=args.diffusion_beta_schedule,
        prediction_type=args.diffusion_prediction_type,
    ).to(device)


def run_stage_a_optimized(
    spec: DatasetSpec,
    model: nn.Module,
    head: nn.Module,
    train_loader,
    valid_loader,
    args,
    device: torch.device,
    run_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Stage A 训练主模型，并支持基于验证集的简单早停。
    """
    if args.freeze_backbone:
        set_requires_grad(model, False)

    param_groups = build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("Stage A 未能构建出任何可训练参数组")

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
    early_stopped = False
    stopped_epoch = int(args.epochs)

    LOGGER.info("[Stage A] 开始训练优化版主链。")
    LOGGER.info("[Stage A] total steps: %s", num_training_steps)
    LOGGER.info("[Stage A] warmup steps: %s", num_warmup_steps)
    LOGGER.info("[Stage A] patience: %s", args.stage_a_patience)
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

        is_new_best = False
        if epoch >= effective_min_best_epoch and should_replace_best(
            score_info=score_info,
            val_auc=val_auc,
            val_loss=val_loss,
            best_state=best_state,
            args=args,
        ):
            is_new_best = True
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

        if (
            not is_new_best
            and args.stage_a_patience > 0
            and best_state is not None
            and epoch >= effective_min_best_epoch
            and (epoch - int(best_state["best_epoch"])) >= int(args.stage_a_patience)
        ):
            early_stopped = True
            stopped_epoch = int(epoch)
            LOGGER.info(
                "[Stage A] 触发早停：当前 epoch=%d，best_epoch=%d，patience=%d",
                epoch,
                int(best_state["best_epoch"]),
                int(args.stage_a_patience),
            )
            break

    if best_state is None:
        raise RuntimeError("Stage A 训练结束但没有生成 best checkpoint")

    load_stage_a_best(model=model, head=head, path=run_paths["stage_a_ckpt_path"], device=device)
    return {
        "best_epoch": int(best_state["best_epoch"]),
        "best_val_auc": float(best_state["best_val_auc"]),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_hybrid_score": float(best_state["best_hybrid_score"]),
        "best_ckpt_path": run_paths["stage_a_ckpt_path"],
        "optimizer_groups": optimizer_group_summary,
        "history": history,
        "early_stopped": bool(early_stopped),
        "stopped_epoch": int(stopped_epoch),
    }


def log_run_header(args, spec: DatasetSpec, device: torch.device, run_paths: Dict[str, str]) -> None:
    """
    记录优化版实验头信息。
    """
    LOGGER.info("=" * 72)
    LOGGER.info("%s chemprior parallel latentdiff optimized", spec.name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  python:                    %s", sys.executable)
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
    LOGGER.info("  lr:                        %s", args.lr)
    LOGGER.info("  backbone_lr:               %s", args.backbone_lr)
    LOGGER.info("  head_lr:                   %s", args.head_lr)
    LOGGER.info("  warmup_ratio:              %s", args.warmup_ratio)
    LOGGER.info("  pooler_dropout:            %s", args.pooler_dropout)
    LOGGER.info("  desc_dropout:              %s", args.desc_dropout)
    LOGGER.info("  fusion_dropout:            %s", args.fusion_dropout)
    LOGGER.info("  synthetic_rho:             %s", args.synthetic_rho)
    LOGGER.info("  use_qgate:                 %s", args.use_qgate)
    LOGGER.info("  diffusion_num_timesteps:   %s", args.diffusion_num_timesteps)
    LOGGER.info("  stage_a_patience:          %s", args.stage_a_patience)
    LOGGER.info("=" * 72)


def run_optimized_experiment(dataset_name: str):
    """
    运行单个数据集的优化版三阶段实验。
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"优化版目前只支持 {sorted(SUPPORTED_DATASETS)}，收到 {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    if int(args.diffusion_epochs) <= 0:
        raise ValueError("diffusion_epochs 必须大于 0")
    if int(args.stage_c_epochs) <= 0:
        raise ValueError("stage_c_epochs 必须大于 0")

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

    LOGGER.info("[2] Building optimized tunev2 latentdiff head...")
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
        diffusion = build_diffusion_model_with_steps(
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

    LOGGER.info("[9] Evaluating full inference chain on test split...")
    test_loss, test_auc = evaluate_full_model(
        spec=spec,
        model=model,
        head_for_feature=head,
        head_stage_c=head_stage_c,
        loader=test_loader,
        device=device,
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
        "qgate": qgate_summary,
        "synthetic_total_after_qgate": int(fused_syn.shape[0]),
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "args": vars(args),
        "run_paths": run_paths,
    }
    save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s optimized latentdiff final results", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  best stage A val_auc: %.4f", stage_a["best_val_auc"])
    LOGGER.info("  best stage C val_auc: %.4f", stage_c["best_val_auc"])
    LOGGER.info("  test protocol:        fixed Stage C")
    LOGGER.info("  test_auc:             %.4f", test_auc)
    LOGGER.info("  summary_path:         %s", run_paths["summary_path"])
    LOGGER.info("=" * 72)

    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    return summary


def main(dataset_name: str):
    """
    供 BACE / SIDER 优化版 wrapper 调用的入口。
    """
    return run_optimized_experiment(dataset_name)
