# -*- coding: utf-8 -*-
"""
BACE-specific tunev2 protocol fix for Group 3.

This module intentionally reuses the formal tunev2 implementation and only
changes the default training envelope for BACE:
1. best_metric -> val_auc
2. scheduler   -> linear
3. pooler_dropout -> 0.1

Head structure, optimizer grouping, and model family stay unchanged.
"""

import logging
import os
from typing import Any, Dict, List

import torch
from torch.optim import AdamW

import chemprior_peak_tunev2_general as base


LOGGER = logging.getLogger(__name__)
EXPERIMENT_SUFFIX = "baselineeq_chemprior_peak_tunev2_bacefix"


def build_parser(dataset_name: str):
    """
    Start from the formal Group 3 parser and override only the BACE defaults
    required for the first-stage protocol fix.
    """
    parser = base.build_parser(dataset_name)
    tasks_dir = os.path.dirname(os.path.dirname(__file__))
    output_ckpt_dir = os.path.join(
        tasks_dir, dataset_name, "outputs", "checkpoints", EXPERIMENT_SUFFIX
    )
    output_log_dir = os.path.join(tasks_dir, dataset_name, "outputs", "logs", EXPERIMENT_SUFFIX)

    parser.description = f"{dataset_name} baselineeq chemprior peak tunev2 bacefix training"
    parser.set_defaults(
        output_dir=output_ckpt_dir,
        log_file=os.path.join(output_log_dir, f"{dataset_name}_{EXPERIMENT_SUFFIX}_seed0.log"),
        best_metric="val_auc",
        scheduler="linear",
        pooler_dropout=0.1,
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
        "best_ckpt_path": os.path.join(run_dir, f"{run_name}_best.pt"),
        "summary_path": os.path.join(run_dir, "summary.json"),
    }


def run_tunev2_experiment(dataset_name: str):
    """
    Reuse the formal Group 3 training skeleton with BACE-specific protocol
    defaults and isolated output naming.
    """
    spec = base.DATASET_SPECS[dataset_name]
    parser = build_parser(dataset_name)
    args = parser.parse_args()

    run_paths = resolve_run_paths(args, dataset_name)
    os.makedirs(run_paths["root_output_dir"], exist_ok=True)
    os.makedirs(run_paths["run_dir"], exist_ok=True)
    base.configure_logging(args.log_file)
    base.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("=" * 72)
    LOGGER.info("%s chemprior peak tunev2 bacefix", dataset_name.upper())
    LOGGER.info("=" * 72)
    LOGGER.info("  task_type:                 %s", spec.task_type)
    LOGGER.info("  output_dim:                %s", spec.output_dim)
    LOGGER.info("  run_dir:                   %s", run_paths["run_dir"])
    LOGGER.info("  best_ckpt:                 %s", run_paths["best_ckpt_path"])
    LOGGER.info("  summary_path:              %s", run_paths["summary_path"])

    LOGGER.info("[1] Loading backbone...")
    model, dictionary, embed_dim = base.load_backbone(args.checkpoint, args.dict)
    model = model.to(device)

    LOGGER.info("[2] Building tunev2 bacefix head...")
    head = base.build_head(args=args, spec=spec, embed_dim=embed_dim, device=device)
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
    ) = base.build_dataloaders(args, dictionary, spec)

    LOGGER.info("[4] Fitting descriptor normalizer on train split...")
    head.fit_normalizer(train_dataset.collect_smiles(), device)

    LOGGER.info("[5] Building optimizer and scheduler...")
    param_groups = base.build_optimizer_param_groups(model, head, args)
    if not param_groups:
        raise RuntimeError("Failed to build any trainable parameter groups.")
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    optimizer_group_summary = base.summarize_optimizer_groups(optimizer)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = base.build_scheduler(
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
        epoch_policy = base.prepare_epoch_training_policy(head, epoch, args)
        train_loss, train_auc = base.run_epoch_tunev2(
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
        val_loss, val_auc = base.run_epoch_tunev2(
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
        score_info = base.compute_best_score(args, val_auc=val_auc, val_loss=val_loss)
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

        if epoch >= effective_min_best_epoch and base.should_replace_best(
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
        raise RuntimeError("Training finished without producing a best checkpoint.")

    checkpoint = torch.load(run_paths["best_ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    base.post_optimizer_step(head)

    LOGGER.info("[7] Final test evaluation...")
    test_loss, test_auc = base.run_epoch_tunev2(
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
    base.save_json(summary, run_paths["summary_path"])

    LOGGER.info("=" * 72)
    LOGGER.info("%s tunev2 bacefix final results", dataset_name.upper())
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
    return run_tunev2_experiment(dataset_name)
