# -*- coding: utf-8 -*-
"""
第一阶段 QGate 共享工具。

说明：
1. 复用旧版二分类 class-wise QGate；
2. 额外补充多任务所需的全局无标签 QGate；
3. 所有统计量都只能由 train split 的真实 fused latent 计算得到。
"""

from typing import Any, Dict, Tuple

import torch

try:
    from latent_qgate import apply_qgate, compute_class_stats_for_qgate, mahalanobis_distance
except ImportError:  # pragma: no cover
    from .latent_qgate import apply_qgate, compute_class_stats_for_qgate, mahalanobis_distance


def compute_global_stats_for_qgate(
    fused_real_train: torch.Tensor,
    quantile: float = 0.95,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    基于真实 train fused latent 计算全局 Mahalanobis 统计量。
    """
    if fused_real_train.ndim != 2:
        raise ValueError("fused_real_train 必须是二维张量 (N, D)。")
    if fused_real_train.shape[0] <= 0:
        raise ValueError("fused_real_train 不能为空。")
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError("quantile 必须位于 (0, 1]。")
    if float(eps) <= 0.0:
        raise ValueError("eps 必须为正数。")

    feature_dim = int(fused_real_train.shape[1])
    work_device = fused_real_train.device
    eye = torch.eye(feature_dim, dtype=torch.float64, device=work_device)
    fused_real_train = fused_real_train.to(dtype=torch.float64, device=work_device)

    mu = fused_real_train.mean(dim=0)
    centered = fused_real_train - mu.unsqueeze(0)
    denom = max(int(fused_real_train.shape[0]) - 1, 1)
    covariance = centered.transpose(0, 1).matmul(centered) / float(denom)
    covariance = covariance + float(eps) * eye

    try:
        inv_cov = torch.linalg.inv(covariance)
    except RuntimeError:
        inv_cov = torch.linalg.pinv(covariance)

    real_distances = mahalanobis_distance(fused_real_train, mu, inv_cov)
    tau = torch.quantile(real_distances, q=float(quantile))

    return {
        "mu": mu.detach(),
        "inv_cov": inv_cov.detach(),
        "tau": tau.detach(),
        "count": torch.tensor(
            int(fused_real_train.shape[0]), dtype=torch.long, device=work_device
        ),
    }


def apply_global_qgate(
    fused_syn: torch.Tensor,
    global_stats: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    使用全局 Mahalanobis 阈值过滤 synthetic fused latent。
    """
    if fused_syn.ndim != 2:
        raise ValueError("fused_syn 必须是二维张量 (N, D)。")

    total_count = int(fused_syn.shape[0])
    if total_count == 0:
        return fused_syn, torch.zeros(0, dtype=torch.bool, device=fused_syn.device), {
            "before_count": 0,
            "after_count": 0,
            "keep_ratio": 0.0,
        }

    distances = mahalanobis_distance(
        fused_syn,
        mu=global_stats["mu"],
        inv_cov=global_stats["inv_cov"],
    )
    tau = global_stats["tau"].to(device=distances.device, dtype=distances.dtype)
    keep_mask = distances <= tau
    fused_kept = fused_syn[keep_mask]
    after_count = int(fused_kept.shape[0])
    keep_ratio = float(after_count) / float(total_count) if total_count > 0 else 0.0

    return fused_kept, keep_mask, {
        "before_count": total_count,
        "after_count": after_count,
        "keep_ratio": keep_ratio,
    }
