from typing import Any, Dict, Tuple

import torch


def mahalanobis_distance(
    x: torch.Tensor,
    mu: torch.Tensor,
    inv_cov: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mahalanobis distance for a batch of vectors.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (N, D).")
    if mu.ndim != 1:
        raise ValueError("mu must have shape (D,).")
    if inv_cov.ndim != 2:
        raise ValueError("inv_cov must have shape (D, D).")
    if x.shape[1] != mu.shape[0]:
        raise ValueError("x and mu dimensions do not match.")
    if inv_cov.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError("inv_cov shape does not match feature dimension.")

    work_dtype = mu.dtype
    work_device = x.device
    delta = x.to(device=work_device, dtype=work_dtype) - mu.to(
        device=work_device, dtype=work_dtype
    ).unsqueeze(0)
    inv_cov = inv_cov.to(device=work_device, dtype=work_dtype)

    squared = torch.einsum("nd,dd,nd->n", delta, inv_cov, delta)
    squared = squared.clamp(min=0.0)
    return torch.sqrt(squared)


def compute_class_stats_for_qgate(
    fused_real_train: torch.Tensor,
    y_real_train: torch.Tensor,
    num_classes: int = 2,
    quantile: float = 0.95,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Compute per-class mean, inverse covariance, and distance threshold tau
    using only the real train fused bank.
    """
    if fused_real_train.ndim != 2:
        raise ValueError("fused_real_train must have shape (N, D).")
    if y_real_train.ndim != 1:
        y_real_train = y_real_train.view(-1)
    if fused_real_train.shape[0] != y_real_train.shape[0]:
        raise ValueError("fused_real_train and y_real_train sizes do not match.")
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError("quantile must be in (0, 1].")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive.")

    stats: Dict[int, Dict[str, torch.Tensor]] = {}
    feature_dim = int(fused_real_train.shape[1])
    work_device = fused_real_train.device
    eye = torch.eye(feature_dim, dtype=torch.float64, device=work_device)

    y_real_train = y_real_train.long().to(work_device)
    fused_real_train = fused_real_train.to(dtype=torch.float64, device=work_device)

    for class_id in range(int(num_classes)):
        class_mask = y_real_train == class_id
        class_latents = fused_real_train[class_mask]
        class_count = int(class_latents.shape[0])
        if class_count <= 0:
            continue

        mu = class_latents.mean(dim=0)
        centered = class_latents - mu.unsqueeze(0)
        denom = max(class_count - 1, 1)
        covariance = centered.transpose(0, 1).matmul(centered) / float(denom)
        covariance = covariance + float(eps) * eye

        try:
            inv_cov = torch.linalg.inv(covariance)
        except RuntimeError:
            inv_cov = torch.linalg.pinv(covariance)

        real_distances = mahalanobis_distance(class_latents, mu, inv_cov)
        tau = torch.quantile(real_distances, q=float(quantile))

        stats[class_id] = {
            "mu": mu.detach(),
            "inv_cov": inv_cov.detach(),
            "tau": tau.detach(),
            "count": torch.tensor(class_count, dtype=torch.long, device=work_device),
        }

    return stats


def apply_qgate(
    fused_syn: torch.Tensor,
    y_syn: torch.Tensor,
    class_stats: Dict[int, Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Filter synthetic fused latents by per-class Mahalanobis threshold.
    """
    if fused_syn.ndim != 2:
        raise ValueError("fused_syn must have shape (N, D).")
    if y_syn.ndim != 1:
        y_syn = y_syn.view(-1)
    if fused_syn.shape[0] != y_syn.shape[0]:
        raise ValueError("fused_syn and y_syn sizes do not match.")

    total_count = int(fused_syn.shape[0])
    y_syn = y_syn.long()

    class_ids = sorted(
        set(int(class_id) for class_id in class_stats.keys())
        | set(int(class_id) for class_id in y_syn.tolist())
    )
    per_class_before = {class_id: 0 for class_id in class_ids}
    per_class_after = {class_id: 0 for class_id in class_ids}

    if total_count == 0:
        return fused_syn, y_syn, {
            "before_count": 0,
            "after_count": 0,
            "keep_ratio": 0.0,
            "per_class_before": per_class_before,
            "per_class_after": per_class_after,
        }

    keep_mask = torch.zeros(total_count, dtype=torch.bool, device=fused_syn.device)

    for class_id in class_ids:
        class_mask = y_syn == class_id
        before_count = int(class_mask.sum().item())
        per_class_before[class_id] = before_count
        if before_count <= 0:
            continue

        class_latents = fused_syn[class_mask]
        stats = class_stats.get(class_id)
        if stats is None:
            continue

        distances = mahalanobis_distance(
            class_latents,
            mu=stats["mu"],
            inv_cov=stats["inv_cov"],
        )
        tau = stats["tau"].to(device=distances.device, dtype=distances.dtype)
        class_keep = distances <= tau
        per_class_after[class_id] = int(class_keep.sum().item())
        keep_mask[class_mask] = class_keep

    fused_kept = fused_syn[keep_mask]
    y_kept = y_syn[keep_mask]
    after_count = int(fused_kept.shape[0])
    keep_ratio = float(after_count) / float(total_count) if total_count > 0 else 0.0

    return fused_kept, y_kept, {
        "before_count": total_count,
        "after_count": after_count,
        "keep_ratio": keep_ratio,
        "per_class_before": per_class_before,
        "per_class_after": per_class_after,
    }
