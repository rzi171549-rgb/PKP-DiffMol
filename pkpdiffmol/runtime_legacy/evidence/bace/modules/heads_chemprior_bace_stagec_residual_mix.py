# -*- coding: utf-8 -*-
"""
BACE Stage C residual mix 方法版专用头。

设计原则：
1. 不改旧版 head 与旧版 Stage C；
2. 仅在 fused latent 之后增加一个轻量 residual adapter；
3. 保持分类头初始化尽量贴近旧版，仅让 adapter 最后一层零初始化，
   使初始 z_refined 约等于原始 z。
"""

import copy

import torch
import torch.nn as nn


class ResidualLatentAdapter(nn.Module):
    """
    轻量 residual adapter。
    结构：
    LayerNorm -> Linear -> GELU -> Dropout -> Linear
    最后一层做零初始化，保证初始时近似恒等映射。
    """

    def __init__(self, latent_dim: int, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        # 中文注释：最后一层零初始化，使初始 residual 近似为 0，便于与旧版稳定对照。
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(fused)
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden


class BaceStageCResidualMixHead(nn.Module):
    """
    BACE Stage C 新头：
    1. 复制旧版 out_proj 作为分类头；
    2. 在 fused latent 上增加 residual adapter；
    3. 推理路径为 z -> z + adapter(z) -> out_proj。
    """

    def __init__(self, base_head: nn.Module, adapter_dropout: float = 0.10):
        super().__init__()
        latent_dim = int(base_head.final_fused_dim)
        self.final_fused_dim = latent_dim
        self.classifier_out_dim = int(base_head.out_proj.out_features)

        # 中文注释：复用旧版分类头权重初始化，保证新旧版本可对照。
        self.out_proj = copy.deepcopy(base_head.out_proj)
        self.adapter = ResidualLatentAdapter(
            latent_dim=latent_dim,
            dropout=float(adapter_dropout),
        )

    def refine_fused(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 2:
            raise ValueError("fused 必须是二维张量 (B, Dz)。")
        if fused.shape[1] != self.final_fused_dim:
            raise ValueError(
                f"fused 维度不匹配：期望 {self.final_fused_dim}，实际 {fused.shape[1]}"
            )
        return fused + self.adapter(fused)

    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        refined = self.refine_fused(fused)
        return self.out_proj(refined)

