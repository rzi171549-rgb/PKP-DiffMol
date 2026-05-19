# -*- coding: utf-8 -*-
"""
第一阶段 latent diffusion denoiser 补充实现。

说明：
1. 复用原有二分类 denoiser，不改旧文件；
2. 额外提供全局无条件版本，供多任务数据集使用；
3. 第一阶段多任务场景不实现 task-wise condition。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from latent_diffusion_core import get_timestep_embedding
except ImportError:  # pragma: no cover
    from .latent_diffusion_core import get_timestep_embedding


class ResidualMLPBlockStage1(nn.Module):
    """
    与旧版结构保持一致的残差 MLP block。
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cond_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.fc2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        h = x + self.cond_proj(cond)
        h = self.norm(h)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return residual + h


class GlobalLatentDiffusionMLPDenoiser(nn.Module):
    """
    第一阶段多任务保守版的全局无条件 denoiser。

    这里仅使用时间步条件，不引入标签或任务条件。
    """

    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int = 128,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.input_layernorm = nn.LayerNorm(self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlockStage1(hidden_dim=self.hidden_dim, dropout=dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.output_layernorm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        生成时间步嵌入。
        """
        t_emb = get_timestep_embedding(timesteps=t.view(-1), embedding_dim=self.time_embed_dim)
        return self.time_mlp(t_emb)

    def forward(
        self,
        z_t: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        接口保持与旧 diffusion core 一致，参数 c 在此版本中被忽略。
        """
        del c
        if z_t.ndim != 2:
            raise ValueError("z_t 必须是二维张量 (B, Dz)。")
        if z_t.shape[1] != self.latent_dim:
            raise ValueError("z_t 的最后维度与 latent_dim 不一致。")

        cond = self.cond_proj(self.encode_time(t))
        h = self.input_proj(z_t.float())
        h = self.input_layernorm(h)

        for block in self.blocks:
            h = block(h, cond)

        h = self.output_layernorm(h)
        h = F.silu(h)
        return self.output_proj(h)
