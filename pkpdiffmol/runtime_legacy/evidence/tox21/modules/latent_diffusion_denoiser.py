# -*- coding: utf-8 -*-
"""
向量版 latent diffusion denoiser。

目标：
1. 输入 z_t、类别条件 c、时间步 t；
2. 输出与 z_t 同维度的 v_hat；
3. 不使用图像卷积，不使用 U-Net；
4. 参考 tab-ddpm 的 MLPDiffusion 思路，但改写成更稳妥的 residual-style MLP。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from latent_diffusion_core import get_timestep_embedding
except ImportError:  # pragma: no cover
    from .latent_diffusion_core import get_timestep_embedding


class ResidualMLPBlock(nn.Module):
    """
    简单的残差 MLP block。

    设计要点：
    1. 条件向量先投影到 hidden_dim；
    2. 每个 block 内部做一次 LayerNorm；
    3. 使用残差连接稳住小规模向量扩散训练。
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


class LatentDiffusionMLPDenoiser(nn.Module):
    """
    最小向量 denoiser：g_theta(z_t, c, t) -> v_hat
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int = 2,
        time_embed_dim: int = 128,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("当前最小实现只支持二分类 num_classes=2。")

        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.time_embed_dim = int(time_embed_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        # 中文注释：类别条件写成 e_c = LayerNorm(Embed(c))
        self.class_embedding = nn.Embedding(self.num_classes, self.cond_dim)
        self.class_layernorm = nn.LayerNorm(self.cond_dim)

        # 中文注释：时间条件写成 Linear(SinusoidalPE(t)) 的小 MLP 版本。
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        # 中文注释：把类别条件与时间条件先融合，再投影到主干隐藏维度。
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 中文注释：主输入 z_t 先投影到 hidden_dim。
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.input_layernorm = nn.LayerNorm(self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim=self.hidden_dim, dropout=dropout)
                for _ in range(self.num_blocks)
            ]
        )

        self.output_layernorm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        生成时间嵌入 e_t。
        """
        t_emb = get_timestep_embedding(
            timesteps=t.view(-1),
            embedding_dim=self.time_embed_dim,
        )
        return self.time_mlp(t_emb)

    def encode_class(self, c: torch.Tensor) -> torch.Tensor:
        """
        生成类别嵌入 e_c。
        """
        c = c.view(-1).long()
        e_c = self.class_embedding(c)
        return self.class_layernorm(e_c)

    def forward(
        self,
        z_t: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入
        ----
        z_t : (B, Dz)
        c   : (B,)
        t   : (B,)

        输出
        ----
        v_hat : (B, Dz)
        """
        if z_t.ndim != 2:
            raise ValueError("z_t 必须是二维张量 (B, Dz)。")
        if z_t.shape[1] != self.latent_dim:
            raise ValueError("z_t 的最后维度与 latent_dim 不一致。")

        e_t = self.encode_time(t)
        e_c = self.encode_class(c)
        cond = self.cond_proj(e_t + e_c)

        h = self.input_proj(z_t.float())
        h = self.input_layernorm(h)

        for block in self.blocks:
            h = block(h, cond)

        h = self.output_layernorm(h)
        h = F.silu(h)
        v_hat = self.output_proj(h)
        return v_hat
