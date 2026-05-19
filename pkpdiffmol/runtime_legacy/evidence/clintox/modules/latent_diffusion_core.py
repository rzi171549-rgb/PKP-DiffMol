# -*- coding: utf-8 -*-
"""
连续 fused latent diffusion 核心骨架。

本文件只服务于最小可行版 Exp-1：
1. 只处理连续向量 latent；
2. 只支持 cosine schedule；
3. 只支持 velocity parameterization（v-pred）；
4. 只接收最终 fused representation z0；
5. 不包含 multinomial / categorical / tabular 逻辑。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    复用 diffusers 风格的 sinusoidal timestep embedding。
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps 必须是一维张量。")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim 必须大于 0。")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return torch.zeros(
            (timesteps.shape[0], embedding_dim),
            device=timesteps.device,
            dtype=torch.float32,
        )

    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / max(1.0, float(half_dim) - float(downscale_freq_shift))

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """
    生成 cosine beta schedule。
    """
    if num_timesteps <= 0:
        raise ValueError("num_timesteps 必须大于 0。")

    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    t = steps / float(num_timesteps)
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(min=1e-8, max=max_beta)
    return betas.float()


def extract_into_tensor(
    buffer: torch.Tensor,
    timesteps: torch.Tensor,
    x_shape: torch.Size,
) -> torch.Tensor:
    """
    从 1D buffer 中按时间步取值，并广播到目标张量形状。
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps 必须是一维张量。")
    out = buffer.gather(0, timesteps.long())
    while out.ndim < len(x_shape):
        out = out.unsqueeze(-1)
    return out


class ContinuousLatentDiffusion(nn.Module):
    """
    面向连续 fused latent 的最小高斯扩散器。

    约束：
    1. 只支持 prediction_type='v'
    2. 只支持 beta_schedule='cosine'
    3. denoiser 的接口固定为 g_theta(z_t, c, t) -> v_hat
    """

    def __init__(
        self,
        denoiser: nn.Module,
        latent_dim: int,
        num_timesteps: int = 50,
        beta_schedule: str = "cosine",
        prediction_type: str = "v",
    ):
        super().__init__()
        if prediction_type != "v":
            raise ValueError("当前最小实现只支持 prediction_type='v'。")
        if beta_schedule != "cosine":
            raise ValueError("当前最小实现只支持 beta_schedule='cosine'。")
        if latent_dim <= 0:
            raise ValueError("latent_dim 必须大于 0。")

        self.denoiser = denoiser
        self.latent_dim = int(latent_dim)
        self.num_timesteps = int(num_timesteps)
        self.beta_schedule = str(beta_schedule)
        self.prediction_type = str(prediction_type)

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [
                torch.ones(1, dtype=alphas_cumprod.dtype),
                alphas_cumprod[:-1],
            ],
            dim=0,
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = posterior_variance.clamp(min=1e-20)

        if posterior_variance.shape[0] > 1:
            posterior_log_variance_clipped = torch.log(
                torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0)
            )
        else:
            posterior_log_variance_clipped = torch.log(posterior_variance)

        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod,
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_log_variance_clipped,
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        训练时均匀采样时间步。
        """
        return torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向加噪：z_t = alpha_t * z0 + sigma_t * eps
        """
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, z0.shape)
        sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return alpha_t * z0 + sigma_t * noise

    def get_velocity_target(
        self,
        z0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造 velocity 监督目标：
        v = alpha_t * eps - sigma_t * z0
        """
        alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, z0.shape)
        sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return alpha_t * noise - sigma_t * z0

    def predict_x0_from_v(
        self,
        z_t: torch.Tensor,
        v_hat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 v 预测重建 x0：
        x0 = alpha_t * z_t - sigma_t * v
        """
        alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, z_t.shape)
        sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        return alpha_t * z_t - sigma_t * v_hat

    def predict_eps_from_v(
        self,
        z_t: torch.Tensor,
        v_hat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 v 反推出噪声：
        eps = sigma_t * z_t + alpha_t * v
        """
        alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, z_t.shape)
        sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        return sigma_t * z_t + alpha_t * v_hat

    def q_posterior_mean_variance(
        self,
        x0_hat: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        q(z_{t-1} | z_t, z0_hat) 的后验均值与方差。
        """
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, z_t.shape) * x0_hat
            + extract_into_tensor(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, z_t.shape)
        log_variance = extract_into_tensor(
            self.posterior_log_variance_clipped,
            t,
            z_t.shape,
        )
        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
        }

    def p_mean_variance(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        反向一步的参数化结果。
        """
        v_hat = self.denoiser(z_t, c, t)
        x0_hat = self.predict_x0_from_v(z_t, v_hat, t)
        eps_hat = self.predict_eps_from_v(z_t, v_hat, t)
        posterior = self.q_posterior_mean_variance(x0_hat=x0_hat, z_t=z_t, t=t)
        posterior["pred_x0"] = x0_hat
        posterior["pred_eps"] = eps_hat
        posterior["pred_v"] = v_hat
        return posterior

    @torch.no_grad()
    def p_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        采样 z_{t-1}。
        """
        out = self.p_mean_variance(z_t=z_t, t=t, c=c)
        noise = torch.randn_like(z_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (z_t.ndim - 1)))
        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    @torch.no_grad()
    def sample(
        self,
        class_labels: torch.Tensor,
        device: torch.device,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        按给定类别条件生成 synthetic fused latents。
        """
        if class_labels.ndim != 1:
            raise ValueError("class_labels 必须是一维张量。")
        batch_size = int(class_labels.shape[0])
        if batch_size == 0:
            return torch.zeros(
                (0, self.latent_dim),
                device=device,
                dtype=torch.float32,
            )

        class_labels = class_labels.to(device=device, dtype=torch.long)
        if noise is None:
            z_t = torch.randn(batch_size, self.latent_dim, device=device)
        else:
            z_t = noise.to(device=device, dtype=torch.float32)
            if z_t.shape != (batch_size, self.latent_dim):
                raise ValueError("noise 形状与 (B, latent_dim) 不一致。")

        for step in reversed(range(self.num_timesteps)):
            t = torch.full(
                (batch_size,),
                fill_value=step,
                device=device,
                dtype=torch.long,
            )
            z_t = self.p_sample(z_t=z_t, t=t, c=class_labels)

        return z_t

    def training_loss(
        self,
        z0: torch.Tensor,
        c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的最小 loss 路径。
        """
        if z0.ndim != 2:
            raise ValueError("z0 必须是二维张量 (B, Dz)。")
        if z0.shape[1] != self.latent_dim:
            raise ValueError("z0 的最后维度与 latent_dim 不一致。")

        c = c.to(device=z0.device, dtype=torch.long).view(-1)
        t = self.sample_timesteps(batch_size=z0.shape[0], device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0=z0, t=t, noise=noise)
        v_target = self.get_velocity_target(z0=z0, noise=noise, t=t)
        v_hat = self.denoiser(z_t, c, t)
        x0_hat = self.predict_x0_from_v(z_t=z_t, v_hat=v_hat, t=t)

        loss = F.mse_loss(v_hat, v_target, reduction="mean")
        return {
            "loss": loss,
            "mse": loss.detach(),
            "t": t.detach(),
            "z_t": z_t.detach(),
            "v_target": v_target.detach(),
            "v_hat": v_hat,
            "pred_x0": x0_hat.detach(),
        }

    def forward(
        self,
        z0: torch.Tensor,
        c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        让扩散器本身可直接作为 nn.Module 调用。
        """
        return self.training_loss(z0=z0, c=c)
