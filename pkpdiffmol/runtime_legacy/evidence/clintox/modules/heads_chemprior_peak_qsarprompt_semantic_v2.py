# -*- coding: utf-8 -*-
"""
BBBP semantic v2 并联 head。

设计原则：
1. 不改 backbone，不改旧 peak，不覆盖 semantic v1。
2. 原 r_enc 主路保持不变。
3. 原 12 维 descriptor 数值路保持不变。
4. semantic 分支仍然只吃 Prompt(d)，不吃 backbone hidden states。
5. v2 重点增强训练策略支撑：独立参数分组、零启动、alpha warmup、alpha clamp。
"""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from heads_chemprior_peak import (
        BaselineEquivalentChemPriorHead,
        RDKIT_AVAILABLE,
    )
    from qsar_prompt_semantic_encoder import QSARPromptSemanticEncoder
except ImportError:  # pragma: no cover
    from .heads_chemprior_peak import (
        BaselineEquivalentChemPriorHead,
        RDKIT_AVAILABLE,
    )
    from .qsar_prompt_semantic_encoder import QSARPromptSemanticEncoder


class BaselineEquivalentChemPriorSemanticHeadV2(BaselineEquivalentChemPriorHead):
    """
    在旧 peak descriptor 分支内部并联 semantic 文本分支，并提供更稳健的训练控制接口。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        desc_hidden: int = 64,
        dropout: float = 0.1,
        desc_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        use_desc_layernorm: bool = False,
        use_qsar_prompt_semantic_branch: bool = True,
        prompt_round_digits: int = 2,
        text_encoder_name_or_path: str = "",
        text_pooling: str = "mean",
        text_proj_dim: int = 64,
        text_dropout: float = 0.1,
        freeze_text_encoder: bool = True,
        prompt_max_length: int = 128,
        desc_text_fusion_dropout: float = 0.1,
        fusion_mode: str = "residual",
        fusion_alpha_init: float = 0.0,
        alpha_max_value: float = 0.2,
        clamp_alpha: bool = True,
        use_separate_semantic_param_group: bool = True,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            desc_hidden=desc_hidden,
            dropout=dropout,
            desc_dropout=desc_dropout,
            fusion_dropout=fusion_dropout,
            use_desc_layernorm=use_desc_layernorm,
        )

        self.use_qsar_prompt_semantic_branch = bool(use_qsar_prompt_semantic_branch)
        self.use_separate_semantic_param_group = bool(use_separate_semantic_param_group)
        self.fusion_mode = str(fusion_mode).lower()
        self.alpha_max_value = float(alpha_max_value)
        self.clamp_alpha = bool(clamp_alpha)

        if self.fusion_mode not in {"residual", "gate"}:
            raise ValueError(
                f"fusion_mode 只支持 residual/gate，当前收到: {fusion_mode}"
            )
        if self.alpha_max_value <= 0:
            raise ValueError("alpha_max_value 必须大于 0。")

        self.semantic_text_encoder = QSARPromptSemanticEncoder(
            descriptor_names=self.DESCRIPTOR_NAMES,
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_pooling=text_pooling,
            text_proj_dim=text_proj_dim,
            text_dropout=text_dropout,
            prompt_round_digits=prompt_round_digits,
            freeze_text_encoder=freeze_text_encoder,
            prompt_max_length=prompt_max_length,
        )

        # 中文注释：先将文本语义表示映射到 descriptor 隐空间，再做保守融合。
        self.text_to_desc_proj = nn.Linear(int(text_proj_dim), int(desc_hidden))
        self.desc_text_fusion_dropout = nn.Dropout(p=float(desc_text_fusion_dropout))

        # 中文注释：alpha 默认为 0，实现真正零启动。
        self.fusion_alpha = nn.Parameter(
            torch.tensor(float(fusion_alpha_init), dtype=torch.float32)
        )

        # 中文注释：alpha 运行时放大系数跟随 epoch warmup，可进入 checkpoint 方便精确复现。
        self.register_buffer(
            "alpha_runtime_scale",
            torch.tensor(1.0, dtype=torch.float32),
        )

        if self.fusion_mode == "gate":
            self.desc_text_gate = nn.Sequential(
                nn.Linear(int(desc_hidden) * 2, int(desc_hidden)),
                nn.Sigmoid(),
            )
        else:
            self.desc_text_gate = None

        if self.clamp_alpha:
            self.clamp_alpha_()

    @property
    def text_encoder_module(self) -> QSARPromptSemanticEncoder:
        """
        暴露 semantic encoder，供训练脚本分组参数。
        """
        return self.semantic_text_encoder

    def named_text_encoder_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """
        返回预训练文本编码器本体参数。
        """
        for name, parameter in self.semantic_text_encoder.text_encoder.named_parameters():
            yield f"semantic_text_encoder.text_encoder.{name}", parameter

    def named_semantic_proj_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """
        返回 semantic 可训练投影层与 adapter 参数。
        """
        for name, parameter in self.semantic_text_encoder.text_proj.named_parameters():
            yield f"semantic_text_encoder.text_proj.{name}", parameter

        for name, parameter in self.text_to_desc_proj.named_parameters():
            yield f"text_to_desc_proj.{name}", parameter

        if self.desc_text_gate is not None:
            for name, parameter in self.desc_text_gate.named_parameters():
                yield f"desc_text_gate.{name}", parameter

    def named_alpha_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """
        只返回 alpha 参数。
        """
        yield "fusion_alpha", self.fusion_alpha

    def set_semantic_proj_trainable(self, trainable: bool) -> None:
        """
        控制 semantic 投影层是否参与训练，不影响旧数值 descriptor 主路。
        """
        for _, parameter in self.named_semantic_proj_parameters():
            parameter.requires_grad = bool(trainable)

    def set_alpha_runtime_scale(self, scale: float) -> None:
        """
        设置 alpha 的运行时缩放因子。
        """
        clipped_scale = max(0.0, min(1.0, float(scale)))
        self.alpha_runtime_scale.fill_(clipped_scale)

    def clamp_alpha_(self) -> None:
        """
        将 alpha 原地限制到 [0, alpha_max_value]。
        """
        if not self.clamp_alpha:
            return

        with torch.no_grad():
            self.fusion_alpha.clamp_(min=0.0, max=self.alpha_max_value)

    def get_effective_alpha(self) -> torch.Tensor:
        """
        返回 forward 中真正生效的 alpha。
        """
        alpha = self.fusion_alpha
        if self.clamp_alpha:
            alpha = alpha.clamp(min=0.0, max=self.alpha_max_value)
        return alpha * self.alpha_runtime_scale

    def _fuse_descriptor_repr(
        self,
        r_num: torch.Tensor,
        r_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        descriptor 内部做保守融合。
        """
        text_delta = self.text_to_desc_proj(r_text)
        text_delta = self.desc_text_fusion_dropout(text_delta)
        alpha = self.get_effective_alpha()

        if self.fusion_mode == "residual":
            return r_num + alpha * text_delta

        gate = self.desc_text_gate(torch.cat([r_num, text_delta], dim=-1))
        return r_num + alpha * gate * text_delta

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        保持旧 peak 的主路与数值 descriptor 路不变，只在 descriptor 内部并联 semantic 分支。
        """
        if not self.use_qsar_prompt_semantic_branch:
            return super().forward(x, smiles)

        batch_size = x.shape[0]

        # 中文注释：旧 peak 的主路 r_enc 完全保持不变。
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        r_enc = self.dropout(x)

        # 中文注释：旧 peak 的 12 维 descriptor 数值路完全保持不变。
        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(r_enc.device)
        else:
            desc_raw = torch.zeros(batch_size, self.DESC_DIM, device=r_enc.device)

        desc_vals = desc_raw[:, :self.NUM_DESC]
        is_valid_flag = desc_raw[:, self.NUM_DESC:]
        desc_normalized = (desc_vals - self.desc_mean) / (self.desc_std + 1e-8)
        desc_full = torch.cat([desc_normalized, is_valid_flag], dim=-1)

        r_num = self.desc_mlp(desc_full)
        r_num = self.desc_dropout(r_num)
        r_num = self.desc_layernorm(r_num)

        # 中文注释：semantic 路仍然只吃 descriptor prompt。
        r_text = self.semantic_text_encoder(
            raw_descriptors=desc_vals,
            valid_flags=is_valid_flag.squeeze(-1),
        )

        r_desc_new = self._fuse_descriptor_repr(r_num=r_num, r_text=r_text)

        # 中文注释：最终接口保持 [r_enc ⊕ r_desc_new] -> out_proj。
        fused = torch.cat([r_enc, r_desc_new], dim=-1)
        fused = self.fusion_dropout(fused)
        logits = self.out_proj(fused)
        return logits
