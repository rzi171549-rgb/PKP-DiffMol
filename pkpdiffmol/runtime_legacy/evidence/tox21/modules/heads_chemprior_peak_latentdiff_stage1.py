# -*- coding: utf-8 -*-
"""
第一阶段 latent diffusion 并行版 ChemPrior 头。

设计目标：
1. 不修改旧的 baseline 头实现；
2. 复用 BaselineEquivalentChemPriorHead 的参数结构；
3. 显式暴露 fused latent，供 Stage B/Stage C 使用；
4. 保持默认前向接口仍然是 hidden_states -> logits。
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from heads_chemprior_peak import BaselineEquivalentChemPriorHead, RDKIT_AVAILABLE
except ImportError:  # pragma: no cover
    from .heads_chemprior_peak import BaselineEquivalentChemPriorHead, RDKIT_AVAILABLE


class BaselineEquivalentChemPriorLatentDiffHead(BaselineEquivalentChemPriorHead):
    """
    在不改动旧头文件的前提下，补充 fused latent 提取接口。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_fused_dim = int(self.out_proj.in_features)
        self.classifier_out_dim = int(self.out_proj.out_features)

    def _encode_fused(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        复用 baseline-equivalent 路径，在 out_proj 之前暴露 fused 表示。
        """
        batch_size = x.shape[0]

        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x_proj = self.dropout(x)

        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(x_proj.device)
        else:
            desc_raw = torch.zeros(batch_size, self.DESC_DIM, device=x_proj.device)

        desc_vals = desc_raw[:, : self.NUM_DESC]
        is_valid_flag = desc_raw[:, self.NUM_DESC :]
        desc_normalized = (desc_vals - self.desc_mean) / (self.desc_std + 1e-8)
        desc_full = torch.cat([desc_normalized, is_valid_flag], dim=-1)

        desc_repr = self.desc_mlp(desc_full)
        desc_repr = self.desc_dropout(desc_repr)
        desc_repr = self.desc_layernorm(desc_repr)

        fused = torch.cat([x_proj, desc_repr], dim=-1)
        fused = self.fusion_dropout(fused)

        return {
            "x_proj": x_proj,
            "desc_repr": desc_repr,
            "fused": fused,
        }

    def extract_fused_latent(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        提取 out_proj 之前的 fused latent。
        """
        fused = self._encode_fused(x=x, smiles=smiles)["fused"]
        return fused.detach() if detach else fused

    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """
        直接在 fused latent 上执行最终分类层。
        """
        if fused.ndim != 2:
            raise ValueError("fused 必须是二维张量 (B, D)。")
        if fused.shape[1] != self.final_fused_dim:
            raise ValueError(
                f"fused 维度不匹配：期望 {self.final_fused_dim}，实际 {fused.shape[1]}"
            )
        return self.out_proj(fused)

    def forward_with_aux(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
        detach_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        同时返回 logits 与中间 fused latent。
        """
        outputs = self._encode_fused(x=x, smiles=smiles)
        fused = outputs["fused"]
        outputs["fused"] = fused.detach() if detach_latent else fused
        outputs["logits"] = self.forward_from_fused(fused)
        return outputs

    def named_classifier_parameters(self) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        """
        只暴露最终分类层参数，供 Stage C 单独训练。
        """
        for name, parameter in self.out_proj.named_parameters():
            yield f"out_proj.{name}", parameter

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        默认行为仍然保持为 hidden_states -> logits。
        """
        outputs = self.forward_with_aux(x=x, smiles=smiles, detach_latent=False)
        return outputs["logits"]
