# -*- coding: utf-8 -*-
"""
BBBP pool_pooler_base tunev2 头。

设计原则：
1. 不改 backbone，不改原有 semantic v2 头。
2. 只补三处轻量归一化，目标是降低高 val / 低 test 的波动。
3. 维度全部从父类真实模块结构推断，不假设父类有 self.in_dim / self.desc_hidden。
"""

from typing import List, Optional

import torch
import torch.nn as nn

try:
    from heads_chemprior_peak import RDKIT_AVAILABLE
    from heads_chemprior_peak_qsarprompt_semantic_v2 import (
        BaselineEquivalentChemPriorSemanticHeadV2,
    )
except ImportError:  # pragma: no cover
    from .heads_chemprior_peak import RDKIT_AVAILABLE
    from .heads_chemprior_peak_qsarprompt_semantic_v2 import (
        BaselineEquivalentChemPriorSemanticHeadV2,
    )


class PoolPoolerBaseTuneV2Head(BaselineEquivalentChemPriorSemanticHeadV2):
    """
    在 semantic v2 头上补三处轻量 LayerNorm：
    1. pooler 主路输出后做一次归一化；
    2. descriptor + semantic 融合后再做一次归一化；
    3. 最终 concat 后、out_proj 前再做一次归一化。

    这样做的目的不是换结构，而是压低不同 seed 下表示尺度的漂移。
    """

    def __init__(
        self,
        *args,
        use_pooler_layernorm: bool = True,
        use_post_fusion_desc_layernorm: bool = True,
        use_fused_layernorm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_pooler_layernorm = bool(use_pooler_layernorm)
        self.use_post_fusion_desc_layernorm = bool(use_post_fusion_desc_layernorm)
        self.use_fused_layernorm = bool(use_fused_layernorm)

        # 中文注释：三处归一化维度从父类已经构建好的真实模块推断。
        (
            self.pooler_repr_dim,
            self.desc_fusion_dim,
            self.final_fused_dim,
        ) = self._infer_norm_dims()

        self.pooler_layernorm = (
            nn.LayerNorm(self.pooler_repr_dim)
            if self.use_pooler_layernorm
            else nn.Identity()
        )
        self.post_fusion_desc_layernorm = (
            nn.LayerNorm(self.desc_fusion_dim)
            if self.use_post_fusion_desc_layernorm
            else nn.Identity()
        )
        self.fused_layernorm = (
            nn.LayerNorm(self.final_fused_dim)
            if self.use_fused_layernorm
            else nn.Identity()
        )

    def _infer_norm_dims(self):
        """
        从父类真实存在的模块结构里推断三处 LayerNorm 维度。

        维度来源：
        1. pooler 主路维度：self.dense.out_features
        2. descriptor / semantic 融合后维度：self.text_to_desc_proj.out_features
        3. final fused 维度：self.out_proj.in_features

        同时做结构一致性校验，避免继续隐性连锁报错。
        """
        if not hasattr(self.dense, "out_features"):
            raise AttributeError("PoolPoolerBaseTuneV2Head: self.dense.out_features 不存在")
        if not hasattr(self.text_to_desc_proj, "out_features"):
            raise AttributeError(
                "PoolPoolerBaseTuneV2Head: self.text_to_desc_proj.out_features 不存在"
            )
        if not hasattr(self.out_proj, "in_features"):
            raise AttributeError(
                "PoolPoolerBaseTuneV2Head: self.out_proj.in_features 不存在"
            )

        pooler_repr_dim = int(self.dense.out_features)
        desc_fusion_dim = int(self.text_to_desc_proj.out_features)
        final_fused_dim = int(self.out_proj.in_features)

        # 中文注释：r_num 的输出维度应与 semantic 投影后的维度一致。
        if (
            isinstance(self.desc_mlp, nn.Sequential)
            and len(self.desc_mlp) > 0
            and hasattr(self.desc_mlp[-1], "out_features")
        ):
            desc_mlp_dim = int(self.desc_mlp[-1].out_features)
            if desc_mlp_dim != desc_fusion_dim:
                raise RuntimeError(
                    "PoolPoolerBaseTuneV2Head: desc_mlp 输出维度与 text_to_desc_proj 输出维度不一致，"
                    f"desc_mlp={desc_mlp_dim}, text_to_desc_proj={desc_fusion_dim}"
                )

        # 中文注释：最终拼接维度必须等于主路维度与 descriptor/semantic 维度之和。
        if pooler_repr_dim + desc_fusion_dim != final_fused_dim:
            raise RuntimeError(
                "PoolPoolerBaseTuneV2Head: final fused 维度与主路/descriptor 维度不一致，"
                f"pooler={pooler_repr_dim}, desc={desc_fusion_dim}, fused={final_fused_dim}"
            )

        return pooler_repr_dim, desc_fusion_dim, final_fused_dim

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        保持原有前向主线，只在三个低风险位置补归一化。
        """
        if not self.use_qsar_prompt_semantic_branch:
            return super().forward(x, smiles)

        batch_size = x.shape[0]

        # 中文注释：pooler 主路保持原来的 CLS -> dropout -> dense -> tanh -> dropout。
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        r_enc = self.dropout(x)
        r_enc = self.pooler_layernorm(r_enc)

        # 中文注释：descriptor 数值分支与旧版一致。
        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(r_enc.device)
        else:
            desc_raw = torch.zeros(batch_size, self.DESC_DIM, device=r_enc.device)

        desc_vals = desc_raw[:, : self.NUM_DESC]
        is_valid_flag = desc_raw[:, self.NUM_DESC :]
        desc_normalized = (desc_vals - self.desc_mean) / (self.desc_std + 1e-8)
        desc_full = torch.cat([desc_normalized, is_valid_flag], dim=-1)

        r_num = self.desc_mlp(desc_full)
        r_num = self.desc_dropout(r_num)
        r_num = self.desc_layernorm(r_num)

        # 中文注释：semantic 分支仍然只吃 descriptor prompt，不引入新复杂模块。
        r_text = self.semantic_text_encoder(
            raw_descriptors=desc_vals,
            valid_flags=is_valid_flag.squeeze(-1),
        )
        r_desc_new = self._fuse_descriptor_repr(r_num=r_num, r_text=r_text)
        r_desc_new = self.post_fusion_desc_layernorm(r_desc_new)

        # 中文注释：最终仍是 concat 后线性输出，只在 out_proj 前加一层轻量归一化。
        fused = torch.cat([r_enc, r_desc_new], dim=-1)
        fused = self.fused_layernorm(fused)
        fused = self.fusion_dropout(fused)
        logits = self.out_proj(fused)
        return logits
