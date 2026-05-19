# -*- coding: utf-8 -*-
"""
并行 latent diffusion 版 tunev2 head。

设计目标：
1. 不改旧主线文件；
2. 保持当前 tunev2 的 r_num / r_text / r_desc / r_enc / fused / logits 主流程；
3. 显式暴露最终 fused representation；
4. extract_fused_latent() 返回的 fused 必须是：
   fused_layernorm -> fusion_dropout 之后、out_proj 之前的张量。
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from heads_chemprior_peak import RDKIT_AVAILABLE
    from heads_chemprior_peak_qsarprompt_semantic_tunev2 import PoolPoolerBaseTuneV2Head
except ImportError:  # pragma: no cover
    from .heads_chemprior_peak import RDKIT_AVAILABLE
    from .heads_chemprior_peak_qsarprompt_semantic_tunev2 import PoolPoolerBaseTuneV2Head


class PoolPoolerBaseTuneV2LatentDiffHead(PoolPoolerBaseTuneV2Head):
    """
    在不污染旧 head 的前提下，增加 fused latent 暴露接口。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 中文注释：显式保存最终 fused 维度，供 diffusion 分支直接读取。
        self.final_fused_dim = int(self.out_proj.in_features)
        self.classifier_out_dim = int(self.out_proj.out_features)

    def _encode_parallel_parts(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        复用当前 tunev2 主流程，并把关键中间表示全部显式返回。
        """
        batch_size = x.shape[0]

        # 中文注释：主干 pooling 路径与当前 tunev2 保持一致。
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        r_enc = self.dropout(x)
        r_enc = self.pooler_layernorm(r_enc)

        # 中文注释：descriptor 数值路保持当前 tunev2 一致。
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

        # 中文注释：最小 Exp-1 默认仍走 semantic descriptor prompt 分支。
        if self.use_qsar_prompt_semantic_branch:
            r_text = self.semantic_text_encoder(
                raw_descriptors=desc_vals,
                valid_flags=is_valid_flag.squeeze(-1),
            )
            r_desc = self._fuse_descriptor_repr(r_num=r_num, r_text=r_text)
        else:
            r_text = None
            r_desc = r_num

        r_desc = self.post_fusion_desc_layernorm(r_desc)

        # 中文注释：这里的 fused 就是后续 diffusion 允许接入的唯一位置。
        fused = torch.cat([r_enc, r_desc], dim=-1)
        fused = self.fused_layernorm(fused)
        fused = self.fusion_dropout(fused)

        return {
            "r_enc": r_enc,
            "r_num": r_num,
            "r_text": r_text,
            "r_desc": r_desc,
            "fused": fused,
        }

    def extract_fused_latent(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        提取最终 fused latent。

        返回值严格对应：
        fused_layernorm -> fusion_dropout 之后、out_proj 之前的 fused。
        """
        outputs = self._encode_parallel_parts(x=x, smiles=smiles)
        fused = outputs["fused"]
        if fused is None:
            raise RuntimeError("未能提取 fused latent。")
        return fused.detach() if detach else fused

    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """
        直接对 fused latent 走分类头。
        """
        if fused.ndim != 2:
            raise ValueError("fused 必须是二维张量 (B, Dz)。")
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
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        返回 logits + 中间表示，供 Stage B 提取 fused latent 使用。
        """
        outputs = self._encode_parallel_parts(x=x, smiles=smiles)
        fused = outputs["fused"]
        if fused is None:
            raise RuntimeError("未能构建 fused latent。")

        logits = self.forward_from_fused(fused)
        outputs["fused"] = fused.detach() if detach_latent else fused
        outputs["logits"] = logits
        return outputs

    def named_classifier_parameters(self) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        """
        只暴露最终分类头参数，供 Stage C 单独训练。
        """
        for name, parameter in self.out_proj.named_parameters():
            yield f"out_proj.{name}", parameter

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        默认行为仍保持为“输入 backbone hidden，输出 logits”。
        """
        outputs = self.forward_with_aux(x=x, smiles=smiles, detach_latent=False)
        logits = outputs["logits"]
        if logits is None:
            raise RuntimeError("未能得到 logits。")
        return logits
