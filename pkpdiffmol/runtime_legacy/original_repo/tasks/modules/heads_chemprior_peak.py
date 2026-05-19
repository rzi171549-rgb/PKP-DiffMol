# -*- coding: utf-8 -*-
"""
tasks/modules/heads.py
======================
可复用 Head 模块（当前仅实现 ChemPriorHead）。

ChemPriorHead
-------------
将 encoder 输出的全序列隐藏态与 RDKit 化学描述符先验融合，
用于替代原 fairseq RobertaClassificationHead（CLS token 方案）。

接口约定
--------
  forward(token_hidden, pad_mask, smiles=None) -> logits (B, out_dim)

  token_hidden : (B, L, D)  encoder 输出全序列隐藏态
  pad_mask     : (B, L)     long tensor，1=有效 token，0=padding
  smiles       : List[str]  长度 B，用于计算 RDKit 描述符；
                            传 None 时描述符全置零（推理阶段兜底）

描述符（共 8 个 + 1 个 is_valid 标志 = 9 维）
--------------------------------------------
  MolWt / LogP / TPSA / NumHDonors / NumHAcceptors /
  NumRotatableBonds / RingCount / NumAromaticRings / is_valid

标准化
------
  mean/std 仅在训练集上 fit（调用 fit_normalizer），
  注册为 buffer 确保 checkpoint 保存 & 加载不丢失。
  未 fit 时 mean=0, std=1，等价于不标准化（安全兜底）。

迁移来源
--------
  原始实现参考：D:/Python深度学习/smi修改/对比实验/heads.py
  本版本精简为：无 gating，concat + MLP，可直接接入现有 BBBP 训练流程。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# --------------------------------------------------------------------------
# RDKit 可选依赖：缺失时 ChemPriorHead 仍可运行，描述符将全部置零
# --------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] tasks/modules/heads.py: RDKit 不可用，"
          "ChemPriorHead 将以全零描述符运行。")


# ===========================================================================
# ChemPriorHead
# ===========================================================================

class ChemPriorHead(nn.Module):
    """
    化学先验头（ChemPrior），简洁 concat 版本。

    数据流
    ------
    token_hidden (B, L, D)
      → masked mean pooling → pooled (B, D)
    smiles
      → RDKit → desc (B, 8) → z-score normalize (train stats)
      → concat is_valid → (B, 9)
      → desc_mlp → desc_embed (B, desc_hidden)
    concat([pooled, desc_embed]) (B, D + desc_hidden)
      → classifier → logits (B, out_dim)
    """

    # 8 个 RDKit 描述符（顺序固定，mean/std 与该顺序对应）
    DESCRIPTOR_NAMES: List[str] = [
        'MolWt',              # 分子量
        'LogP',               # 脂溶性
        'TPSA',               # 拓扑极性表面积
        'NumHDonors',         # 氢键供体数
        'NumHAcceptors',      # 氢键受体数
        'NumRotatableBonds',  # 可旋转键数
        'RingCount',          # 环数
        'NumAromaticRings',   # 芳香环数
    ]
    NUM_DESC = len(DESCRIPTOR_NAMES)   # = 8
    DESC_DIM = NUM_DESC + 1            # = 9（含 is_valid）

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        desc_hidden: int = 64,
        cls_hidden: int = 256,
        dropout: float = 0.1,
    ):
        """
        参数
        ----
        in_dim      : encoder 输出维度（LevenshteinEncoder = 768）
        out_dim     : 输出维度（BBBP=1，SIDER=27）
        desc_hidden : 描述符 MLP 隐藏层 / 输出维度
        cls_hidden  : 分类器隐藏层维度
        dropout     : Dropout 比率
        """
        super().__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.desc_hidden = desc_hidden

        # ---------- 描述符处理 MLP（9 → desc_hidden） ----------
        self.desc_mlp = nn.Sequential(
            nn.Linear(self.DESC_DIM, desc_hidden),
            nn.ReLU(),
            nn.Linear(desc_hidden, desc_hidden),
        )

        # ---------- 分类器 MLP（D + desc_hidden → out_dim） ----------
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + desc_hidden, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, out_dim),
        )

        # ---------- 标准化参数：注册为 buffer，随 checkpoint 一起保存 ----------
        # 初始值 mean=0, std=1 → 未 fit 时等价于不做标准化
        self.register_buffer('desc_mean', torch.zeros(self.NUM_DESC))  # (8,)
        self.register_buffer('desc_std',  torch.ones(self.NUM_DESC))   # (8,)
        self._normalizer_fitted = False   # 仅作记录，不进入 state_dict

    # ------------------------------------------------------------------
    # 公开 API：fit_normalizer（训练前调用一次）
    # ------------------------------------------------------------------

    def fit_normalizer(self, smiles_list: List[str], device: torch.device) -> None:
        """
        用训练集 SMILES 拟合描述符标准化参数（mean / std）。

        必须在训练循环开始前调用，且只能使用 train set 的 SMILES，
        禁止传入 valid / test 集数据（防止数据泄露）。

        参数
        ----
        smiles_list : 训练集所有 SMILES 字符串
        device      : 将 mean/std buffer 移至该设备
        """
        desc_np, valid_mask = self._compute_raw_descriptors(smiles_list)

        # 只对前 8 列（描述符值）做统计，is_valid 列不参与
        desc_vals = torch.from_numpy(desc_np[:, :self.NUM_DESC])  # (N, 8)
        valid     = torch.from_numpy(valid_mask).bool()           # (N,)

        n_valid = valid.sum().item()
        if n_valid > 0:
            valid_desc = desc_vals[valid]  # (n_valid, 8)
            new_mean   = valid_desc.mean(0)                   # (8,)
            new_std    = valid_desc.std(0).clamp(min=1e-6)    # (8,)，避免除零
            # in-place 更新 buffer（保持 device 一致）
            self.desc_mean.copy_(new_mean.to(device))
            self.desc_std.copy_(new_std.to(device))
            self._normalizer_fitted = True
            print(f"  [ChemPrior] fit_normalizer：{n_valid}/{len(smiles_list)} "
                  f"个有效分子，已更新 mean/std")
        else:
            print(f"  [ChemPrior] fit_normalizer：未找到任何有效分子，"
                  f"保留默认 mean=0, std=1")

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        token_hidden: torch.Tensor,
        pad_mask: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        参数
        ----
        token_hidden : (B, L, D)  encoder 全序列隐藏态
        pad_mask     : (B, L)     long，1=有效，0=padding
        smiles       : List[str] 长度 B；None 时描述符全零

        返回
        ----
        logits : (B, out_dim)
        """
        B, L, D = token_hidden.shape

        # ---- 1. masked mean pooling ----
        mask   = pad_mask.float().unsqueeze(-1)              # (B, L, 1)
        pooled = (token_hidden * mask).sum(1) / \
                 (mask.sum(1) + 1e-8)                        # (B, D)

        # ---- 2. RDKit 描述符计算 ----
        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(token_hidden.device)  # (B, 9)
        else:
            # 兜底：全零描述符（is_valid=0，表示无效）
            desc_raw = torch.zeros(B, self.DESC_DIM,
                                   device=token_hidden.device)             # (B, 9)

        # ---- 3. 标准化（只对前 8 列，使用 train stats） ----
        desc_vals      = desc_raw[:, :self.NUM_DESC]         # (B, 8)
        is_valid_flag  = desc_raw[:, self.NUM_DESC:]         # (B, 1)

        desc_normalized = (desc_vals - self.desc_mean) / \
                          (self.desc_std + 1e-8)             # (B, 8)，broadcasting

        # 拼回 is_valid 标志（不做标准化）
        desc_full = torch.cat([desc_normalized, is_valid_flag], dim=-1)  # (B, 9)

        # ---- 4. 描述符 embedding ----
        desc_embed = self.desc_mlp(desc_full)                # (B, desc_hidden)

        # ---- 5. concat + 分类器 ----
        fused  = torch.cat([pooled, desc_embed], dim=-1)     # (B, D + desc_hidden)
        logits = self.classifier(fused)                      # (B, out_dim)

        return logits

    # ------------------------------------------------------------------
    # 内部工具：批量计算 RDKit 描述符
    # ------------------------------------------------------------------

    def _compute_raw_descriptors(
        self,
        smiles_list: List[str],
    ):
        """
        批量计算 RDKit 描述符（CPU numpy，不需要梯度）。

        返回
        ----
        desc      : (N, 9) float32 ndarray
                    前 8 列为描述符值，第 9 列为 is_valid（0 或 1）
        valid_mask: (N,) float32 ndarray，1=解析成功
        """
        N    = len(smiles_list)
        desc = np.zeros((N, self.DESC_DIM), dtype=np.float32)
        valid = np.zeros(N, dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi) if RDKIT_AVAILABLE else None
                if mol is None:
                    continue
                desc[i, 0] = Descriptors.MolWt(mol)
                desc[i, 1] = Descriptors.MolLogP(mol)
                desc[i, 2] = Descriptors.TPSA(mol)
                desc[i, 3] = Descriptors.NumHDonors(mol)
                desc[i, 4] = Descriptors.NumHAcceptors(mol)
                desc[i, 5] = Descriptors.NumRotatableBonds(mol)
                desc[i, 6] = rdMolDescriptors.CalcNumRings(mol)
                desc[i, 7] = rdMolDescriptors.CalcNumAromaticRings(mol)
                desc[i, 8] = 1.0   # is_valid = 1
                valid[i]   = 1.0
            except Exception:
                # RDKit 解析异常：保持全零（is_valid=0）
                pass

        return desc, valid


# ===========================================================================
# AtomasDPCPoolingHead
# ===========================================================================

class AtomasDPCPoolingHead(nn.Module):
    """
    Atomas DPC-KNN 分簇 + WAM 多 query 注意力聚合头。

    数据流
    ------
    token_hidden (B, L, D)
      [自动排除 CLS/SEP]
      → DPC-KNN 聚类 → K 个 cluster 表示 V (B, K, D)
      → M 个可学习 query 对 V 做注意力（WAM）→ M 个 evidence 向量 (B, M, D)
      → evidence 加权求和 → pooled (B, D)
      → classifier MLP → logits (B, out_dim)

    接口
    ----
    forward(token_hidden, pad_mask) -> logits (B, out_dim)
      不需要 SMILES 字符串。

    参数
    ----
    in_dim       : encoder 输出维度（LevenshteinEncoder = 768）
    out_dim      : 输出维度（BBBP=1）
    num_clusters : DPC-KNN 聚类数 K（默认 8）
    knn_k        : KNN 局部密度的邻居数（默认 5）
    num_queries  : WAM evidence query 数量 M（默认 8）
    hidden_dim   : 分类器隐藏层维度（默认 256）
    dropout      : Dropout 比率（默认 0.1）

    移植来源
    --------
    原始: D:/Python深度学习/smi修改/对比实验/heads.py  AtomasDPCPoolingHead
    本版本重写以对接 tasks/modules/dpc_knn.py，去掉动态 import，
    接口简化为 forward(token_hidden, pad_mask) -> logits。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        num_clusters: int = 8,
        knn_k: int = 5,
        num_queries: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.num_clusters = num_clusters
        self.knn_k        = knn_k
        self.num_queries  = num_queries
        self.hidden_dim   = hidden_dim

        # ---------- WAM：M 个可学习 evidence query (M, D) ----------
        self.evidence_queries = nn.Parameter(torch.randn(num_queries, in_dim))

        # ---------- evidence 权重网络 (D -> 1) ----------
        self.evidence_weight_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---------- 分类器 MLP (D -> out_dim) ----------
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数
        ----
        token_hidden : (B, L, D)  encoder 全序列隐藏态
        pad_mask     : (B, L)     long，1=有效，0=padding

        返回
        ----
        logits : (B, out_dim)
        """
        from dpc_knn import dpc_knn_cluster_and_merge  # 同目录，运行时导入

        B, L, D = token_hidden.shape
        M = self.num_queries

        # ---- 1. DPC-KNN 聚类 ----
        # 【来源一致版】dpc_knn_cluster_and_merge 返回 4 个值；
        # 本 head 只需要 V 和 cluster_mask，centers_idx / assign_idx 忽略。
        V, cluster_mask, _centers_idx, _assign_idx = dpc_knn_cluster_and_merge(
            token_hidden, pad_mask,
            num_clusters=self.num_clusters,
            knn_k=self.knn_k,
        )
        # V: (B, K, D)，cluster_mask: (B, K) all-ones

        K = V.shape[1]  # 实际 cluster 数（通常 = num_clusters）

        # ---- 2. WAM：M queries 对 K clusters 做注意力 ----
        queries = self.evidence_queries.unsqueeze(0).expand(B, M, D)  # (B, M, D)

        # scores: (B, M, K)
        scores = torch.bmm(queries, V.transpose(1, 2)) / (D ** 0.5)

        # cluster_mask 全 1，此处无需遮蔽，直接 softmax
        cluster_attention = F.softmax(scores, dim=-1)  # (B, M, K)

        # evidence 向量：(B, M, K) @ (B, K, D) -> (B, M, D)
        evidence = torch.bmm(cluster_attention, V)     # (B, M, D)

        # ---- 3. evidence 加权聚合 ----
        ev_weight_raw = self.evidence_weight_net(evidence).squeeze(-1)  # (B, M)
        ev_weight     = F.softmax(ev_weight_raw, dim=-1)                # (B, M)

        # (B, M, D) * (B, M, 1) -> sum over M -> (B, D)
        pooled = (evidence * ev_weight.unsqueeze(-1)).sum(dim=1)        # (B, D)

        # ---- 4. 分类器 ----
        logits = self.classifier(pooled)                                # (B, out_dim)
        return logits


# ===========================================================================
# BaselineEquivalentHead
# ===========================================================================

class BaselineEquivalentHead(nn.Module):
    """
    外挂版 BBBP baseline 预测头（CLS token 路径）。

    严格复刻 fairseq RobertaClassificationHead 的主路径：
      features[:, 0, :]
        → dropout
        → dense(D → D)
        → tanh
        → dropout
        → out_proj(D → out_dim)

    与原头完全对齐的设计决策
    -------------------------
    - 一个 dropout 模块，在 dense 前和 tanh 后各用一次（与原头完全相同）
    - activation = tanh（对齐 pooler_activation_fn 默认值 'tanh'）
    - dense: Linear(in_dim → in_dim)（对齐原头 inner_dim = input_dim = 768）
    - out_proj: Linear(in_dim → out_dim)（对齐原头 out_proj）
    - 权重初始化：PyTorch 默认 Linear init（与原头相同，原头未调用 init_bert_params）

    接口
    ----
    forward(x) -> logits (B, out_dim)
      x : (B, L, D)  encoder 全序列隐藏态（features_only=True 的输出）

    对应来源
    --------
    fairseq/models/roberta/model.py  RobertaClassificationHead  line 498-533
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        """
        参数
        ----
        in_dim  : encoder 输出维度（SMI-Editor = 768）
        out_dim : 输出类别数（BBBP = 1）
        dropout : Dropout 概率，对齐原 baseline pooler_dropout=0.1
        """
        super().__init__()
        # 对应原头 self.dense = nn.Linear(input_dim, inner_dim)
        self.dense    = nn.Linear(in_dim, in_dim)
        # 对应原头 self.dropout = nn.Dropout(p=pooler_dropout)
        # 单个实例，在 forward 中调用两次——与原头完全相同
        self.dropout  = nn.Dropout(p=dropout)
        # 对应原头 self.out_proj = nn.Linear(inner_dim, num_classes)
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : (B, L, D)  encoder 全序列隐藏态

        返回
        ----
        logits : (B, out_dim)
        """
        x = x[:, 0, :]      # CLS token — 对应原头 features[:, 0, :]  shape: (B, D)
        x = self.dropout(x)  # 第一次 dropout                          shape: (B, D)
        x = self.dense(x)    # Linear(D → D)                           shape: (B, D)
        x = torch.tanh(x)   # tanh — 对应原头 pooler_activation_fn     shape: (B, D)
        x = self.dropout(x)  # 第二次 dropout                          shape: (B, D)
        x = self.out_proj(x) # Linear(D → out_dim)                     shape: (B, out_dim)
        return x


# ===========================================================================
# BaselineEquivalentChemPriorHead
# ===========================================================================

class BaselineEquivalentChemPriorHead(nn.Module):
    """
    严格版 ChemPrior 模块消融头。

    在 BaselineEquivalentHead 主路径上，仅增加 RDKit descriptor 分支：
      - baseline 主路径：与 BaselineEquivalentHead 完全一致（CLS token）
      - descriptor 分支：12 个 RDKit 描述符 + is_valid flag（共 13 维）→ 小 MLP
      - 融合：concat(x_proj_after_dropout, desc_repr) → out_proj

    对比 BaselineEquivalentHead 的唯一改动
    ----------------------------------------
    1. 多一个 desc_mlp：Linear(13 → desc_hidden) → ReLU → Linear(desc_hidden → desc_hidden)
    2. out_proj 输入维度：768 → 768 + desc_hidden
    3. forward 多接收 smiles 参数，用于计算 RDKit 描述符
    4. fit_normalizer：训练前调用一次，用 train 集 fit mean/std（注册为 buffer）

    严格消融说明
    ------------
    pooling 策略：CLS token（不是 mean pooling，与 baseline 一致）
    baseline 路径：x_cls → dropout → dense(768→768) → tanh → x_proj → dropout
    descriptor 融合：在 x_proj 经过第二次 dropout 之后、out_proj 之前做 concat
    这样确保：除 descriptor 分支外，其余路径与 baseline-equivalent 完全一致

    接口
    ----
    fit_normalizer(smiles_list, device) -> None   训练前调用一次
    forward(x, smiles=None) -> logits (B, out_dim)
      x      : (B, L, D)  encoder 全序列隐藏态
      smiles : List[str] 长度 B；None 时描述符全零

    对应来源
    --------
    baseline 路径参考: fairseq/models/roberta/model.py RobertaClassificationHead line 526-533
    descriptor 参考:  tasks/modules/heads.py ChemPriorHead（描述符计算逻辑一致）
    """

    DESCRIPTOR_NAMES: List[str] = [
        'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
        'FormalCharge', 'MolMR', 'LabuteASA', 'FractionCSP3',
    ]
    NUM_DESC = len(DESCRIPTOR_NAMES)   # = 12
    DESC_DIM = NUM_DESC + 1            # = 13（含 is_valid）

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        desc_hidden: int = 64,
        dropout: float = 0.1,
        desc_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        use_desc_layernorm: bool = False,
    ):
        """
        参数
        ----
        in_dim      : encoder 输出维度（SMI-Editor = 768）
        out_dim     : 输出类别数（BBBP = 1）
        desc_hidden : descriptor MLP 输出维度（默认 64）
        dropout            : Dropout 概率，对齐原 baseline pooler_dropout=0.1
        desc_dropout       : descriptor 分支上的额外 dropout（默认关闭）
        fusion_dropout     : concat 后、out_proj 前的额外 dropout（默认关闭）
        use_desc_layernorm : 是否在 concat 前对 desc_repr 做 LayerNorm（默认关闭）
        """
        super().__init__()

        # ---------- baseline 主路径（与 BaselineEquivalentHead 完全一致）----------
        self.dense   = nn.Linear(in_dim, in_dim)     # 对应原头 dense: Linear(768→768)
        self.dropout = nn.Dropout(p=dropout)          # 单模块，调用两次，与原头一致

        # ---------- descriptor 分支（新增）----------
        self.desc_mlp = nn.Sequential(
            nn.Linear(self.DESC_DIM, desc_hidden),
            nn.ReLU(),
            nn.Linear(desc_hidden, desc_hidden),
        )
        self.desc_dropout = nn.Dropout(p=desc_dropout)
        self.desc_layernorm = (
            nn.LayerNorm(desc_hidden) if use_desc_layernorm else nn.Identity()
        )
        self.fusion_dropout = nn.Dropout(p=fusion_dropout)

        # ---------- 融合分类器（in_dim 扩宽为 in_dim + desc_hidden）----------
        # 对应原头 out_proj，但输入维度由 768 扩展到 768 + desc_hidden
        self.out_proj = nn.Linear(in_dim + desc_hidden, out_dim)

        # ---------- 标准化参数：注册为 buffer，随 checkpoint 一起保存 ----------
        self.register_buffer('desc_mean', torch.zeros(self.NUM_DESC))  # (12,)
        self.register_buffer('desc_std',  torch.ones(self.NUM_DESC))   # (12,)
        self._normalizer_fitted = False

    # ------------------------------------------------------------------
    # 公开 API：fit_normalizer（训练前调用一次，只用 train 集）
    # ------------------------------------------------------------------

    def fit_normalizer(self, smiles_list: List[str], device: torch.device) -> None:
        """
        用训练集 SMILES 拟合描述符均值和标准差。

        必须在训练循环开始前调用一次，且只能传入 train 集（防数据泄露）。
        """
        desc_np, valid_mask = self._compute_raw_descriptors(smiles_list)
        desc_vals = torch.from_numpy(desc_np[:, :self.NUM_DESC])  # (N, 12)
        valid     = torch.from_numpy(valid_mask).bool()           # (N,)

        n_valid = valid.sum().item()
        if n_valid > 0:
            valid_desc = desc_vals[valid]
            new_mean   = valid_desc.mean(0)
            new_std    = valid_desc.std(0).clamp(min=1e-6)
            self.desc_mean.copy_(new_mean.to(device))
            self.desc_std.copy_(new_std.to(device))
            self._normalizer_fitted = True
            print(f"  [BaselineEqChemPrior] fit_normalizer: "
                  f"{n_valid}/{len(smiles_list)} 个有效分子，已更新 mean/std")
        else:
            print(f"  [BaselineEqChemPrior] fit_normalizer: "
                  f"未找到有效分子，保留默认 mean=0, std=1")

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        参数
        ----
        x      : (B, L, D)  encoder 全序列隐藏态
        smiles : List[str] 长度 B；None 时描述符全零

        返回
        ----
        logits : (B, out_dim)
        """
        B = x.shape[0]

        # ---- baseline 主路径（与 BaselineEquivalentHead 完全一致）----
        x = x[:, 0, :]       # CLS token                   shape: (B, D)
        x = self.dropout(x)   # 第一次 dropout               shape: (B, D)
        x = self.dense(x)     # Linear(D → D)                shape: (B, D)
        x = torch.tanh(x)    # tanh                          shape: (B, D)
        x = self.dropout(x)   # 第二次 dropout               shape: (B, D)
        # ↑ 此时 x 就是 baseline 中 out_proj 的输入；融合在此处发生

        # ---- descriptor 分支（新增）----
        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(x.device)  # (B, 13)
        else:
            desc_raw = torch.zeros(B, self.DESC_DIM, device=x.device)  # (B, 13)

        # 对前 NUM_DESC 列做 z-score 标准化（使用 train 统计量）
        desc_vals       = desc_raw[:, :self.NUM_DESC]          # (B, 12)
        is_valid_flag   = desc_raw[:, self.NUM_DESC:]          # (B, 1)
        desc_normalized = (desc_vals - self.desc_mean) / (self.desc_std + 1e-8)
        desc_full       = torch.cat([desc_normalized, is_valid_flag], dim=-1)  # (B, 13)

        desc_repr = self.desc_mlp(desc_full)                   # (B, desc_hidden)
        desc_repr = self.desc_dropout(desc_repr)               # (B, desc_hidden)
        desc_repr = self.desc_layernorm(desc_repr)             # (B, desc_hidden)

        # ---- concat 融合（x_proj_after_dropout + desc_repr）----
        fused  = torch.cat([x, desc_repr], dim=-1)             # (B, D + desc_hidden)
        fused  = self.fusion_dropout(fused)                    # (B, D + desc_hidden)
        logits = self.out_proj(fused)                          # (B, out_dim)
        return logits

    # ------------------------------------------------------------------
    # 内部工具：批量计算 RDKit 描述符（与 ChemPriorHead 相同逻辑）
    # ------------------------------------------------------------------

    def _compute_raw_descriptors(self, smiles_list: List[str]):
        """
        返回 (desc: (N, 13) float32 ndarray, valid_mask: (N,) float32 ndarray)
        前 12 列为描述符值，第 13 列为 is_valid（0 或 1）。
        """
        N    = len(smiles_list)
        desc = np.zeros((N, self.DESC_DIM), dtype=np.float32)
        valid = np.zeros(N, dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi) if RDKIT_AVAILABLE else None
                if mol is None:
                    continue
                desc[i, 0] = Descriptors.MolWt(mol)
                desc[i, 1] = Descriptors.MolLogP(mol)
                desc[i, 2] = Descriptors.TPSA(mol)
                desc[i, 3] = Descriptors.NumHDonors(mol)
                desc[i, 4] = Descriptors.NumHAcceptors(mol)
                desc[i, 5] = Descriptors.NumRotatableBonds(mol)
                desc[i, 6] = rdMolDescriptors.CalcNumRings(mol)
                desc[i, 7] = rdMolDescriptors.CalcNumAromaticRings(mol)
                desc[i, 8] = Chem.GetFormalCharge(mol)
                desc[i, 9] = Descriptors.MolMR(mol)
                desc[i, 10] = rdMolDescriptors.CalcLabuteASA(mol)
                desc[i, 11] = rdMolDescriptors.CalcFractionCSP3(mol)
                desc[i, 12] = 1.0
                valid[i]   = 1.0
            except Exception:
                pass

        return desc, valid


# ===========================================================================
# BaselineEqChemPriorDPCHead
# ===========================================================================

class BaselineEqChemPriorDPCHead(nn.Module):
    """
    严格版 ChemPrior DPC 消融头。

    在 BaselineEquivalentChemPriorHead 的基础上，
    仅把 CLS token 提取（x = x[:, 0, :]）替换为
    DPC-KNN 分簇 + WAM pooling，其余路径完全不变：
      - baseline 主路径（dropout → dense(768→768) → tanh → dropout）
      - descriptor 分支（RDKit 8 描述符 → desc_mlp → desc_repr）
      - 融合层（concat → out_proj(832→1)）

    数据流
    ------
    x (B, L, D)
      → DPC-KNN clustering + merge → V (B, K, D)
      → WAM: M queries × K clusters → evidence (B, M, D)
      → evidence weighting → pooled (B, D)
      → dropout → dense(D→D) → tanh → dropout      ← baseline 主路径（不变）
    smiles
      → RDKit(8) + is_valid → desc_mlp(9→desc_hidden) → desc_repr (B, desc_hidden)
    concat([pooled_after_dropout, desc_repr])         (B, D + desc_hidden)
      → out_proj(D+desc_hidden→out_dim)              (B, out_dim)

    与 BaselineEquivalentChemPriorHead 的唯一差异
    -----------------------------------------------
    原:  x = x[:, 0, :]         （CLS 单点提取）
    新:  DPC-KNN clustering + WAM pooling → pooled (B, D)

    新增超参
    --------
    num_clusters : int  DPC-KNN 聚类数 K（默认 8）
    knn_k        : int  KNN 局部密度邻居数（默认 5）
    num_queries  : int  WAM evidence query 数量 M（默认 8）

    接口
    ----
    fit_normalizer(smiles_list, device) -> None   训练前调用一次
    forward(x, smiles=None, pad_mask=None) -> logits (B, out_dim)
      x        : (B, L, D)  encoder 全序列隐藏态
      smiles   : List[str] 长度 B；None 时描述符全零
      pad_mask : (B, L) long，1=有效，0=padding；不可为 None

    来源
    ----
    DPC-WAM pooling  : tasks/modules/heads.py  AtomasDPCPoolingHead（逻辑一致）
    descriptor 路径  : tasks/modules/heads.py  BaselineEquivalentChemPriorHead（原样）
    baseline 主路径  : tasks/modules/heads.py  BaselineEquivalentChemPriorHead（原样）
    DPC-KNN 算法     : tasks/modules/dpc_knn.py（不修改）
    """

    DESCRIPTOR_NAMES: List[str] = [
        'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
    ]
    NUM_DESC = len(DESCRIPTOR_NAMES)   # = 8
    DESC_DIM = NUM_DESC + 1            # = 9（含 is_valid）

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        desc_hidden: int = 64,
        dropout: float = 0.1,
        num_clusters: int = 8,
        knn_k: int = 5,
        num_queries: int = 8,
    ):
        """
        参数
        ----
        in_dim       : encoder 输出维度（SMI-Editor = 768）
        out_dim      : 输出类别数（BBBP = 1）
        desc_hidden  : descriptor MLP 输出维度（默认 64）
        dropout      : Dropout 概率，对齐原 baseline pooler_dropout=0.1
        num_clusters : DPC-KNN 聚类数 K（默认 8）
        knn_k        : KNN 局部密度邻居数（默认 5）
        num_queries  : WAM evidence query 数量 M（默认 8）
        """
        super().__init__()

        # ---------- DPC-WAM pooling 组件（新增，替代 CLS 提取）----------
        self.num_clusters = num_clusters
        self.knn_k        = knn_k
        self.num_queries  = num_queries

        # 可学习的 evidence queries: (M, D)，与 AtomasDPCPoolingHead 结构一致
        self.evidence_queries = nn.Parameter(torch.randn(num_queries, in_dim))

        # evidence 权重网络 (D → 1)，与 AtomasDPCPoolingHead 结构一致
        self.evidence_weight_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )

        # ---------- baseline 主路径（与 BaselineEquivalentChemPriorHead 完全一致）----------
        self.dense   = nn.Linear(in_dim, in_dim)   # 对应原头 dense: Linear(768→768)
        self.dropout = nn.Dropout(p=dropout)        # 单模块，调用两次，与原头一致

        # ---------- descriptor 分支（与 BaselineEquivalentChemPriorHead 完全一致）----------
        self.desc_mlp = nn.Sequential(
            nn.Linear(self.DESC_DIM, desc_hidden),
            nn.ReLU(),
            nn.Linear(desc_hidden, desc_hidden),
        )

        # ---------- 融合分类器（与 BaselineEquivalentChemPriorHead 完全一致）----------
        # 输入维度：in_dim + desc_hidden（默认 768+64=832）
        self.out_proj = nn.Linear(in_dim + desc_hidden, out_dim)

        # ---------- 标准化参数 buffer（与 BaselineEquivalentChemPriorHead 完全一致）----------
        self.register_buffer('desc_mean', torch.zeros(self.NUM_DESC))   # (8,)
        self.register_buffer('desc_std',  torch.ones(self.NUM_DESC))    # (8,)
        self._normalizer_fitted = False

    # ------------------------------------------------------------------
    # 公开 API：fit_normalizer（逻辑与 BaselineEquivalentChemPriorHead 完全相同）
    # ------------------------------------------------------------------

    def fit_normalizer(self, smiles_list: List[str], device: torch.device) -> None:
        """
        用训练集 SMILES 拟合描述符均值和标准差。
        逻辑与 BaselineEquivalentChemPriorHead.fit_normalizer 完全相同。
        """
        desc_np, valid_mask = self._compute_raw_descriptors(smiles_list)
        desc_vals = torch.from_numpy(desc_np[:, :self.NUM_DESC])   # (N, 8)
        valid     = torch.from_numpy(valid_mask).bool()            # (N,)

        n_valid = valid.sum().item()
        if n_valid > 0:
            valid_desc = desc_vals[valid]
            new_mean   = valid_desc.mean(0)
            new_std    = valid_desc.std(0).clamp(min=1e-6)
            self.desc_mean.copy_(new_mean.to(device))
            self.desc_std.copy_(new_std.to(device))
            self._normalizer_fitted = True
            print(f"  [BaselineEqChemPriorDPC] fit_normalizer: "
                  f"{n_valid}/{len(smiles_list)} 个有效分子，已更新 mean/std")
        else:
            print(f"  [BaselineEqChemPriorDPC] fit_normalizer: "
                  f"未找到有效分子，保留默认 mean=0, std=1")

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        smiles: Optional[List[str]] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数
        ----
        x        : (B, L, D)  encoder 全序列隐藏态
        smiles   : List[str] 长度 B；None 时描述符全零
        pad_mask : (B, L) long，1=有效，0=padding；不可为 None

        返回
        ----
        logits : (B, out_dim)
        """
        if pad_mask is None:
            raise ValueError(
                "BaselineEqChemPriorDPCHead.forward: pad_mask 不可为 None。"
                "请在训练循环中构造 pad_mask = (tokens != pad_idx).long() 后传入。"
            )

        from dpc_knn import dpc_knn_cluster_and_merge

        B, L, D = x.shape
        M = self.num_queries

        # ---- 1. DPC-KNN 分簇 + merge（替换原 CLS 提取）----
        # cluster_mask 由 dpc_knn_cluster_and_merge 返回，当前实现全为 1，无需额外遮蔽
        V, _cluster_mask, _, _ = dpc_knn_cluster_and_merge(
            x, pad_mask,
            num_clusters=self.num_clusters,
            knn_k=self.knn_k,
        )
        # V: (B, K, D)

        # ---- 2. WAM pooling（与 AtomasDPCPoolingHead 逻辑一致）----
        queries = self.evidence_queries.unsqueeze(0).expand(B, M, D)    # (B, M, D)
        scores  = torch.bmm(queries, V.transpose(1, 2)) / (D ** 0.5)   # (B, M, K)
        cluster_attention = F.softmax(scores, dim=-1)                    # (B, M, K)
        evidence = torch.bmm(cluster_attention, V)                       # (B, M, D)

        ev_weight_raw = self.evidence_weight_net(evidence).squeeze(-1)   # (B, M)
        ev_weight     = F.softmax(ev_weight_raw, dim=-1)                 # (B, M)

        pooled = (evidence * ev_weight.unsqueeze(-1)).sum(dim=1)         # (B, D)

        # ---- 3. baseline 主路径（与 BaselineEquivalentChemPriorHead L611–L614 完全一致）----
        x = pooled
        x = self.dropout(x)    # 第一次 dropout               shape: (B, D)
        x = self.dense(x)      # Linear(D → D)                shape: (B, D)
        x = torch.tanh(x)     # tanh                          shape: (B, D)
        x = self.dropout(x)    # 第二次 dropout               shape: (B, D)
        # ↑ 此时 x 等价于 BaselineEquivalentChemPriorHead 中 out_proj 前的 x_proj_after_dropout

        # ---- 4. descriptor 分支（与 BaselineEquivalentChemPriorHead L618–L630 完全一致）----
        if smiles is not None and RDKIT_AVAILABLE:
            desc_np, _ = self._compute_raw_descriptors(smiles)
            desc_raw = torch.from_numpy(desc_np).to(x.device)   # (B, 9)
        else:
            desc_raw = torch.zeros(B, self.DESC_DIM, device=x.device)   # (B, 9)

        desc_vals       = desc_raw[:, :self.NUM_DESC]            # (B, 8)
        is_valid_flag   = desc_raw[:, self.NUM_DESC:]            # (B, 1)
        desc_normalized = (desc_vals - self.desc_mean) / (self.desc_std + 1e-8)
        desc_full       = torch.cat([desc_normalized, is_valid_flag], dim=-1)   # (B, 9)

        desc_repr = self.desc_mlp(desc_full)                     # (B, desc_hidden)

        # ---- 5. concat 融合（与 BaselineEquivalentChemPriorHead L633–L634 完全一致）----
        fused  = torch.cat([x, desc_repr], dim=-1)               # (B, D + desc_hidden)
        logits = self.out_proj(fused)                            # (B, out_dim)
        return logits

    # ------------------------------------------------------------------
    # 内部工具：批量计算 RDKit 描述符（与 BaselineEquivalentChemPriorHead 完全相同）
    # ------------------------------------------------------------------

    def _compute_raw_descriptors(self, smiles_list: List[str]):
        """
        返回 (desc: (N, 9) float32 ndarray, valid_mask: (N,) float32 ndarray)
        逻辑与 BaselineEquivalentChemPriorHead._compute_raw_descriptors 完全相同。
        """
        N     = len(smiles_list)
        desc  = np.zeros((N, self.DESC_DIM), dtype=np.float32)
        valid = np.zeros(N, dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi) if RDKIT_AVAILABLE else None
                if mol is None:
                    continue
                desc[i, 0] = Descriptors.MolWt(mol)
                desc[i, 1] = Descriptors.MolLogP(mol)
                desc[i, 2] = Descriptors.TPSA(mol)
                desc[i, 3] = Descriptors.NumHDonors(mol)
                desc[i, 4] = Descriptors.NumHAcceptors(mol)
                desc[i, 5] = Descriptors.NumRotatableBonds(mol)
                desc[i, 6] = rdMolDescriptors.CalcNumRings(mol)
                desc[i, 7] = rdMolDescriptors.CalcNumAromaticRings(mol)
                desc[i, 8] = 1.0
                valid[i]   = 1.0
            except Exception:
                pass

        return desc, valid
