# -*- coding: utf-8 -*-
"""
tasks/modules/dpc_knn.py
========================
【来源严格一致层】

本文件是
  D:/Python深度学习/smi修改/对比实验/分簇_DPC_KNN.py
的直接迁移，核心算法逻辑与来源保持严格一致。

唯一允许的最小适配（见末尾"适配说明"）：
  1. 文件名改为 ASCII（dpc_knn.py），使新项目可直接 import
  2. 模块顶部 docstring 增加来源声明

函数列表（均来自旧文件）
------------------------
  index_points              -- 工具函数，来自 Atomas cluster.py line 96-105
  cluster_dpc_knn           -- 核心算法，来自 Atomas cluster.py line 313-372
  merge_tokens              -- 核心算法，来自 Atomas cluster.py line 375-424
  dpc_knn_cluster_and_merge -- 适配层封装，来自旧文件 分簇_DPC_KNN.py line 166-228

适配说明（相对于旧文件，仅此两处，其余严格一致）
------------------------------------------------
  1. 文件名：分簇_DPC_KNN.py → dpc_knn.py（ASCII，解决 import 问题）
  2. 模块 docstring：新增本段说明文字

来源：D:/Python深度学习/smi修改/对比实验/分簇_DPC_KNN.py
原始来源（算法）：D:/Python深度学习/Atomas-main/models/cluster.py
"""

import torch
import torch.nn.functional as F


def index_points(points, idx):
    """
    根据 index 从 points 中提取对应的点

    Copy from Atomas cluster.py line 96-105

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, ...)

    Returns:
        new_points: (B, S, C) or (B, S, ..., C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """
    Cluster tokens with DPC-KNN algorithm.

    Copy from Atomas cluster.py line 313-372

    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """
    Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)

    Copy from Atomas cluster.py line 375-424

    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight

    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = None
    return out_dict


# ==================== 适配层：为 SMILES token 序列封装 ====================

def dpc_knn_cluster_and_merge(
    token_hidden: torch.Tensor,
    padding_mask: torch.Tensor,
    num_clusters: int = 8,
    knn_k: int = 8,
):
    """
    对 token_hidden 执行 DPC-KNN 聚类并合并

    Args:
        token_hidden: (B, L, D) - encoder 输出的 token 级别隐藏状态
        padding_mask: (B, L) - padding 掩码，1 表示有效 token，0 表示 pad
        num_clusters: 聚类数量 K
        knn_k: KNN 的 k 值，用于计算局部密度

    Returns:
        V: (B, K, D) - 聚类合并后的 token 表示
        cluster_mask: (B, K) - 聚类掩码，1 表示有效聚类
        centers_idx: (B, K) - 每个聚类中心的 token 索引
        assign_idx: (B, L) - 每个 token 的聚类分配索引
    """
    B, L, D = token_hidden.shape
    device = token_hidden.device

    # 构造 token_dict（Atomas 原接口需要的格式）
    token_dict = {
        'x': token_hidden,  # (B, L, D)
        'idx_token': torch.arange(L, device=device)[None, :].expand(B, -1),  # (B, L)
        'agg_weight': torch.ones(B, L, 1, device=device),  # (B, L, 1)
    }

    # 调用 Atomas 的 DPC-KNN 聚类
    idx_cluster, cluster_num = cluster_dpc_knn(
        token_dict,
        cluster_num=num_clusters,
        k=knn_k,
        token_mask=padding_mask,  # padding_mask: 1=有效, 0=pad
    )

    # 调用 Atomas 的 merge
    out_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None)

    # 提取结果
    V = out_dict['x']  # (B, K, D)

    # 生成 cluster_mask（实际上所有 cluster 都是有效的）
    cluster_mask = torch.ones(B, cluster_num, device=device, dtype=torch.long)

    # centers_idx: 找到每个聚类的中心 token
    # idx_cluster: (B, L) - 每个 token 属于哪个 cluster
    centers_idx = torch.zeros(B, cluster_num, dtype=torch.long, device=device)
    for b in range(B):
        for k in range(cluster_num):
            # 找到属于 cluster k 的所有 token
            mask = (idx_cluster[b] == k)
            if mask.any():
                # 取第一个作为中心（简化处理）
                centers_idx[b, k] = mask.nonzero()[0, 0]

    # assign_idx 就是 idx_cluster
    assign_idx = idx_cluster

    return V, cluster_mask, centers_idx, assign_idx
