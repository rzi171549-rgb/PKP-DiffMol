# -*- coding: utf-8 -*-
"""
BACE / SIDER 第4组小步回调版默认参数。

设计原则：
1. 复用已经对齐正式 protocol 的 optimized 三阶段实现。
2. 只覆盖 BACE / SIDER 的少量默认超参，不改旧 general / stage1 / head。
3. 最终 test protocol 仍然固定为 Stage C。
"""

import copy

import chemprior_parallel_latentdiff_bace_sider_optimized as _base


EXPERIMENT_SUFFIX = "chemprior_parallel_latentdiff_optimized_v2"
OPTIMIZED_DEFAULTS = copy.deepcopy(_base.OPTIMIZED_DEFAULTS)

# BACE：保留本轮早停框架，只把 diffusion 强度和正则往旧高分区小步回调。
OPTIMIZED_DEFAULTS["bace"].update(
    {
        "epochs": 60,
        "batch_size": 64,
        "lr": 1e-4,
        "warmup_ratio": 0.06,
        "pooler_dropout": 0.20,
        "best_metric": "hybrid",
        "min_best_epoch": 1,
        "stage_a_patience": 0,
        "synthetic_rho": 0.05,
        "use_qgate": True,
        "qgate_quantile": 0.95,
        "diffusion_num_timesteps": 50,
    }
)

# SIDER：保留当前稳定的 Stage A，只小步放松 qgate，并略微延长 diffusion。
OPTIMIZED_DEFAULTS["sider"].update(
    {
        "diffusion_epochs": 22,
        "qgate_quantile": 0.97,
    }
)


def _activate_v2_defaults():
    """
    将 v2 默认值写回复用模块，避免复制整套训练流程。
    """
    _base.EXPERIMENT_SUFFIX = EXPERIMENT_SUFFIX
    _base.OPTIMIZED_DEFAULTS = copy.deepcopy(OPTIMIZED_DEFAULTS)


def build_parser(dataset_name: str):
    """
    复用旧实现的 parser，并切换到 v2 默认值。
    """
    _activate_v2_defaults()
    return _base.build_parser(dataset_name)


def run_optimized_experiment(dataset_name: str):
    """
    复用旧实现的三阶段训练流程，并切换到 v2 默认值。
    """
    _activate_v2_defaults()
    return _base.run_optimized_experiment(dataset_name)


def main(dataset_name: str):
    """
    供数据集 wrapper 调用的入口。
    """
    return run_optimized_experiment(dataset_name)
