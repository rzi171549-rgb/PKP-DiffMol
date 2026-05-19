#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BACE Stage C residual mix 方法版训练入口。"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../modules"))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from chemprior_parallel_latentdiff_bace_stagec_residual_mix import main


if __name__ == "__main__":
    main("bace")
