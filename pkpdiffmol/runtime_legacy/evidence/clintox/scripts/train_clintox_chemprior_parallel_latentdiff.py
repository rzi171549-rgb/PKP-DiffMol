#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ClinTox 第4组训练入口。"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../modules"))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from chemprior_parallel_latentdiff_general import main


if __name__ == "__main__":
    main("clintox")
