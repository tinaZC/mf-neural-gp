#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]   # .../code_v2
PROJECT_ROOT = CODE_ROOT.parent                   # .../mf_neural_gp

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mf_train_baseline.mf_baseline import BaselineDefaults, main


if __name__ == "__main__":
    defaults = BaselineDefaults(
        data_dir=str(PROJECT_ROOT / "data/mf_sweep_datasets_nano_tm/hf100_lfx10"),
        out_dir=str(PROJECT_ROOT / "result_out/mf_baseline_nano_tm_out"),
        run_prefix="bl0",
        ours_train_script=str(CODE_ROOT / "nanophotonic_tm/mf_train_tm.py"),
    )
    main(defaults=defaults)