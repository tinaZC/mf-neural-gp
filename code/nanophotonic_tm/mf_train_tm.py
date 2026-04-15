#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]      # .../code_v2
PROJECT_ROOT = CODE_ROOT.parent                      # .../mf_neural_gp

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mf_train_baseline.mf_train import TrainDefaults, main


if __name__ == "__main__":
    defaults = TrainDefaults(
        data_dir=str(PROJECT_ROOT / "data/mf_sweep_datasets_nano_tm/hf100_lfx10"),
        out_dir=str(PROJECT_ROOT / "result_out/mf_train_nano_tm"),
        run_prefix="tm0",
    )
    main(defaults=defaults)