#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PYTHON_BIN = Path("/Users/hima_3114/Desktop/Paper_1/experiments/.venv/bin/python3")
TARGET_SCRIPT = Path("/Users/hima_3114/Desktop/Paper_1/experiments/snn_generalized_pt2/cvx_snn_vs_ste_arithmetic_gapcheck.py")
RESULTS_CSV = Path("/Users/hima_3114/Desktop/Paper_1/experiments/cvx_snn_vs_ste_arithmetic_seq_results.csv")

# 5-step sweep:
# - n_train grows from default (512) to 15x (7680)
# - n_test grows from default (256) to 5x (1280)
TRAINS = [512, 2304, 4096, 5888, 7680]
TESTS = [1024]


def main() -> None:
    for i in range(len(TRAINS)):
        n_val = 512  # 20%
        cmd = [
            str(PYTHON_BIN),
            str(TARGET_SCRIPT),
            "--ops",
            "add",
            "--bases",
            "2",
            "--L",
            "3",
            "--n_digits",
            "8",
            "--P_last",
            "256",
            "--P_rec",
            "128",
            "--n_train",
            str(TRAINS[i]),
            "--n_val",
            str(n_val),
            "--n_test",
            str(TESTS[0]),
            "--results_csv",
            str(RESULTS_CSV),
        ]
        print(f"\n[step {i}/5] n_train={TRAINS[i]} n_val={n_val} n_test={TESTS[0]}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
