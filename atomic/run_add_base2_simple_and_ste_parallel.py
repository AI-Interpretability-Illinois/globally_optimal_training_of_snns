from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


N_TRAIN_LIST = [2304, 4096, 8192, 16384]
SEEDS = [0]

# Shared config requested by user.
COMMON = {
    "dataset": "arithmetic_seq",
    "arith_op": "add",
    "arith_base": 2,
    "n_digits": 8,
    "L": 3,
    "P_last": 4608,
    "P_rec": 2304,
    "n_val": 256,
    "n_test": 2048,
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run arithmetic add base2 sweeps: CVX and/or STE via simple_testing (sequential; no parallel_sweep_helper).",
    )
    parser.add_argument(
        "--side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Choose which sweep block to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_dir = Path(__file__).resolve().parent
    python_exe = "python3"

    # 1) CVX-only runs via simple_testing (3 seeds per n_train).
    if args.side in ("both", "cvx_only"):
        for n_train in N_TRAIN_LIST:
            for seed in SEEDS:
                cmd = [
                    python_exe,
                    "-m",
                    "simple_testing",
                    "--mode",
                    "simple",
                    "--simple_side",
                    "cvx_only",
                    "--dataset",
                    COMMON["dataset"],
                    "--loss_type",
                    "hinge_ovr",
                    "--cvx_method",
                    "cvx",
                    "--seed",
                    str(seed),
                    "--n_train",
                    str(n_train),
                    "--n_val",
                    str(COMMON["n_val"]),
                    "--n_test",
                    str(COMMON["n_test"]),
                    "--L",
                    str(COMMON["L"]),
                    "--P_last",
                    str(COMMON["P_last"]),
                    "--P_rec",
                    str(COMMON["P_rec"]),
                    "--arith_op",
                    COMMON["arith_op"],
                    "--arith_base",
                    str(COMMON["arith_base"]),
                    "--n_digits",
                    str(COMMON["n_digits"]),
                    "--cvx_epochs",
                    "200",
                ]
                run_cmd(cmd, cwd=atomic_dir)

    # 2) STE-only runs via simple_testing (grid over lr/beta inside _run_simple_mode; one process per seed/n_train).
    if args.side in ("both", "ste_only"):
        for n_train in N_TRAIN_LIST:
            for seed in SEEDS:
                cmd = [
                    python_exe,
                    "-m",
                    "simple_testing",
                    "--mode",
                    "simple",
                    "--simple_side",
                    "ste_only",
                    "--dataset",
                    COMMON["dataset"],
                    "--loss_type",
                    "hinge_ovr",
                    "--cvx_method",
                    "cvx",
                    "--seed",
                    str(seed),
                    "--n_train",
                    str(n_train),
                    "--n_val",
                    str(COMMON["n_val"]),
                    "--n_test",
                    str(COMMON["n_test"]),
                    "--L",
                    str(COMMON["L"]),
                    "--P_last",
                    str(COMMON["P_last"]),
                    "--P_rec",
                    str(COMMON["P_rec"]),
                    "--arith_op",
                    COMMON["arith_op"],
                    "--arith_base",
                    str(COMMON["arith_base"]),
                    "--n_digits",
                    str(COMMON["n_digits"]),
                    "--ste_epochs",
                    "200",
                ]
                run_cmd(cmd, cwd=atomic_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
