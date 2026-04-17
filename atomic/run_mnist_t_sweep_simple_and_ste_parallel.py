from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


T_LIST = [2, 4, 7, 14, 28, 56, 196, 784]
L_LIST = [3, 5, 10, 15, 20, 30, 50, 80, 120]
SEEDS = [0, 1, 2]
CVX_BIAS_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Requested fixed MNIST configuration.
COMMON = {
    "dataset": "mnist_seq",
    "P_rec": 10000,
    "P_last": 12000,
    "n_train": 10000,
    "n_val": 2000,
    "n_test": 2000,
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST T-sweeps for CVX/SNN.")
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

    # 1) CVX-only runs via simple_testing (3 seeds per (T, L)).
    if args.side in ("both", "cvx_only"):
        for t in T_LIST:
            for l in L_LIST:
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
                        "--cvx_method",
                        "sgd",
                        "--seed",
                        str(seed),
                        "--T",
                        str(t),
                        "--n_train",
                        str(COMMON["n_train"]),
                        "--n_val",
                        str(COMMON["n_val"]),
                        "--n_test",
                        str(COMMON["n_test"]),
                        "--L",
                        str(l),
                        "--P_last",
                        str(COMMON["P_last"]),
                        "--P_rec",
                        str(COMMON["P_rec"]),
                        "--cvx_epochs",
                        "200",
                        "--bias_grid",
                        *[str(b) for b in CVX_BIAS_GRID],
                    ]
                    run_cmd(cmd, cwd=atomic_dir)

    # 2) SNN-only sweep via parallel helper (parallel over seeds for each (T, L)).
    if args.side in ("both", "ste_only"):
        for t in T_LIST:
            for l in L_LIST:
                cmd = [
                    python_exe,
                    "-m",
                    "parallel_sweep_helper",
                    "--pipeline_mode",
                    "simple",
                    "--simple_side",
                    "ste_only",
                    "--dataset",
                    COMMON["dataset"],
                    "--cvx_method",
                    "sgd",
                    "--seeds",
                    *[str(s) for s in SEEDS],
                    "--T",
                    str(t),
                    "--n_train",
                    str(COMMON["n_train"]),
                    "--n_val",
                    str(COMMON["n_val"]),
                    "--n_test",
                    str(COMMON["n_test"]),
                    "--L",
                    str(l),
                    "--P_last",
                    str(COMMON["P_last"]),
                    "--P_rec",
                    str(COMMON["P_rec"]),
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
