from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


T_LIST = [2, 4, 7, 14, 28, 56, 196, 784]
L_LIST = [3, 5, 10, 15, 20, 30, 50, 80, 120]
SEEDS = [0, 1, 2]
XOR_SPECS = ["first_last_xor"]
CVX_BIAS_GRID = [0.0]
K_PARALLEL_LIST = [50, 100, 500, 1000, 2000]

# Placeholder widths/sample sizes, aligned with MNIST sweep style.
# P_rec and P_last must be divisible by every K_parallel in K_PARALLEL_LIST.
COMMON = {
    "dataset": "dfa",
    "P_rec": 10000,
    "P_last": 12000,
    "n_train": 10000,  # placeholder: change later as needed
    "n_val": 2000,     # placeholder
    "n_test": 2000,    # placeholder: change later as needed
    "cvx_method": "sgd",
    "loss_type": "hinge_ovr",
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFA XOR T/L sweeps for CVX/SNN.")
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

    if args.side in ("both", "cvx_only"):
        for k_par in K_PARALLEL_LIST:
            for dfa_spec in XOR_SPECS:
                for t in T_LIST:
                    for l in L_LIST:
                        cmd = [
                            python_exe,
                            "-m",
                            "parallel_sweep_helper",
                            "--pipeline_mode",
                            "simple",
                            "--simple_side",
                            "cvx_only",
                            "--dataset",
                            COMMON["dataset"],
                            "--dfa_spec",
                            dfa_spec,
                            "--cvx_method",
                            COMMON["cvx_method"],
                            "--loss_type",
                            COMMON["loss_type"],
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
                            "--K_parallel",
                            str(k_par),
                            "--cvx_epochs",
                            "200",
                            "--bias_grid",
                            *[str(b) for b in CVX_BIAS_GRID],
                        ]
                        run_cmd(cmd, cwd=atomic_dir)

    if args.side in ("both", "ste_only"):
        for k_par in K_PARALLEL_LIST:
            for dfa_spec in XOR_SPECS:
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
                            "--dfa_spec",
                            dfa_spec,
                            "--cvx_method",
                            COMMON["cvx_method"],
                            "--loss_type",
                            COMMON["loss_type"],
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
                            "--K_parallel",
                            str(k_par),
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
