from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


T_LIST = [7, 14, 28, 56, 196, 784]
L_LIST = [3, 5, 10, 15, 20, 30, 50, 80, 120]
SEEDS = [0, 1, 2]
PARITY_SPECS = ["parity_2", "parity_3", "parity_5", "parity_7"]
CVX_BIAS_GRID = [0.0]

# Placeholder widths/sample sizes, aligned with MNIST sweep style.
COMMON = {
    "dataset": "dfa",
    "P_rec": 10000,
    "P_last": 12000,
    "n_train": 10000,  # placeholder: change later as needed
    "n_val": 2000,     # placeholder
    "n_test": 2000,    # placeholder: change later as needed
    "cvx_method": "sgd",
    "loss_type": "hinge",
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFA parity T/L sweeps for CVX/SNN.")
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
        for dfa_spec in PARITY_SPECS:
            for l in L_LIST:
                for t in T_LIST:
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
                            "--dfa_spec",
                            dfa_spec,
                            "--cvx_method",
                            COMMON["cvx_method"],
                            "--loss_type",
                            COMMON["loss_type"],
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

    if args.side in ("both", "ste_only"):
        for dfa_spec in PARITY_SPECS:
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
