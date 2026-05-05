from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


T_LIST = [6,8,11,14]
L_LIST = [5,15]
SEEDS = [0,1,2]
XOR_SPECS = ["first_last_xor"]
CVX_BIAS_GRID = [0.0]
K_PARALLEL_LIST = [2]

# P_rec and P_last must be divisible by every K_parallel in K_PARALLEL_LIST.
COMMON = {
    "dataset": "dfa",
    "P_rec": 500,
    "P_last": 1000,
    "n_train": 2000,
    "n_val": 2000,
    "n_test": 4000,
    "cvx_method": "cvx",
    "loss_type": "hinge",
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DFA XOR T/L sweeps via simple_testing (CVX + STE in one process when --side both)."
    )
    parser.add_argument(
        "--side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Which simple_testing simple_side to run. 'both' runs CVX and STE in one invocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_dir = Path(__file__).resolve().parent
    python_exe = "python3"

    if args.side == "both":
        simple_side = "both"
    elif args.side == "cvx_only":
        simple_side = "cvx_only"
    else:
        simple_side = "ste_only"

    for k_par in K_PARALLEL_LIST:
        for dfa_spec in XOR_SPECS:
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
                            simple_side,
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
                            "--K_parallel",
                            str(k_par),
                            "--cvx_epochs",
                            "200",
                            "--ste_epochs",
                            "200",
                            "--last_layer_readout",
                            "membrane",
                        ]
                        if args.side in ("both", "cvx_only"):
                            cmd.extend(["--bias_grid", *[str(b) for b in CVX_BIAS_GRID]])
                        run_cmd(cmd, cwd=atomic_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
