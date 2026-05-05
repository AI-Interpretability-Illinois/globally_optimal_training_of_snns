from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


T_LIST = [2,14,28,56]
L_LIST = [3,5,10,15]
SEEDS = [0,1]
CVX_BIAS_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
K_PARALLEL_LIST = [2]

# Requested fixed MNIST configuration.
# P_rec and P_last must be divisible by every K_parallel in K_PARALLEL_LIST.
COMMON = {
    "dataset": "mnist_seq",
    "P_rec": 4000,
    "P_last": 7000,
    "n_train": 6000,
    "n_val": 2000,
    "n_test": 2000,
}


def _child_env(*, blas_threads: int | None) -> dict[str, str] | None:
    """If set, each child uses this many BLAS threads (use without --parallel for fastest per-job time)."""
    if blas_threads is None:
        return None
    if blas_threads < 1:
        raise ValueError(f"blas_threads must be >= 1, got {blas_threads}.")
    env = os.environ.copy()
    s = str(blas_threads)
    env["OMP_NUM_THREADS"] = s
    env["OPENBLAS_NUM_THREADS"] = s
    env["MKL_NUM_THREADS"] = s
    env["VECLIB_MAXIMUM_THREADS"] = s
    return env


def _build_cmd(
    *,
    simple_side: str,
    k_par: int,
    l: int,
    t: int,
    seed: int,
    cvx_ovr_workers: int,
    include_bias_grid: bool,
) -> list[str]:
    cmd = [
        "python3",
        "-m",
        "simple_testing",
        "--mode", "simple",
        "--simple_side", simple_side,
        "--dataset", COMMON["dataset"],
        "--cvx_method", "cvx",
        "--loss_type", "ce",
        "--seed", str(seed),
        "--T", str(t),
        "--n_train", str(COMMON["n_train"]),
        "--n_val", str(COMMON["n_val"]),
        "--n_test", str(COMMON["n_test"]),
        "--L", str(l),
        "--P_last", str(COMMON["P_last"]),
        "--P_rec", str(COMMON["P_rec"]),
        "--K_parallel", str(k_par),
        "--cvx_epochs", "200",
        "--ste_epochs", "200",
        "--last_layer_readout", "membrane",
        "--cvx_ovr_workers", str(cvx_ovr_workers),
    ]
    if include_bias_grid:
        cmd.extend(["--bias_grid", *[str(b) for b in CVX_BIAS_GRID]])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST T/L sweeps via simple_testing (CVX + STE in one process when --side both)."
    )
    parser.add_argument(
        "--side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Which simple_testing simple_side to run.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Launch all (K, L, T, seed) combinations as concurrent subprocesses (max throughput, "
            "slower per job). Omit this to run one simple_testing at a time (fastest per job); "
            "pair with --blas_threads."
        ),
    )
    parser.add_argument(
        "--blas_threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Set OMP/MKL/OpenBLAS/Accelerate thread count for each child process. "
            "Use with serial runs (no --parallel), e.g. --blas_threads 32 on a 64-core box. "
            "Default: inherit from your shell (unset this flag)."
        ),
    )
    parser.add_argument(
        "--cvx_ovr_workers",
        type=int,
        default=1,
        help=(
            "Number of parallel classwise CVXPY solves inside each simple_testing run "
            "(passed through as --cvx_ovr_workers to simple_testing). "
            "For --parallel, each subprocess already runs concurrently, so set this "
            "to total_cores // n_parallel_jobs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_dir = Path(__file__).resolve().parent

    include_bias = args.side in ("both", "cvx_only")

    all_cmds = [
        _build_cmd(
            simple_side="both" if args.side == "both" else args.side,
            k_par=k_par,
            l=l,
            t=t,
            seed=seed,
            cvx_ovr_workers=int(args.cvx_ovr_workers),
            include_bias_grid=include_bias,
        )
        for k_par in K_PARALLEL_LIST
        for t in T_LIST
        for l in L_LIST
        for seed in SEEDS
    ]

    child_env = _child_env(blas_threads=args.blas_threads)
    if not args.parallel:
        for cmd in all_cmds:
            print("\n" + "=" * 120)
            print(" ".join(cmd))
            print("=" * 120)
            subprocess.run(cmd, cwd=atomic_dir, env=child_env, check=True)
    else:
        procs: list[tuple[list[str], subprocess.Popen[bytes]]] = []
        for cmd in all_cmds:
            print("\n[launch] " + " ".join(cmd))
            proc = subprocess.Popen(cmd, cwd=atomic_dir, env=child_env)
            procs.append((cmd, proc))

        failed: list[tuple[list[str], int]] = []
        for cmd, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failed.append((cmd, rc))

        if failed:
            for cmd, rc in failed:
                print(f"\n[FAILED rc={rc}] {' '.join(cmd)}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"\n[done] all {len(procs)} jobs completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
