#!/usr/bin/env python3
"""Hybrid carry SG (pretrain + finetune) matching carry_hybrid_b2_d5_20260423_022747.

Only n_train changes: 10000 on both the pretrain split and the finetune split.
Grid: bases {2,3,5} × L {3,5,10}. Full five-stage pipeline.

From atomic/:

    python run_hybrid_sg_match_ntrain10k.py
    python run_hybrid_sg_match_ntrain10k.py --debug
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_BASES = (2, 3, 5)
_L_VALUES = (3, 5, 10)
_LAMBDA_CARRY = (0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0)
_LR_GRID = (1e-3, 5e-3, 1e-2, 1e-1)
_BETA_GRID = (1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", type=int, nargs="+", default=list(_BASES))
    ap.add_argument("--L", type=int, nargs="+", default=list(_L_VALUES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--out_root", type=str, default="sweep_results/carry_hybrid_ntrain10000")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Tiny n/seeds/epochs/lambda so implementation errors surface quickly.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    atomic_dir = Path(__file__).resolve().parent
    py = sys.executable
    bench = atomic_dir / "run_arithmetic_add_carry_finetune_bench.py"
    out_root = Path(args.out_root).expanduser()
    if not out_root.is_absolute():
        out_root = (atomic_dir / out_root).resolve()
    else:
        out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    n_train = int(args.n_train)
    seeds = [int(s) for s in args.seeds]
    lambdas = [str(x) for x in _LAMBDA_CARRY]
    lr_grid = [str(x) for x in _LR_GRID]
    beta_grid = [str(x) for x in _BETA_GRID]
    pre_epochs = 100
    ft_epochs = 100
    ood = ["10", "20", "50"]
    if bool(args.debug):
        n_train = min(n_train, 32)
        seeds = [seeds[0]]
        lambdas = [lambdas[0]]
        lr_grid = [lr_grid[0]]
        beta_grid = [beta_grid[0]]
        pre_epochs = 5
        ft_epochs = 5
        ood = ["10"]

    print(
        f"[hybrid-sg-match] n_train_pre=n_train_ft={n_train} seeds={seeds} "
        f"bases={list(args.bases)} L={list(args.L)} debug={bool(args.debug)} out_root={out_root}",
        flush=True,
    )
    for base in args.bases:
        if int(base) not in _BASES:
            raise ValueError(f"base={base} not in {_BASES}")
        for L in args.L:
            if int(L) not in _L_VALUES:
                raise ValueError(f"L={L} not in {_L_VALUES}")
            cell = out_root / f"b{int(base)}_L{int(L)}"
            cell.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                str(bench),
                "--run_mode",
                "full",
                "--arith_base",
                str(int(base)),
                "--n_digits",
                "5",
                "--L",
                str(int(L)),
                "--P_rec",
                "256",
                "--P_last",
                "512",
                "--K_parallel",
                "2",
                "--ste_last_layer_readout",
                "spike",
                "--cvx_last_layer_readout",
                "spike",
                "--add_initial_carry",
                "random",
                "--n_train_pre",
                str(n_train),
                "--n_train_ft",
                str(n_train),
                "--n_val_pre",
                "512" if not bool(args.debug) else "16",
                "--n_val_ft",
                "512" if not bool(args.debug) else "16",
                "--n_test",
                "1024" if not bool(args.debug) else "16",
                "--n_test_ood",
                "1024" if not bool(args.debug) else "16",
                "--ood_digits",
                *ood,
                "--seeds",
                *[str(s) for s in seeds],
                "--batch_size",
                "-1",
                "--ste_pretrain_epochs",
                str(pre_epochs),
                "--ste_finetune_epochs",
                str(ft_epochs),
                "--optimizer_name",
                "adam",
                "--lambda_sum",
                "1.0",
                "--lambda_carry_grid",
                *lambdas,
                "--ste_lr_grid",
                *lr_grid,
                "--ste_beta_grid",
                *beta_grid,
                "--cvx_beta_grid",
                *beta_grid,
                "--cvx_bias_grid",
                "0.0",
                "--ste_time_loss",
                "ramp",
                "--cvx_time_loss",
                "ramp",
                "--tf_objective",
                "joint",
                "--out_root",
                str(cell),
            ]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(atomic_dir), check=True)
    print(f"[hybrid-sg-match] done. {out_root}", flush=True)


if __name__ == "__main__":
    main()
