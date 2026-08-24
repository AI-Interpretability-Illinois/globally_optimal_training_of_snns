"""Non-CVX side: SG + LSM ridge (xor/mnist) and STE + LSM AR addition.

Runs, in order:

1. ``run_baselines_xor_mnist.py --side non_cvx`` on xor/mnist — surrogate-gradient;
   writes ``*_sg.npz``.
2. ``run_lsm_xor_mnist.py --side non_cvx`` on xor/mnist — criticality + ridge;
   writes ``*_lsm.npz``.
3. ``run_arithmetic_add_carry_autoregressive_rollout_matched.py --side non_cvx``
   for every addition base — carry-augmented STE pretrain, optional STE-from-STE
   finetune, and LSM ridge, all AR-eval'd. Hidden-only checkpoints go to
   ``--ckpt_dir`` as ``*_ste_*.npz``, ``*_ste_ft_from_ste_*.npz``, ``*_lsm.npz``.

Rsync ``--ckpt_dir`` (default ``<out_root>/ckpts``) onto the CVX machine, then
run ``run_cvx_side.py``. Do not pass ``--side all`` or invoke the hybrid five-stage
bench from this launcher: those run Gaussian CVX / SG-CVX / R-CVX.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import SUPPORTED_BASES
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
else:
    from .data_loaders.arithmetic_data_loader import SUPPORTED_BASES
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT

_CLASSIFICATION_TASKS = ("xor", "mnist")
_MATCHED_SG_LAMBDA_CARRY = (0.125, 10.0)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tasks",
        nargs="*",
        choices=_CLASSIFICATION_TASKS,
        default=list(_CLASSIFICATION_TASKS),
        help="xor/mnist SG+LSM. Pass --tasks with no values to skip (AR only).",
    )
    ap.add_argument(
        "--only_arith",
        action="store_true",
        help="Skip xor/mnist. Only run carry-AR STE (+ optional STE ft) + LSM ridge.",
    )
    ap.add_argument("--arith_bases", type=int, nargs="*", default=list(SUPPORTED_BASES))
    ap.add_argument("--arith_L", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument(
        "--n_train_ft",
        type=int,
        default=10000,
        help="STE-from-STE finetune train size. 0 skips finetune (pretrain + LSM only).",
    )
    ap.add_argument("--finetune_seed_offset", type=int, default=1000)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--P_rec", type=int, default=256)
    ap.add_argument("--P_last", type=int, default=512)
    ap.add_argument(
        "--ste_last_layer_readout",
        type=str,
        default="spike",
        choices=["membrane", "spike"],
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="STE batch size. -1 = full batch (matched previous SG).",
    )
    ap.add_argument("--ste_epochs", type=int, default=100)
    ap.add_argument("--ste_finetune_epochs", type=int, default=100)
    ap.add_argument("--lambda_carry_grid", type=float, nargs="*", default=list(_MATCHED_SG_LAMBDA_CARRY))
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=list(LR_GRID_DEFAULT))
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_root", type=str, default="sweep_results/non_cvx_side")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Shared SG / LSM / STE-AR checkpoints. Default: <out_root>/ckpts.",
    )
    ap.add_argument("--sg_epochs", type=int, default=None)
    return ap.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = _parse_args()
    if bool(args.only_arith):
        args.tasks = []
    for b in args.arith_bases:
        if int(b) not in SUPPORTED_BASES:
            raise ValueError(f"Unsupported arith_base={b}. Supported: {SUPPORTED_BASES}.")
    atomic_dir = Path(__file__).resolve().parent
    py = sys.executable
    out_root = Path(args.out_root).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else (out_root / "ckpts")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[non-cvx] ckpt_dir={ckpt_dir} tasks={list(args.tasks)} arith_bases={list(args.arith_bases)} "
        f"L={list(args.arith_L)} n_train={int(args.n_train)} n_train_ft={int(args.n_train_ft)} "
        f"K={int(args.K_parallel)} P_rec={int(args.P_rec)} P_last={int(args.P_last)} "
        f"batch_size={int(args.batch_size)} seeds={list(args.seeds)} side=non_cvx",
        flush=True,
    )

    if len(args.tasks) > 0:
        shared = [
            py,
            "--side",
            "non_cvx",
            "--tasks",
            *list(args.tasks),
            "--seeds",
            *[str(int(s)) for s in args.seeds],
            "--ckpt_dir",
            str(ckpt_dir),
        ]
        if args.debug:
            shared.append("--debug")
        sg_cmd = [
            shared[0],
            str(atomic_dir / "run_baselines_xor_mnist.py"),
            *shared[1:],
            "--out_root",
            str(out_root / "baselines"),
        ]
        if args.sg_epochs is not None:
            sg_cmd.extend(["--sg_epochs", str(int(args.sg_epochs))])
        lsm_cmd = [
            shared[0],
            str(atomic_dir / "run_lsm_xor_mnist.py"),
            *shared[1:],
            "--out_root",
            str(out_root / "lsm"),
        ]
        _run(sg_cmd, cwd=atomic_dir)
        _run(lsm_cmd, cwd=atomic_dir)

    for base in args.arith_bases:
        for L in args.arith_L:
            ar_cmd = [
                py,
                str(atomic_dir / "run_arithmetic_add_carry_autoregressive_rollout_matched.py"),
                "--side",
                "non_cvx",
                "--arith_base",
                str(int(base)),
                "--L",
                str(int(L)),
                "--n_train",
                str(int(args.n_train)),
                "--n_train_ft",
                str(int(args.n_train_ft)),
                "--finetune_seed_offset",
                str(int(args.finetune_seed_offset)),
                "--seeds",
                *[str(int(s)) for s in args.seeds],
                "--ckpt_dir",
                str(ckpt_dir),
                "--out_root",
                str(out_root / "ar" / f"add_ar_b{int(base)}_L{int(L)}"),
                "--P_rec",
                str(int(args.P_rec)),
                "--P_last",
                str(int(args.P_last)),
                "--K_parallel",
                str(int(args.K_parallel)),
                "--ste_last_layer_readout",
                str(args.ste_last_layer_readout),
                "--lambda_carry_grid",
                *[str(float(x)) for x in args.lambda_carry_grid],
                "--ste_lr_grid",
                *[str(float(x)) for x in args.ste_lr_grid],
                "--ste_beta_grid",
                *[str(float(x)) for x in args.ste_beta_grid],
                "--ste_epochs",
                str(int(args.ste_epochs)),
                "--ste_finetune_epochs",
                str(int(args.ste_finetune_epochs)),
                "--batch_size",
                str(int(args.batch_size)),
            ]
            if args.debug:
                ar_cmd.append("--debug")
            _run(ar_cmd, cwd=atomic_dir)

    print(f"[non-cvx] done. rsync {ckpt_dir} onto the CVX machine.", flush=True)


if __name__ == "__main__":
    main()
