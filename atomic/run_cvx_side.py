"""CVX side: CVX + SG-CVX + R-CVX (xor/mnist) and AR two-head CVX (addition).

Loads checkpoints written by ``run_non_cvx_side.py``. Default readout solver is
``cvx_lite``.

Runs, in order:

1. ``run_baselines_xor_mnist.py --side cvx --cvx_method cvx_lite`` — Gaussian CVX
   and SG-CVX on xor/mnist.
2. ``run_lsm_xor_mnist.py --side cvx --cvx_method cvx_lite`` — R-CVX on xor/mnist.
3. ``run_arithmetic_add_carry_autoregressive_rollout_matched.py --side cvx
   --cvx_method cvx_lite`` for every addition base — Gaussian two-head CVX,
   STE-CVX, and R-CVX, all evaluated with autoregressive rollout. Default
   ``n_train=10000``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import SUPPORTED_BASES
else:
    from .data_loaders.arithmetic_data_loader import SUPPORTED_BASES

_CLASSIFICATION_TASKS = ("xor", "mnist")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tasks",
        nargs="*",
        choices=_CLASSIFICATION_TASKS,
        default=list(_CLASSIFICATION_TASKS),
        help="xor/mnist CVX. Pass --tasks with no values to skip (AR only).",
    )
    ap.add_argument(
        "--only_arith",
        action="store_true",
        help="Skip xor/mnist. Only run carry-AR Gaussian CVX + STE-CVX + R-CVX.",
    )
    ap.add_argument("--arith_bases", type=int, nargs="*", default=list(SUPPORTED_BASES))
    ap.add_argument("--arith_L", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--P_rec", type=int, default=256)
    ap.add_argument("--P_last", type=int, default=512)
    ap.add_argument(
        "--lambda_carry_grid",
        type=float,
        nargs="*",
        default=[0.125, 10.0],
        help="Must match the STE ckpt lambda tags written by the non-CVX side.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, nargs="+", default=None, help="Override xor/mnist preset T_list.")
    ap.add_argument("--L", type=int, nargs="+", default=None, help="Override xor/mnist preset L_list.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_root", type=str, default="sweep_results/cvx_side")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory rsynced from the non-CVX machine (SG / LSM / STE-AR npz files).",
    )
    ap.add_argument("--cvx_method", choices=("cvx", "cvx_lite"), default="cvx_lite")
    ap.add_argument("--lite_max_iter", type=int, default=5000)
    ap.add_argument("--lite_tol", type=float, default=1e-6)
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
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt_dir does not exist: {ckpt_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[cvx] ckpt_dir={ckpt_dir} method={args.cvx_method} tasks={list(args.tasks)} "
        f"T={args.T} L={args.L} arith_bases={list(args.arith_bases)} n_train={int(args.n_train)}",
        flush=True,
    )

    if len(args.tasks) > 0:
        shared = [
            py,
            "--side",
            "cvx",
            "--cvx_method",
            str(args.cvx_method),
            "--lite_max_iter",
            str(int(args.lite_max_iter)),
            "--lite_tol",
            str(float(args.lite_tol)),
            "--tasks",
            *list(args.tasks),
            "--seeds",
            *[str(int(s)) for s in args.seeds],
            "--ckpt_dir",
            str(ckpt_dir),
        ]
        if args.debug:
            shared.append("--debug")
        if args.T is not None:
            shared.extend(["--T", *[str(int(t)) for t in args.T]])
        if args.L is not None:
            shared.extend(["--L", *[str(int(x)) for x in args.L]])
        _run(
            [
                shared[0],
                str(atomic_dir / "run_baselines_xor_mnist.py"),
                *shared[1:],
                "--out_root",
                str(out_root / "baselines"),
            ],
            cwd=atomic_dir,
        )
        _run(
            [
                shared[0],
                str(atomic_dir / "run_lsm_xor_mnist.py"),
                *shared[1:],
                "--out_root",
                str(out_root / "lsm"),
            ],
            cwd=atomic_dir,
        )

    for base in args.arith_bases:
        for L in args.arith_L:
            ar_cmd = [
                py,
                str(atomic_dir / "run_arithmetic_add_carry_autoregressive_rollout_matched.py"),
                "--side",
                "cvx",
                "--cvx_method",
                str(args.cvx_method),
                "--lite_max_iter",
                str(int(args.lite_max_iter)),
                "--lite_tol",
                str(float(args.lite_tol)),
                "--arith_base",
                str(int(base)),
                "--L",
                str(int(L)),
                "--n_train",
                str(int(args.n_train)),
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
                "--lambda_carry_grid",
                *[str(float(x)) for x in args.lambda_carry_grid],
            ]
            if args.debug:
                ar_cmd.append("--debug")
            _run(ar_cmd, cwd=atomic_dir)

    print(f"[cvx] done. wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
