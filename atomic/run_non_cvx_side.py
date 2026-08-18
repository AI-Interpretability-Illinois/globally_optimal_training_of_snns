"""Non-CVX side of the XOR / MNIST / addition split: SG and LSM ridge.

Runs, in order:

1. ``run_baselines_xor_mnist.py --side non_cvx`` — surrogate-gradient SNN; writes
   ``seed*_sg.npz`` for later SG-CVX.
2. ``run_lsm_xor_mnist.py --side non_cvx`` — criticality-tuned reservoir + ridge;
   writes ``seed*_lsm.npz`` for later R-CVX.

Checkpoints share ``--ckpt_dir`` (default ``<out_root>/ckpts``). Rsync that
directory onto the CVX machine, then run the matching ``--side cvx`` jobs.

Default ``--tasks`` is the full preset list (xor, mnist, add_b2/3/5/7/10).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    from run_lsm_xor_mnist import TASK_NAMES
else:
    from .run_lsm_xor_mnist import TASK_NAMES


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_root", type=str, default="sweep_results/non_cvx_side")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Shared SG + LSM checkpoints. Default: <out_root>/ckpts.",
    )
    ap.add_argument("--sg_epochs", type=int, default=None, help="Override SG epochs. Default: baselines script default.")
    return ap.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = _parse_args()
    atomic_dir = Path(__file__).resolve().parent
    py = sys.executable
    out_root = Path(args.out_root).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else (out_root / "ckpts")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"[non-cvx] ckpt_dir={ckpt_dir} tasks={list(args.tasks)} seeds={list(args.seeds)}", flush=True)
    _run(sg_cmd, cwd=atomic_dir)
    _run(lsm_cmd, cwd=atomic_dir)
    print(f"[non-cvx] done. rsync {ckpt_dir} onto the CVX machine.", flush=True)


if __name__ == "__main__":
    main()
