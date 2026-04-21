from __future__ import annotations

"""
Ablation over K_parallel (``K``) with explicit width accounting.

Parallel SNN/CVX use total concat widths ``P_rec`` and ``P_last``. Each of ``K`` branches
has hidden width ``P_sub_rec = P_rec // K`` and last hidden ``P_sub_last = P_last // K``,
so **total** recurrent width is ``P_sub_rec * K = P_rec`` (and similarly for the readout
dimension into the classifier).

**Modes**

- ``total`` (default): ``P_rec`` and ``P_last`` are **fixed** across the ablation; only ``K``
  changes, so ``P_sub = P_rec / K`` shrinks as ``K`` increases — **constant total width**
  (constant parameter budget in the usual concat sense).

- ``sub``: ``P_sub_rec`` and ``P_sub_last`` are **fixed**; set ``P_rec = P_sub_rec * K``,
  ``P_last = P_sub_last * K`` — total concat width **grows** linearly with ``K``.

Defaults: ``--k_list`` is ``5 10 20 50 100 200``; ``--cvx_method cvx`` (convex solver). Override ``--simple_side`` if you need STE or SGD.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_k_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("K list is empty.")
    return out


def build_cmd(
    *,
    python_exe: str,
    k: int,
    p_rec: int,
    p_last: int,
    args: argparse.Namespace,
) -> list[str]:
    simple_side = str(getattr(args, "simple_side", "both")).strip()
    dataset = str(getattr(args, "dataset", "")).strip()
    cvx_method = str(getattr(args, "cvx_method", "cvx")).strip()
    loss_type = str(getattr(args, "loss_type", "ce")).strip()
    last_readout = str(getattr(args, "last_layer_readout", "membrane")).strip()
    cmd = [
        python_exe,
        "-m",
        "simple_testing",
        "--mode",
        "fine_tune",
        "--simple_side",
        simple_side,
        "--dataset",
        dataset,
        "--cvx_method",
        cvx_method,
        "--loss_type",
        loss_type,
        "--seed",
        str(args.seed),
        "--T",
        str(args.T),
        "--n_train",
        str(args.n_train),
        "--n_val",
        str(args.n_val),
        "--n_test",
        str(args.n_test),
        "--L",
        str(args.L),
        "--P_rec",
        str(p_rec),
        "--P_last",
        str(p_last),
        "--K_parallel",
        str(k),
        "--cvx_epochs",
        str(args.cvx_epochs),
        "--ste_epochs",
        str(args.ste_epochs),
        "--last_layer_readout",
        last_readout,
    ]
    if args.bias_grid is not None:
        cmd.extend(["--bias_grid", *[str(float(x)) for x in args.bias_grid]])
    ds = dataset
    if ds.startswith("dfa:"):
        pass
    elif ds == "dfa" and getattr(args, "dfa_spec", ""):
        cmd.extend(["--dfa_spec", str(args.dfa_spec).strip()])
    if dataset == "arithmetic_seq":
        cmd.extend(
            [
                "--arith_op",
                args.arith_op,
                "--arith_base",
                str(args.arith_base),
                "--n_digits",
                str(args.n_digits),
            ]
        )
    return cmd


def validate_total_fixed(k: int, p_rec: int, p_last: int) -> tuple[int, int]:
    if p_rec % k != 0 or p_last % k != 0:
        raise ValueError(
            f"total-fixed mode: need P_rec % K == 0 and P_last % K == 0; "
            f"got K={k}, P_rec={p_rec}, P_last={p_last}."
        )
    return p_rec // k, p_last // k


def validate_sub_fixed(k: int, p_sub_rec: int, p_sub_last: int) -> tuple[int, int]:
    return p_sub_rec * k, p_sub_last * k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--width_mode",
        choices=("total", "sub"),
        default="total",
        help="total: fixed P_rec,P_last; sub: fixed per-branch P_sub, scale totals as P_sub*K.",
    )
    p.add_argument(
        "--k_list",
        type=str,
        default="5 10 20 50 100 200",
        help="Whitespace- or comma-separated K values (default: 5 10 20 50 100 200).",
    )
    p.add_argument(
        "--simple_side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Default cvx_only: ablation is CVX-only (method=cvx). Use both/ste_only if needed.",
    )
    p.add_argument("--dataset", default="mnist_seq")
    p.add_argument("--dfa_spec", default="", help="For dataset=dfa.")
    p.add_argument("--arith_op", default="add")
    p.add_argument("--arith_base", type=int, default=2)
    p.add_argument("--n_digits", type=int, default=8)
    p.add_argument("--cvx_method", default="cvx", choices=("cvx", "sgd"), help="Convex solver (default cvx) or CVX-SGD.")
    p.add_argument("--loss_type", default="ce")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=28)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--n_train", type=int, default=6000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--cvx_epochs", type=int, default=200)
    p.add_argument("--ste_epochs", type=int, default=200)
    p.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="spike")
    p.add_argument("--bias_grid", type=float, nargs="*", default=None)
    p.add_argument(
        "--p_rec",
        type=int,
        default=1000,
        help="total-fixed: total P_rec (= P_sub_rec * K). Ignored for --width_mode sub if --p_sub_rec set.",
    )
    p.add_argument(
        "--p_last",
        type=int,
        default=1200,
        help="total-fixed: total P_last (= P_sub_last * K). Ignored for --width_mode sub if --p_sub_last set.",
    )
    p.add_argument("--p_sub_rec", type=int, default=200, help="sub-fixed: per-branch recurrent width.")
    p.add_argument("--p_sub_last", type=int, default=240, help="sub-fixed: per-branch last hidden width.")
    p.add_argument("--dry_run", action="store_true", help="Print K, P_sub, P_rec, commands; do not execute.")
    return p.parse_args()


def main() -> None:
    ns = parse_args()
    ks = _parse_k_list(ns.k_list)
    atomic_dir = Path(__file__).resolve().parent
    py = sys.executable if sys.executable else "python3"

    print("# K_parallel ablation — width: P_rec = P_sub_rec * K, P_last = P_sub_last * K (exact integers).")
    print(f"# width_mode={ns.width_mode}")

    for k in ks:
        if int(k) <= 0:
            raise ValueError(f"Invalid K_parallel={k}.")

        if ns.width_mode == "total":
            p_rec, p_last = int(ns.p_rec), int(ns.p_last)
            sub_r, sub_l = validate_total_fixed(k, p_rec, p_last)
        else:
            p_rec, p_last = validate_sub_fixed(k, int(ns.p_sub_rec), int(ns.p_sub_last))
            sub_r, sub_l = int(ns.p_sub_rec), int(ns.p_sub_last)

        print(
            f"\n# K={k}  P_sub_rec={sub_r}  P_sub_last={sub_l}  "
            f"P_rec={p_rec}=P_sub_rec*K  P_last={p_last}=P_sub_last*K"
        )

        cmd = build_cmd(python_exe=py, k=k, p_rec=p_rec, p_last=p_last, args=ns)

        print(" ".join(cmd))
        if not ns.dry_run:
            print("=" * 100, flush=True)
            subprocess.run(cmd, cwd=atomic_dir, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
