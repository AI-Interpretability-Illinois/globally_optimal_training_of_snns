from __future__ import annotations

"""
Binary hinge + CVX K_parallel ablation on **Tomita 3** (DFA benchmark).

  * Default ``--dataset dfa:tomita_3``; ``loss_type = hinge`` (binary accept/reject labels).
  * ``cvx_method = cvx``; width accounting matches ``run_k_parallel_ablation`` (total vs sub).
  * Runs ``python -m simple_testing --mode simple`` (CVX vs STE sweep), **not** ``fine_tune``.
  * Default ``T = 20`` (override with ``--T``); ``L = 3``.
  * ``n_train`` schedule defaults to a single block (2304); extend via ``--n_train_list``.

Use ``dfa:<spec>`` to swap automata without editing this file; plain ``dfa`` would use ``--dfa_spec``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from run_k_parallel_ablation import _parse_k_list, validate_sub_fixed, validate_total_fixed


# Match arithmetic hinge K ablation (user runs): sub widths and K grid.
DEFAULT_K_LIST = "2,16,32,64"
DEFAULT_P_REC = 1024
DEFAULT_P_LAST = 2048

# Same n_train progression as run_add_base2_simple_and_ste_parallel.py
DEFAULT_N_TRAIN_LIST = "8192,16384"
N_VAL = 256
N_TEST = 2048

# Defaults tuned for Tomita 3 (harder than Tomita 1); override with --T / --L.
T_DEFAULT = 20
L_DEFAULT = 3
# Parsed by simple_testing as dfa_spec=tomita_3 after ``dfa:`` split.
DATASET_DFA = "dfa:tomita_6"
DFA_SPEC = "tomita_6"


def _build_simple_testing_cmd(
    *,
    python_exe: str,
    k: int,
    p_rec: int,
    p_last: int,
    args: argparse.Namespace,
) -> list[str]:
    """Same argv shape as ``run_k_parallel_ablation.build_cmd``, but ``--mode simple``."""
    simple_side = str(getattr(args, "simple_side", "both")).strip()
    dataset = str(getattr(args, "dataset", "")).strip()
    cvx_method = str(getattr(args, "cvx_method", "cvx")).strip()
    loss_type = str(getattr(args, "loss_type", "ce")).strip()
    last_readout = str(getattr(args, "last_layer_readout", "membrane")).strip()
    optimizer_name = str(getattr(args, "optimizer_name", "adam")).strip()
    cmd = [
        python_exe,
        "-m",
        "simple_testing",
        "--mode",
        "simple",
        "--simple_side",
        simple_side,
        "--dataset",
        dataset,
        "--cvx_method",
        cvx_method,
        "--loss_type",
        loss_type,
        "--optimizer_name",
        optimizer_name,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k_list", type=str, default=DEFAULT_K_LIST, help="Comma/space K_parallel values.")
    p.add_argument(
        "--n_train_list",
        type=str,
        default=DEFAULT_N_TRAIN_LIST,
        help="Comma/space n_train values (same schedule as addition baselines).",
    )
    p.add_argument("--width_mode", choices=("total", "sub"), default="total")
    p.add_argument("--p_rec", type=int, default=DEFAULT_P_REC, help="total-fixed: total P_rec.")
    p.add_argument("--p_last", type=int, default=DEFAULT_P_LAST, help="total-fixed: total P_last.")
    # p.add_argument("--p_sub_rec", type=int, default=DEFAULT_P_SUB_REC)
    # p.add_argument("--p_sub_last", type=int, default=DEFAULT_P_SUB_LAST)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--L", type=int, default=L_DEFAULT)
    p.add_argument("--n_val", type=int, default=N_VAL)
    p.add_argument("--n_test", type=int, default=N_TEST)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cvx_epochs", type=int, default=200)
    p.add_argument("--ste_epochs", type=int, default=200)
    p.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="spike")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _ns_template(base: argparse.Namespace, *, n_train: int) -> argparse.Namespace:
    """Namespace expected by ``_build_simple_testing_cmd`` (STE optimizer for simple mode)."""
    return argparse.Namespace(
        simple_side="both",
        dataset=DATASET_DFA,
        cvx_method="cvx",
        optimizer_name="adam",
        loss_type="hinge",
        seed=base.seed,
        T=base.T,
        n_train=n_train,
        n_val=base.n_val,
        n_test=base.n_test,
        L=base.L,
        cvx_epochs=base.cvx_epochs,
        ste_epochs=base.ste_epochs,
        last_layer_readout=base.last_layer_readout,
        bias_grid=None,
        dfa_spec=DFA_SPEC,
        arith_op="add",
        arith_base=2,
        n_digits=8,
    )


def main() -> None:
    cfg = parse_args()
    ks = _parse_k_list(cfg.k_list)
    n_trains = _parse_k_list(cfg.n_train_list)  # reuse parser: integers only
    if any(int(x) <= 0 for x in n_trains):
        raise ValueError("n_train values must be positive.")

    atomic_dir = Path(__file__).resolve().parent
    py = sys.executable if sys.executable else "python3"

    print("# DFA Tomita 3 | hinge | simple_testing --mode simple | dfa:tomita_3")
    print(f"# width_mode={cfg.width_mode}")
    print(f"# K_list={ks}  n_train_list={n_trains}")
    if cfg.width_mode == "sub":
        print(f"# p_sub_rec={cfg.p_sub_rec}  p_sub_last={cfg.p_sub_last}")

    for n_train in n_trains:
        nt = int(n_train)
        print(f"\n######## n_train={nt} ########")
        for k in ks:
            kk = int(k)
            if kk <= 0:
                raise ValueError(f"Invalid K_parallel={kk}.")

            if cfg.width_mode == "total":
                p_rec, p_last = int(cfg.p_rec), int(cfg.p_last)
                sub_r, sub_l = validate_total_fixed(kk, p_rec, p_last)
            else:
                p_rec, p_last = validate_sub_fixed(kk, int(cfg.p_sub_rec), int(cfg.p_sub_last))
                sub_r, sub_l = int(cfg.p_sub_rec), int(cfg.p_sub_last)

            print(
                f"\n# K={kk}  n_train={nt}  P_sub_rec={sub_r}  P_sub_last={sub_l}  "
                f"P_rec={p_rec}  P_last={p_last}"
            )

            args_ns = _ns_template(cfg, n_train=nt)
            cmd = _build_simple_testing_cmd(python_exe=py, k=kk, p_rec=p_rec, p_last=p_last, args=args_ns)
            print(" ".join(cmd))
            if not cfg.dry_run:
                print("=" * 100, flush=True)
                subprocess.run(cmd, cwd=atomic_dir, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
