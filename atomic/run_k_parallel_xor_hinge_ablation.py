from __future__ import annotations

"""
Binary hinge + CVX K_parallel ablation on the first/last XOR DFA, matched to the arithmetic addition setup:

  * ``T = 8``
  * ``--dataset dfa:first_last_xor`` (exact automaton); ``loss_type = hinge`` (binary 0/1; DFA tasks use hinge)
  * ``cvx_method = cvx``, ``simple_side = cvx_only``
  * ``width_mode = sub``: fixed ``p_sub_rec`` / ``p_sub_last``, ``P_rec = p_sub_rec * K``,
    ``P_last = p_sub_last * K`` (same progression as addition ablation)
  * ``n_train`` schedule: same as ``run_add_base2_simple_and_ste_parallel`` / addition K-sweeps
    (defaults: 2304, 4096, 8192, 16384)

``dfa`` alone refers to any DFA from ``--dfa_spec``; ``dfa:first_last_xor`` pins this run to that construction.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from run_k_parallel_ablation import _parse_k_list, build_cmd, validate_sub_fixed, validate_total_fixed


# Match arithmetic hinge K ablation (user runs): sub widths and K grid.
DEFAULT_K_LIST = "2,16,32,64"
DEFAULT_P_REC = 128
DEFAULT_P_LAST = 512

# Same n_train progression as run_add_base2_simple_and_ste_parallel.py
DEFAULT_N_TRAIN_LIST = "2304"
N_VAL = 256
N_TEST = 2048

T_XOR = 11
L_XOR = 3
# Encodes exact DFA in one flag (parsed by simple_testing); binary task uses hinge.
DATASET_XOR = "dfa:first_last_xor"


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
    p.add_argument("--T", type=int, default=T_XOR)
    p.add_argument("--L", type=int, default=L_XOR)
    p.add_argument("--n_val", type=int, default=N_VAL)
    p.add_argument("--n_test", type=int, default=N_TEST)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cvx_epochs", type=int, default=200)
    p.add_argument("--ste_epochs", type=int, default=200)
    p.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="spike")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _ns_template(base: argparse.Namespace, *, n_train: int) -> argparse.Namespace:
    """Namespace expected by ``build_cmd`` (same shape as run_k_parallel_ablation.parse_args())."""
    return argparse.Namespace(
        simple_side="ste_only",
        dataset=DATASET_XOR,
        cvx_method="cvx",
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
        dfa_spec="first_last_xor",
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

    print("# DFA XOR | hinge + cvx | T=8 (default) | --dataset dfa:first_last_xor")
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
            cmd = build_cmd(python_exe=py, k=kk, p_rec=p_rec, p_last=p_last, args=args_ns)
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
