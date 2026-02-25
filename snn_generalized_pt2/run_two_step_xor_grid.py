#!/usr/bin/env python3
"""
Run two_step_xor_seq over a grid of T and L.
Scales P_in, P_rec, P_last by factor = max(T, N) / min(T, N) with N=6 (reference).
Baseline widths at T=6: P_in=2000, P_rec=2000, P_last=5000.

Usage:
  python3 run_two_step_xor_grid.py [--dry-run] [extra args for snn.py]
  e.g. python3 run_two_step_xor_grid.py --log_train
  e.g. python3 run_two_step_xor_grid.py --epochs 50 --seeds 0 1
"""
import argparse
import subprocess
import sys
import os

# Grid
T_LIST = [2, 6, 10, 18, 26, 42, 54, 70, 86, 106]
L_LIST = [2, 3, 5, 9, 15, 23, 33, 45, 59, 77, 97, 119]

# Reference (T ~ 6) baseline widths
T_REF = 6
P_IN_BASE = 2000
P_REC_BASE = 2000
P_LAST_BASE = 5000


def scale_factor(T: int, N: int = T_REF) -> float:
    if T > N:
        return max(T, N) / min(T, N)
    else:
        return 1.0 


def scaled_widths(T: int) -> tuple[int, int, int]:
    f = scale_factor(T)
    return (
        max(1, round(P_IN_BASE * f)),
        max(1, round(P_REC_BASE * f)),
        max(1, round(P_LAST_BASE * f)),
    )


def main():
    parser = argparse.ArgumentParser(description="Run two_step_xor_seq grid; pass extra args for snn.py (e.g. --log_train --epochs 200)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    parser.add_argument("--lr-grid", dest="lr_grid", type=float, nargs="+", default=[1e-3],
                        help="CVX learning rate grid (default: 1e-3)")
    args, extra = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    snn_py = os.path.join(script_dir, "snn.py")
    venv_python = os.path.join(os.path.dirname(script_dir), ".venv", "bin", "python3")
    if not os.path.isfile(venv_python):
        venv_python = sys.executable

    total = len(T_LIST) * len(L_LIST)
    idx = 0
    for L in L_LIST:
        for T in T_LIST:
            if L == 2 and T in [2, 6, 10]:
                continue
            P_in, P_rec, P_last = scaled_widths(T)
            idx += 1
            cmd = [
                venv_python,
                snn_py,
                "--task", "two_step_xor_seq",
                "--T", str(T),
                "--L", str(L),
                "--P_in", str(P_in),
                "--P_rec", str(P_rec),
                "--P_last", str(P_last),
            ] + ["--log_train", "--seeds", "0", "--epochs", "200"]
            cmd += ["--lr_grid"] + [str(v) for v in args.lr_grid]
            cmd += extra
            print(f"[{idx}/{total}] L={L} T={T} P_in={P_in} P_rec={P_rec} P_last={P_last}")
            sys.stdout.flush()
            if args.dry_run:
                print("  ", " ".join(cmd))
                continue
            ret = subprocess.run(cmd, cwd=script_dir)
            if ret.returncode != 0:
                print(f"Exit code {ret.returncode} for L={L} T={T}", file=sys.stderr)
                sys.exit(ret.returncode)
    print("Grid done.")


if __name__ == "__main__":
    main()
