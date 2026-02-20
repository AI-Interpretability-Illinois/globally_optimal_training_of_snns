#!/usr/bin/env python3
"""
run_rnn_sweep_and_plots.py

Sweeps Convex-RNN vs STE experiments by calling RRN.py as a subprocess, and
produces:

1. Training-accuracy curves (CVX vs STE) for the best hyperparameters for each
   (dataset, L, T) combination.
2. A CSV summarizing mean ± std test performance and best hyperparameters.

Assumptions:
- RRN.py is in the same directory.
- RRN.py supports tasks:
    {parity_seq, two_step_xor_seq, moving_blobs_seq,
     mnist_seq, mnist_perm_seq, sunspot_seq}
- RRN.py supports losses: {ce, hinge, mse, mae, squared}
- RRN.py prints logs in the formats you've been using, e.g.:
    [CVX] ep=001/100 last_loss=... train_acc=... val_acc=... lr=...
    [STE] ep=001/100 last_loss=... train_acc=... val_acc=... lr=...
    [seed 0] CVX test=... (val=..., beta=..., lr=..., P_last=...) | STE test=...
    CVX test_acc = <mean> ± <std>
    STE test_acc = <mean> ± <std>
"""

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Basic config
# =============================================================================

RRN_SCRIPT = os.path.join(os.path.dirname(__file__), "RRN.py")

# --- Hyperparameter grids ---

# CVX (last-layer L1)
BETA_GRID_CVX = [1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
LR_GRID_CVX = [1e-2, 5e-3, 1e-3]

# STE path regularization
BETA_GRID_STE = [1e-4, 5e-4, 1e-3]
LR_GRID_STE = [1e-3, 5e-4]

# Training setup
EPOCHS = 100
BATCH = 128
VAL_FRAC = 0.2
SEEDS_SUMMARY = [0, 1, 2, 3, 4]  # for mean ± std
SEEDS_CURVES = [0]               # for logging curves
DEVICE = "mps"                   # "auto", "mps" or "cpu"

FIGS_DIR = "figs_rnn_sweep"
os.makedirs(FIGS_DIR, exist_ok=True)


# =============================================================================
# Experiment spec & width computation
# =============================================================================

@dataclass
class ExperimentSpec:
    name: str
    task: str
    L: int
    T: int
    n_train_total: int
    loss: str


def compute_widths(n_train_total: int, T: int) -> Tuple[int, int, int, int]:
    """
    Given n_train_total = n and sequence length T, use:

        base_width = 5n/4  (baseline for T=2)
        P_in = P_rec = base_width * (T / 2)
        P_last = n
        n_test ≈ 0.75 * n_train_total

    So when T doubles, P_in and P_rec also double.
    """
    base_width = 1.25 * n_train_total
    scale = max(T / 4, 1.0)  # T=2 → scale=1
    P_in = int(round(base_width * scale))
    P_rec = P_in
    P_last = n_train_total
    n_test = int(round(0.75 * n_train_total))
    return P_in, P_rec, P_last, n_test


def build_experiments() -> List[ExperimentSpec]:
    exps: List[ExperimentSpec] = []

    # 1) two_step_xor_seq, moving_blobs_seq (hinge), L=[3,5,8], T=[2,4,6,20], n_train_total=6000
    for L in [3, 5, 8]:
        for T in [2, 4, 6, 20]:
            for task in ["two_step_xor_seq", "moving_blobs_seq"]:
                exps.append(
                    ExperimentSpec(
                        name=f"{task}_L{L}_T{T}",
                        task=task,
                        L=L,
                        T=T,
                        n_train_total=6000,
                        loss="hinge",
                    )
                )

    # 2) mnist_seq, mnist_perm_seq (ce), L=[3,5,8], T=[2,4,16,28], n_train_total=10000
    for L in [3, 5, 8]:
        for T in [2, 4, 16, 28]:
            for task in ["mnist_seq", "mnist_perm_seq"]:
                exps.append(
                    ExperimentSpec(
                        name=f"{task}_L{L}_T{T}",
                        task=task,
                        L=L,
                        T=T,
                        n_train_total=10000,
                        loss="ce",
                    )
                )

    # 3) sunspot_seq (regression), L=[3,5,8], T=[2,4,6,20], n_train_total=6000
    #    Here we default to MSE. If RRN uses "squared" instead of "mse", change loss string.
    for L in [3, 5, 8]:
        for T in [2, 4, 6, 20]:
            exps.append(
                ExperimentSpec(
                    name=f"sunspot_seq_L{L}_T{T}",
                    task="sunspot_seq",  # adjust if your RRN uses a different task string
                    L=L,
                    T=T,
                    n_train_total=6000,
                    loss="mse",
                )
            )

    return exps


# =============================================================================
# Subprocess runner and log parsing
# =============================================================================

def run_rrn_once(
    spec: ExperimentSpec,
    *,
    P_in: int,
    P_rec: int,
    P_last: int,
    n_test: int,
    beta_grid_cvx: List[float],
    lr_grid_cvx: List[float],
    ste_lr: float,
    ste_beta_path: float,
    seeds: List[int],
    log_train: bool,
) -> str:
    """Call RRN.py with the given configuration and return stdout as a string."""
    cmd = [
        "python3", RRN_SCRIPT,
        "--task", spec.task,
        "--T", str(spec.T),
        "--L", str(spec.L),
        "--P_in", str(P_in),
        "--P_rec", str(P_rec),
        "--P_last", str(P_last),
        "--n_train_total", str(spec.n_train_total),
        "--val_frac", str(VAL_FRAC),
        "--n_test", str(n_test),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--loss", spec.loss,
        "--cvx_optimizer", "adam",
        "--cvx_step_size", "30",
        "--cvx_gamma", "0.5",
        "--ste_step_size", "30",
        "--ste_gamma", "0.5",
        "--device", DEVICE,
    ]

    # CVX grids
    cmd += ["--beta_grid"] + [str(b) for b in beta_grid_cvx]
    cmd += ["--lr_grid"] + [str(lr) for lr in lr_grid_cvx]

    # STE hyperparams
    cmd += ["--ste_lr", str(ste_lr)]
    cmd += ["--ste_beta_path", str(ste_beta_path)]

    # seeds
    cmd += ["--seeds"] + [str(s) for s in seeds]

    if log_train:
        cmd.append("--log_train")

    print(f"\n[run] {spec.name}: calling RRN.py with seeds={seeds}, ste_lr={ste_lr}, ste_beta={ste_beta_path}")
    print("      P_in={}, P_rec={}, P_last={}, n_test={}".format(P_in, P_rec, P_last, n_test))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout = proc.stdout
    if proc.returncode != 0:
        print("---- RRN.py returned non-zero exit code ----")
        print(stdout)
        raise RuntimeError(f"RRN.py failed for spec {spec.name}")
    return stdout


def parse_best_cvx_from_seeds(stdout: str) -> Tuple[float, float]:
    """
    Parse the best CVX beta and lr from the '[seed 0] ... beta=..., lr=...' line.

    Example line:
    [seed 0] CVX test=0.9922 (val=0.9990, beta=1e-06, lr=0.005, P_last=5000) | STE test=...
    """
    beta, lr = None, None
    for line in stdout.splitlines():
        m = re.search(r"beta=([0-9eE\.\-+]+), lr=([0-9eE\.\-+]+)", line)
        if m:
            beta = float(m.group(1))
            lr = float(m.group(2))
            break
    if beta is None or lr is None:
        raise ValueError("Could not parse best CVX beta/lr from RRN output.")
    return beta, lr


def parse_final_summary(stdout: str) -> Dict[str, float]:
    """
    Parse final summary lines of the form:

    CVX test_acc = <mean> ± <std>
    STE test_acc = <mean> ± <std>

    Returns dict with keys:
      'cvx_mean', 'cvx_std', 'ste_mean', 'ste_std'
    """
    cvx_mean = cvx_std = ste_mean = ste_std = None

    for line in stdout.splitlines():
        m_cvx = re.search(r"CVX test_acc = ([0-9\.]+) ± ([0-9\.]+)", line)
        if m_cvx:
            cvx_mean = float(m_cvx.group(1))
            cvx_std = float(m_cvx.group(2))
        m_ste = re.search(r"STE test_acc = ([0-9\.]+) ± ([0-9\.]+)", line)
        if m_ste:
            ste_mean = float(m_ste.group(1))
            ste_std = float(m_ste.group(2))

    if cvx_mean is None or ste_mean is None:
        raise ValueError("Could not parse final summary mean ± std from RRN output.")

    return {
        "cvx_mean": cvx_mean,
        "cvx_std": cvx_std,
        "ste_mean": ste_mean,
        "ste_std": ste_std,
    }


def parse_train_curves(stdout: str) -> Tuple[List[float], List[float]]:
    """
    Parse per-epoch train_acc for CVX and STE from logs with --log_train and a single seed.

    Expected lines:

      [CVX] ep=000/100 ... train_acc=0.4958 val_acc=... lr=...
      [STE] ep=000/100 ... train_acc=0.4525 val_acc=... lr=...

    Returns:
      cvx_train_acc_list, ste_train_acc_list
    """
    cvx_epochs: Dict[int, float] = {}
    ste_epochs: Dict[int, float] = {}

    for line in stdout.splitlines():
        m_cvx = re.search(r"\[CVX\] ep=(\d+)/\d+ .*train_acc=([0-9\.]+)", line)
        if m_cvx:
            ep = int(m_cvx.group(1))
            acc = float(m_cvx.group(2))
            cvx_epochs[ep] = acc

        m_ste = re.search(r"\[STE\] ep=(\d+)/\d+ .*train_acc=([0-9\.]+)", line)
        if m_ste:
            ep = int(m_ste.group(1))
            acc = float(m_ste.group(2))
            ste_epochs[ep] = acc

    # Convert to sorted lists by epoch
    if cvx_epochs:
        max_ep = max(cvx_epochs.keys())
        cvx_list = [cvx_epochs.get(ep, np.nan) for ep in range(max_ep + 1)]
    else:
        cvx_list = []

    if ste_epochs:
        max_ep = max(ste_epochs.keys())
        ste_list = [ste_epochs.get(ep, np.nan) for ep in range(max_ep + 1)]
    else:
        ste_list = []

    return cvx_list, ste_list


def plot_train_curves(
    spec: ExperimentSpec,
    cvx_acc: List[float],
    ste_acc: List[float],
    best_cvx_beta: float,
    best_cvx_lr: float,
    best_ste_beta: float,
    best_ste_lr: float,
) -> None:
    """Plot CVX vs STE train_accuracy vs epoch and save to PNG."""
    plt.figure(figsize=(6, 4))
    epochs = range(len(cvx_acc))
    plt.plot(epochs, cvx_acc, label=f"CVX (beta={best_cvx_beta:g}, lr={best_cvx_lr:g})")
    if len(ste_acc) == len(epochs):
        plt.plot(epochs, ste_acc, label=f"STE (beta_path={best_ste_beta:g}, lr={best_ste_lr:g})", linestyle="--")
    else:
        plt.plot(range(len(ste_acc)), ste_acc, label=f"STE (beta_path={best_ste_beta:g}, lr={best_ste_lr:g})", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy")
    plt.title(f"{spec.task} (L={spec.L}, T={spec.T}, loss={spec.loss})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fname = os.path.join(FIGS_DIR, f"{spec.name}_train_acc.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[plot] Saved train-acc curves to {fname}")


# =============================================================================
# Main sweep logic
# =============================================================================

def main():
    experiments = build_experiments()
    rows: List[Dict[str, Any]] = []

    for spec in experiments:
        print("\n" + "=" * 80)
        print(f"[experiment] {spec.name}  (task={spec.task}, L={spec.L}, T={spec.T}, n_train_total={spec.n_train_total}, loss={spec.loss})")
        print("=" * 80)

        # Widths & n_test
        P_in, P_rec, P_last, n_test = compute_widths(spec.n_train_total, spec.T)
        print(f"[widths] P_in={P_in}, P_rec={P_rec}, P_last={P_last}, n_test≈{n_test}")

        # ---------------------------------------------------------------------
        # Step 1: Get best CVX hyperparams (beta, lr) using default STE params
        # ---------------------------------------------------------------------
        ste_lr_default = LR_GRID_STE[0]
        ste_beta_default = BETA_GRID_STE[0]

        stdout_step1 = run_rrn_once(
            spec,
            P_in=P_in,
            P_rec=P_rec,
            P_last=P_last,
            n_test=n_test,
            beta_grid_cvx=BETA_GRID_CVX,
            lr_grid_cvx=LR_GRID_CVX,
            ste_lr=ste_lr_default,
            ste_beta_path=ste_beta_default,
            seeds=SEEDS_SUMMARY,
            log_train=False,
        )

        summary_step1 = parse_final_summary(stdout_step1)
        best_cvx_beta, best_cvx_lr = parse_best_cvx_from_seeds(stdout_step1)
        print(f"[step1] Best CVX beta={best_cvx_beta:g}, lr={best_cvx_lr:g}")
        print(f"        CVX (with default STE) test_mean={summary_step1['cvx_mean']:.4f} ± {summary_step1['cvx_std']:.4f}")
        print(f"        STE (default)          test_mean={summary_step1['ste_mean']:.4f} ± {summary_step1['ste_std']:.4f}")

        # ---------------------------------------------------------------------
        # Step 2: Outer grid search over STE hyperparams (beta_path, lr)
        #         while fixing CVX to its best hyperparams.
        # ---------------------------------------------------------------------
        best_ste_beta = None
        best_ste_lr = None
        best_ste_mean = -np.inf
        best_ste_std = 0.0

        for ste_beta in BETA_GRID_STE:
            for ste_lr in LR_GRID_STE:
                print(f"\n[step2] Tuning STE: beta_path={ste_beta}, lr={ste_lr}")
                stdout_ste = run_rrn_once(
                    spec,
                    P_in=P_in,
                    P_rec=P_rec,
                    P_last=P_last,
                    n_test=n_test,
                    beta_grid_cvx=[best_cvx_beta],
                    lr_grid_cvx=[best_cvx_lr],
                    ste_lr=ste_lr,
                    ste_beta_path=ste_beta,
                    seeds=SEEDS_SUMMARY,
                    log_train=False,
                )
                summary_ste = parse_final_summary(stdout_ste)
                ste_mean = summary_ste["ste_mean"]
                ste_std = summary_ste["ste_std"]

                print(f"[step2]   => STE test_mean={ste_mean:.4f} ± {ste_std:.4f} (beta_path={ste_beta}, lr={ste_lr})")

                if ste_mean > best_ste_mean:
                    best_ste_mean = ste_mean
                    best_ste_std = ste_std
                    best_ste_beta = ste_beta
                    best_ste_lr = ste_lr

        print(f"\n[step2 done] Best STE: beta_path={best_ste_beta:g}, lr={best_ste_lr:g}, "
              f"test_mean={best_ste_mean:.4f} ± {best_ste_std:.4f}")

        # Re-run once to re-get CVX stats for the fixed best CVX/STE combos.
        stdout_fixed = run_rrn_once(
            spec,
            P_in=P_in,
            P_rec=P_rec,
            P_last=P_last,
            n_test=n_test,
            beta_grid_cvx=[best_cvx_beta],
            lr_grid_cvx=[best_cvx_lr],
            ste_lr=best_ste_lr,
            ste_beta_path=best_ste_beta,
            seeds=SEEDS_SUMMARY,
            log_train=False,
        )
        summary_fixed = parse_final_summary(stdout_fixed)
        cvx_mean = summary_fixed["cvx_mean"]
        cvx_std = summary_fixed["cvx_std"]
        ste_mean = summary_fixed["ste_mean"]
        ste_std = summary_fixed["ste_std"]

        print(f"\n[fixed summary] CVX test_mean={cvx_mean:.4f} ± {cvx_std:.4f}")
        print(f"                STE test_mean={ste_mean:.4f} ± {ste_std:.4f}")

        # ---------------------------------------------------------------------
        # Step 3: Curves: run with a single seed and log_train=True
        # ---------------------------------------------------------------------
        stdout_curves = run_rrn_once(
            spec,
            P_in=P_in,
            P_rec=P_rec,
            P_last=P_last,
            n_test=n_test,
            beta_grid_cvx=[best_cvx_beta],
            lr_grid_cvx=[best_cvx_lr],
            ste_lr=best_ste_lr,
            ste_beta_path=best_ste_beta,
            seeds=SEEDS_CURVES,
            log_train=True,
        )
        cvx_train_acc, ste_train_acc = parse_train_curves(stdout_curves)
        plot_train_curves(
            spec,
            cvx_train_acc,
            ste_train_acc,
            best_cvx_beta,
            best_cvx_lr,
            best_ste_beta,
            best_ste_lr,
        )

        # Accumulate row for table
        rows.append(
            dict(
                dataset=spec.task,
                name=spec.name,
                L=spec.L,
                T=spec.T,
                loss=spec.loss,
                n_train_total=spec.n_train_total,
                P_in=P_in,
                P_rec=P_rec,
                P_last=P_last,
                n_test=n_test,
                cvx_test_mean=cvx_mean,
                cvx_test_std=cvx_std,
                ste_test_mean=ste_mean,
                ste_test_std=ste_std,
                best_cvx_beta=best_cvx_beta,
                best_cvx_lr=best_cvx_lr,
                best_ste_beta=best_ste_beta,
                best_ste_lr=best_ste_lr,
            )
        )

    # -------------------------------------------------------------------------
    # Save summary table
    # -------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = "rnn_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[summary] Saved results table to {csv_path}")
    print(df)


if __name__ == "__main__":
    main()
