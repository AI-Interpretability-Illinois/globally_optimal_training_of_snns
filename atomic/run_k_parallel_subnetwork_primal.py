"""K-parallel subnetwork sweep: fixed per-branch width, K grows with n, record primal.

Each of the ``K`` parallel LIF branches has a **constant** hidden width
``--width`` (default 10) in every layer. Total concat widths therefore scale as

    P_rec  = width * K
    P_last = width * K

``K`` is swept against the training sample size ``n = n_train``:

    K = 1, 0.1 n, 0.2 n, …, n, 1.1 n

(rounded to positive integers, duplicates dropped, order preserved). Override
with ``--k_list`` if you need an explicit schedule.

For every ``K`` we run Gaussian-init ``cvx_lite`` (primal-only LASSO / FISTA)
once (single beta, single bias — this is not a hyperparameter sweep) and dump
the **primal objective at every solver iteration**.

``--debug`` shrinks ``n_train`` to 20, uses ``frac_step=0.5`` (unless ``--k_list``
is given), and caps ``lite_max_iter`` at 50.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

if __package__ in (None, ""):
    from run_baselines_xor_mnist import _cvx_split_accs_from_config
    from run_lsm_xor_mnist import TASK_NAMES, TASK_PRESETS, TaskPreset, _load_task_data
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
else:
    from .run_baselines_xor_mnist import _cvx_split_accs_from_config
    from .run_lsm_xor_mnist import TASK_NAMES, TASK_PRESETS, TaskPreset, _load_task_data
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve


def build_k_schedule(n: int, *, frac_step: float, overshoot: float) -> List[int]:
    """K = 1, frac_step*n, 2*frac_step*n, …, n, overshoot*n (unique, K>=1)."""
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if float(frac_step) <= 0.0:
        raise ValueError(f"frac_step must be > 0, got {frac_step}.")
    if float(overshoot) < 1.0:
        raise ValueError(f"overshoot must be >= 1, got {overshoot}.")
    ks: List[int] = [1]
    f = float(frac_step)
    while f <= 1.0 + 1e-12:
        ks.append(max(1, int(round(f * float(n)))))
        f += float(frac_step)
    ks.append(max(1, int(round(float(overshoot) * float(n)))))
    out: List[int] = []
    seen = set()
    for k in ks:
        if k in seen:
            continue
        seen.add(k)
        out.append(int(k))
    return out


def _parse_k_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError(f"K values must be >= 1, got {k}.")
        out.append(k)
    if len(out) == 0:
        raise ValueError("k_list is empty.")
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _md_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = (
        "| K | K/n | P_rec | P_last | n_iter | primal | train_acc | val_acc | test_acc |\n"
        "|---|-----|-------|--------|--------|--------|-----------|---------|----------|\n"
    )
    lines = [header]
    for r in rows:
        kn = r["K_over_n"]
        lines.append(
            f"| {r['K']} | {kn:.4g} | {r['P_rec']} | {r['P_last']} | {r['n_iter']} | "
            f"{r['primal_value']:.8g} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['test_acc']:.4f} |\n"
        )
    return "".join(lines)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASK_NAMES, default="xor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--T", type=int, default=None, help="Override preset T. Default: first T in the preset.")
    ap.add_argument("--L", type=int, default=None, help="Override preset L. Default: first L in the preset.")
    ap.add_argument(
        "--width",
        type=int,
        default=10,
        help="Neurons per layer in each subnetwork. P_rec = P_last = width * K.",
    )
    ap.add_argument("--frac_step", type=float, default=0.1, help="K grid step as a fraction of n (ignored if --k_list).")
    ap.add_argument("--overshoot", type=float, default=1.1, help="Final K = overshoot * n (default 1.1 n).")
    ap.add_argument("--k_list", type=str, default="", help="Optional explicit K list; skips the n-relative schedule.")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--cvx_method", choices=("cvx_lite",), default="cvx_lite")
    ap.add_argument("--lite_max_iter", type=int, default=5000)
    ap.add_argument("--lite_tol", type=float, default=1e-6)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_root", type=str, default="sweep_results/k_parallel_subnetwork_primal")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.width) < 1:
        raise ValueError(f"width must be >= 1, got {args.width}.")
    if int(args.n_train) < 1:
        raise ValueError(f"n_train must be >= 1, got {args.n_train}.")

    n_train = 20 if args.debug else int(args.n_train)
    n_val = 10 if args.debug else int(args.n_val)
    n_test = 20 if args.debug else int(args.n_test)
    lite_max_iter = 50 if args.debug else int(args.lite_max_iter)
    frac_step = 0.5 if args.debug and not str(args.k_list).strip() else float(args.frac_step)

    base_preset = TASK_PRESETS[str(args.task)]
    T = int(base_preset.T_list[0]) if args.T is None else int(args.T)
    L = int(base_preset.L_list[0]) if args.L is None else int(args.L)
    preset = TaskPreset(
        **{
            **asdict(base_preset),
            "T_list": (T,),
            "L_list": (L,),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
        }
    )

    if str(args.k_list).strip():
        k_values = _parse_k_list(str(args.k_list))
    else:
        k_values = build_k_schedule(int(n_train), frac_step=float(frac_step), overshoot=float(args.overshoot))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).expanduser().resolve() / f"{preset.name}_w{int(args.width)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_task_data(preset, T=T, seed=int(args.seed))
    if int(data["x_train"].shape[0]) != int(n_train):
        raise ValueError(
            f"Loaded n_train={data['x_train'].shape[0]} != requested n_train={n_train}."
        )

    print(
        (
            f"[k-sub] task={preset.name} n={n_train} T={T} L={L} width={int(args.width)} "
            f"K={k_values} method={args.cvx_method} lite_max_iter={lite_max_iter} "
            f"loss={preset.loss_name} readout={preset.last_layer_readout}"
        ),
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for K in k_values:
        P_rec = int(args.width) * int(K)
        P_last = int(args.width) * int(K)
        if P_rec % int(K) != 0 or P_last % int(K) != 0:
            raise ValueError(f"P_rec={P_rec} / P_last={P_last} not divisible by K={K}.")
        reservoir_seed = int(args.seed) * 100003 + int(K)
        init_cfg = InitializationConfig(
            mode="gaussian",
            variant="standard",
            seed=int(reservoir_seed),
            feature_count=int(P_last),
            bias=float(args.bias),
            pretrained_weights=None,
            L=int(L),
            P_rec=int(P_rec),
            P_last=int(P_last),
            K_parallel=int(K),
            beta_leak=0.99,
            threshold=1.0,
            last_layer_readout=str(preset.last_layer_readout),
        )
        solve_cfg = SolveConfig(
            loss_name=str(preset.loss_name),
            method=str(args.cvx_method),
            beta=float(args.beta),
            compute_ce_dual=False,
            cvx_ovr_workers=1,
            lite_max_iter=int(lite_max_iter),
            lite_tol=float(args.lite_tol),
        )
        print(
            f"[k-sub] solving K={K} K/n={K / float(n_train):.4g} P_rec={P_rec} P_last={P_last}",
            flush=True,
        )
        out = cvx_solve(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_val=data["x_val"],
            y_val=data["y_val"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            init_cfg=init_cfg,
            solve_cfg=solve_cfg,
        )
        if not isinstance(out.trained_model, dict) or "weights" not in out.trained_model:
            raise TypeError(f"cvx_lite must return trained_model['weights']; got {type(out.trained_model)}.")
        accs = _cvx_split_accs_from_config(
            init_cfg=init_cfg,
            cvx_weights=out.trained_model["weights"],
            data=data,
        )
        primal_history = [float(v) for v in list(out.loss_history)]
        if len(primal_history) == 0:
            raise RuntimeError(f"Empty primal history for K={K}.")
        entry: Dict[str, Any] = {
            "task": preset.name,
            "dataset": preset.dataset,
            "seed": int(args.seed),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "T": int(T),
            "L": int(L),
            "K": int(K),
            "K_over_n": float(K) / float(n_train),
            "width": int(args.width),
            "P_rec": int(P_rec),
            "P_last": int(P_last),
            "beta": float(args.beta),
            "bias": float(args.bias),
            "loss_name": str(preset.loss_name),
            "last_layer_readout": str(preset.last_layer_readout),
            "cvx_method": str(args.cvx_method),
            "lite_max_iter": int(lite_max_iter),
            "lite_tol": float(args.lite_tol),
            "n_iter": int(len(primal_history)),
            "primal_value": float(out.diagnostics.primal_value),
            "primal_history": primal_history,
            "train_acc": float(accs["train_acc"]),
            "val_acc": float(accs["val_acc"]),
            "test_acc": float(accs["test_acc"]),
            "final_losses": {k: float(v) for k, v in out.final_losses.items()},
        }
        _write_json(out_dir / f"seed{int(args.seed)}_K{int(K)}.json", entry)
        rows.append(entry)
        print(
            (
                f"[k-sub] K={K} n_iter={entry['n_iter']} primal={entry['primal_value']:.8g} "
                f"train_acc={entry['train_acc']:.4f} val_acc={entry['val_acc']:.4f} "
                f"test_acc={entry['test_acc']:.4f}"
            ),
            flush=True,
        )

    summary = {
        "task": preset.name,
        "config": asdict(preset),
        "width": int(args.width),
        "n_train": int(n_train),
        "frac_step": float(frac_step),
        "overshoot": float(args.overshoot),
        "k_values": [int(k) for k in k_values],
        "beta": float(args.beta),
        "bias": float(args.bias),
        "lite_max_iter": int(lite_max_iter),
        "debug": bool(args.debug),
        "results": [
            {k: v for k, v in r.items() if k != "primal_history"} | {"n_iter": r["n_iter"]}
            for r in rows
        ],
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "metrics.json", summary)
    (out_dir / "primal_vs_K.md").write_text(
        f"# K-parallel subnetworks (width={int(args.width)}, n={n_train}, task={preset.name})\n\n"
        + _md_table(rows)
        + "\nPer-K `primal_history` is in `seed*_K*.json`.\n"
    )
    print(f"[k-sub] wrote {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
