#!/usr/bin/env python3
"""
Grid sweep: random DFA ``random_{n_states}_{alphabet}_seed{…}`` for the five-stage
``run_dfa_laststep_finetune_bench`` pipeline.

**Default (non-legacy)**: for each ``( |Q|, |Σ| )`` cell, run **three stationary variants**
× **three instance seeds** (27 bench runs per grid pair):

- ``_dense`` — uniform stationary / column-balanced π under uniform input (default for comparability).
- ``_bias_accept`` — i.i.d. transitions until **some accepting state** has π(q) ≥ **k/Q** (default **k=2** ⇒ twice the uniform mass ``1/Q``; use ``--stationary_peak_factor 3`` or spec ``_bias_accept_k3``).
- ``_bias_reject`` — same for **some rejecting** state: max_{q∉F} π(q) ≥ **k/Q**.

Use ``--legacy_iid_transitions`` for the old layout: one **plain** ``random_*_seed{cell}`` (no suffix) per ``(q,a)``.

- Grid defaults: |Q| ∈ {5, 10, 15}, |Σ| ∈ {2, 3, 5, 7, 10}
- Train length T near 8–10 (default T=9)
- Split sizes aligned with the addition length-gen recipe: n_train=5000 for pre/finetune,
  n_val scaled from the 2304/512 ratio (~1111)
- Network grids match addition (L=3, P_rec=256, P_last=512, K=2, spike readout, β_leak=0.99)
- Each cell directory gets ``dfa.json`` (full transition table) + ``grid_cell.json`` before the bench runs

Run from ``atomic/``:

    python3 run_dfa_random_ablation_sweep.py

Or dry-run:

    python3 run_dfa_random_ablation_sweep.py --dry_run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ATOMIC = Path(__file__).resolve().parent
if str(ATOMIC) not in sys.path:
    sys.path.insert(0, str(ATOMIC))

_MOD_NAME = "_dfa_data_loader_ablation_sweep"
_SPEC = importlib.util.spec_from_file_location(
    _MOD_NAME, ATOMIC / "data_loaders" / "dfa_data_loader.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load data_loaders/dfa_data_loader.py")
_DFA_MOD = importlib.util.module_from_spec(_SPEC)
# Required for @dataclass on Py>=3.14: processing looks up sys.modules[cls.__module__].
sys.modules[_MOD_NAME] = _DFA_MOD
_SPEC.loader.exec_module(_DFA_MOD)
get_dfa = _DFA_MOD.get_dfa
save_dfa_json = _DFA_MOD.save_dfa_json


def _default_n_val_for_train(n_train: int) -> int:
    """Match 2304 train / 512 val ratio from addition sweeps."""
    return max(32, int(round(512 * float(n_train) / 2304.0)))


# Per-variant seed offset so instances and variants do not collide.
_VARIANT_CODE = {"dense": 1, "bias_accept": 2, "bias_reject": 3}


def _normalize_stationary_variants(raw: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in raw:
        t = str(x).strip().lower().replace("-", "_")
        if t in ("uniform", "uniform_stationary", "dense", "dstationary"):
            t = "dense"
        if t not in ("dense", "bias_accept", "bias_reject"):
            raise ValueError(f"unknown stationary variant {x!r} (use dense, bias_accept, bias_reject)")
        out.append(t)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate stationary variants: {list(raw)}")
    return out


def _build_laststep_cmd(
    *,
    dfa_spec: str,
    out_root: Path,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_test_ood: int,
    seeds: List[int],
    ood_T_multipliers: List[int],
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    beta_leak: float,
    threshold: float,
    last_layer_readout: str,
    ste_pretrain_epochs: int,
    ste_finetune_epochs: int,
    ste_lr_grid: List[float],
    ste_beta_grid: List[float],
    cvx_beta_grid: List[float],
    cvx_bias_grid: List[float],
    optimizer_name: str,
    finetune_seed_offset: int,
    eval_seed_offset: int,
) -> List[str]:
    bench = ATOMIC / "run_dfa_laststep_finetune_bench.py"
    cmd: List[str] = [
        sys.executable,
        str(bench),
        "--dfa_spec",
        dfa_spec,
        "--T",
        str(int(T)),
        "--n_train_pre",
        str(int(n_train)),
        "--n_val_pre",
        str(int(n_val)),
        "--n_train_ft",
        str(int(n_train)),
        "--n_val_ft",
        str(int(n_val)),
        "--n_test",
        str(int(n_test)),
        "--n_test_ood",
        str(int(n_test_ood)),
        "--L",
        str(int(L)),
        "--P_rec",
        str(int(P_rec)),
        "--P_last",
        str(int(P_last)),
        "--K_parallel",
        str(int(K_parallel)),
        "--beta_leak",
        str(float(beta_leak)),
        "--threshold",
        str(float(threshold)),
        "--last_layer_readout",
        str(last_layer_readout),
        "--optimizer_name",
        str(optimizer_name),
        "--ste_pretrain_epochs",
        str(int(ste_pretrain_epochs)),
        "--ste_finetune_epochs",
        str(int(ste_finetune_epochs)),
        "--finetune_seed_offset",
        str(int(finetune_seed_offset)),
        "--eval_seed_offset",
        str(int(eval_seed_offset)),
        "--out_root",
        str(out_root),
    ]
    cmd.extend(["--seeds", *[str(int(s)) for s in seeds]])
    cmd.extend(["--ood_T_multipliers", *[str(int(m)) for m in ood_T_multipliers]])
    cmd.extend(["--ste_lr_grid", *[str(float(x)) for x in ste_lr_grid]])
    cmd.extend(["--ste_beta_grid", *[str(float(x)) for x in ste_beta_grid]])
    cmd.extend(["--cvx_beta_grid", *[str(float(x)) for x in cvx_beta_grid]])
    cmd.extend(["--cvx_bias_grid", *[str(float(x)) for x in cvx_bias_grid]])
    return cmd


def _variant_to_suffix(variant: str, stationary_peak_factor: int) -> str:
    pk = int(stationary_peak_factor)
    if pk < 2:
        raise ValueError("stationary_peak_factor must be >= 2 (π_peak >= (factor)/Q, factor≥2 → ≥2/Q).")
    if variant == "dense":
        return "_dense"
    if variant == "bias_accept":
        s = "_bias_accept"
    elif variant == "bias_reject":
        s = "_bias_reject"
    else:
        raise ValueError(variant)
    if pk != 2:
        s += f"_k{pk}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Random DFA grid → last-step bench + saved DFA JSON")
    ap.add_argument("--n_states_grid", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--alphabet_grid", type=int, nargs="*", default=[2, 3, 5, 7, 10])
    ap.add_argument(
        "--dfa_seed_base",
        type=int,
        default=42,
        help="Base seed; per (q,a) anchor = base + n_states*10007 + alphabet*10009.",
    )
    ap.add_argument("--T", type=int, default=9, help="Train / ID sequence length (8–10 regime).")
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_val", type=int, default=-1, help="If <0, derive from 2304/512 ratio.")
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--n_test_ood", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--ood_T_multipliers", type=int, nargs="*", default=[2, 5, 10])
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=512)
    ap.add_argument("--P_last", type=int, default=2048)
    ap.add_argument("--K_parallel", type=int, default=128)
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--last_layer_readout", choices=["membrane", "spike"], default="spike")
    ap.add_argument("--optimizer_name", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--ste_pretrain_epochs", type=int, default=100)
    ap.add_argument("--ste_finetune_epochs", type=int, default=100)
    ap.add_argument("--finetune_seed_offset", type=int, default=1000)
    ap.add_argument("--eval_seed_offset", type=int, default=2000)
    ap.add_argument(
        "--ste_lr_grid",
        type=float,
        nargs="*",
        default=[0.001, 0.005, 0.01, 0.1],
    )
    ap.add_argument(
        "--ste_beta_grid",
        type=float,
        nargs="*",
        default=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    )
    ap.add_argument(
        "--cvx_beta_grid",
        type=float,
        nargs="*",
        default=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    )
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument(
        "--sweep_root",
        type=str,
        default="",
        help="Parent directory; default sweep_results/dfa_random_ablation_T{T}_{timestamp}",
    )
    ap.add_argument(
        "--legacy_iid_transitions",
        action="store_true",
        help="One plain random_* spec per (q,a) (no _dense / bias); ignore stationary variants.",
    )
    ap.add_argument(
        "--stationary_variants",
        type=str,
        nargs="*",
        default=["dense", "bias_accept", "bias_reject"],
        help="Suffix variants when not --legacy_iid_transitions (default: all three).",
    )
    ap.add_argument(
        "--dfa_instances_per_variant",
        type=int,
        default=1,
        help="Number of independent DFA construction seeds per (q,a, variant).",
    )
    ap.add_argument(
        "--stationary_peak_factor",
        type=int,
        default=2,
        help="Bias modes: require max π on some accept/reject state ≥ this × (1/Q). Appends _kN when N≠2.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a cell if its out_root already contains metrics.json",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    n_train = int(args.n_train)
    n_val = int(args.n_val) if int(args.n_val) > 0 else _default_n_val_for_train(n_train)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.sweep_root).strip():
        sweep_parent = Path(str(args.sweep_root)).expanduser().resolve()
    else:
        sweep_parent = (
            Path.cwd() / "sweep_results" / f"dfa_random_ablation_T{int(args.T)}_ntrain{n_train}_{stamp}"
        )
    sweep_parent.mkdir(parents=True, exist_ok=True)

    if bool(args.legacy_iid_transitions):
        variant_plan: List[Tuple[str, str]] = [("iid", "")]
        n_inst = 1
    else:
        variant_plan = [
            (v, _variant_to_suffix(v, int(args.stationary_peak_factor)))
            for v in _normalize_stationary_variants(args.stationary_variants)
        ]
        n_inst = int(args.dfa_instances_per_variant)
        if n_inst < 1:
            raise ValueError("dfa_instances_per_variant must be >= 1")

    manifest: Dict[str, Any] = {
        "created": stamp,
        "T": int(args.T),
        "n_train": n_train,
        "n_val": n_val,
        "n_states_grid": list(args.n_states_grid),
        "alphabet_grid": list(args.alphabet_grid),
        "dfa_seed_base": int(args.dfa_seed_base),
        "legacy_iid_transitions": bool(args.legacy_iid_transitions),
        "stationary_variants": [v for v, _ in variant_plan],
        "stationary_peak_factor": int(args.stationary_peak_factor),
        "dfa_instances_per_variant": n_inst,
        "cells": [],
    }

    for n_states in args.n_states_grid:
        for n_alpha in args.alphabet_grid:
            ns, na = int(n_states), int(n_alpha)
            if ns < 1 or na < 2:
                raise ValueError(f"Invalid grid cell n_states={ns} alphabet={na}")
            anchor = int(args.dfa_seed_base) + ns * 10007 + na * 10009

            for vlabel, vsfx in variant_plan:
                for inst in range(n_inst):
                    if bool(args.legacy_iid_transitions):
                        dfa_spec = f"random_{ns}_{na}_seed{anchor}"
                        cell_name = f"q{ns}_a{na}_seed{anchor}"
                        cell_seed = anchor
                    else:
                        vcode = int(_VARIANT_CODE[vlabel])
                        cell_seed = anchor + vcode * 1_000_000 + int(inst) * 97_981
                        dfa_spec = f"random_{ns}_{na}_seed{cell_seed}{vsfx}"
                        cell_name = f"q{ns}_a{na}_{vlabel}_i{inst}"
                    cell_dir = sweep_parent / cell_name
                    cell_dir.mkdir(parents=True, exist_ok=True)

                    dfa = get_dfa(dfa_spec)
                    save_dfa_json(cell_dir / "dfa.json", dfa)
                    cell_meta = {
                        "n_states": ns,
                        "alphabet_size": na,
                        "dfa_spec": dfa_spec,
                        "dfa_seed": cell_seed,
                        "stationary_variant": vlabel,
                        "instance_id": int(inst),
                        "anchor_seed": anchor,
                        "stationary_peak_factor": int(args.stationary_peak_factor),
                        "T": int(args.T),
                        "n_train": n_train,
                        "n_val": n_val,
                    }
                    (cell_dir / "grid_cell.json").write_text(
                        json.dumps(cell_meta, indent=2) + "\n", encoding="utf-8"
                    )

                    skip_reason = None
                    if args.skip_existing and (cell_dir / "metrics.json").exists():
                        skip_reason = "metrics.json exists"

                    cmd = _build_laststep_cmd(
                        dfa_spec=dfa_spec,
                        out_root=cell_dir,
                        T=int(args.T),
                        n_train=n_train,
                        n_val=n_val,
                        n_test=int(args.n_test),
                        n_test_ood=int(args.n_test_ood),
                        seeds=list(args.seeds),
                        ood_T_multipliers=list(args.ood_T_multipliers),
                        L=int(args.L),
                        P_rec=int(args.P_rec),
                        P_last=int(args.P_last),
                        K_parallel=int(args.K_parallel),
                        beta_leak=float(args.beta_leak),
                        threshold=float(args.threshold),
                        last_layer_readout=str(args.last_layer_readout),
                        ste_pretrain_epochs=int(args.ste_pretrain_epochs),
                        ste_finetune_epochs=int(args.ste_finetune_epochs),
                        ste_lr_grid=[float(x) for x in args.ste_lr_grid],
                        ste_beta_grid=[float(x) for x in args.ste_beta_grid],
                        cvx_beta_grid=[float(x) for x in args.cvx_beta_grid],
                        cvx_bias_grid=[float(x) for x in args.cvx_bias_grid],
                        optimizer_name=str(args.optimizer_name),
                        finetune_seed_offset=int(args.finetune_seed_offset),
                        eval_seed_offset=int(args.eval_seed_offset),
                    )
                    manifest["cells"].append(
                        {
                            "dir": str(cell_dir),
                            "dfa_spec": dfa_spec,
                            "stationary_variant": vlabel,
                            "instance_id": int(inst),
                            "status": "skipped" if skip_reason else ("dry_run" if args.dry_run else "pending"),
                            "skip_reason": skip_reason,
                            "command": cmd,
                        }
                    )
                    rec = manifest["cells"][-1]

                    print("---", cell_name, "---", flush=True)
                    print("saved", cell_dir / "dfa.json", flush=True)
                    if skip_reason:
                        print("skip:", skip_reason, flush=True)
                        (sweep_parent / "sweep_manifest.json").write_text(
                            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
                        )
                        continue
                    if args.dry_run:
                        print(" ".join(cmd), flush=True)
                        (sweep_parent / "sweep_manifest.json").write_text(
                            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
                        )
                        continue
                    r = subprocess.run(cmd, cwd=str(ATOMIC))
                    rec["status"] = "ok" if r.returncode == 0 else f"failed_exit_{r.returncode}"
                    (sweep_parent / "sweep_manifest.json").write_text(
                        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
                    )
                    if r.returncode != 0:
                        raise SystemExit(f"last-step bench failed for {dfa_spec} (exit {r.returncode})")

    print(f"[done] manifest: {sweep_parent / 'sweep_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
