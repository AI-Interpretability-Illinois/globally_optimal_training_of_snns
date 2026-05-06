#!/usr/bin/env python3
"""
Re-run ``run_dfa_laststep_finetune_bench`` for several ``K_parallel`` values using the
**same** training/eval recipe as an existing sweep cell (``run_config.json`` + ``dfa.json``).

Typical use (reference cell from a prior DFA ablation):

    cd atomic
    python3 run_dfa_k_parallel_sweep_from_cell.py \\
        --cell_dir sweep_results/dfa_random_ablation_T9_ntrain5000_20260505_131956/q3_a3_seed60090 \\
        --k_list 4 8 16

Outputs under ``sweep_results/dfa_kpar_<dfa_spec>_T<T>_<timestamp>/K<K>/`` with ``dfa.json``,
``grid_cell.json``, ``metrics.json`` (from the bench), etc.

``P_rec`` and ``P_last`` must be divisible by each ``K`` (same rule as the bench).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

ATOMIC = Path(__file__).resolve().parent
BENCH = ATOMIC / "run_dfa_laststep_finetune_bench.py"

# Forwarded into the bench CLI (must exist in run_config.json).
_RUN_CONFIG_FORWARD_KEYS = (
    "seeds",
    "dfa_spec",
    "T",
    "n_train_pre",
    "n_val_pre",
    "n_train_ft",
    "n_val_ft",
    "n_test",
    "n_test_ood",
    "ood_T_multipliers",
    "finetune_seed_offset",
    "eval_seed_offset",
    "L",
    "P_rec",
    "P_last",
    "last_layer_readout",
    "beta_leak",
    "threshold",
    "optimizer_name",
    "ste_pretrain_epochs",
    "ste_finetune_epochs",
    "ste_lr_grid",
    "ste_beta_grid",
    "cvx_beta_grid",
    "cvx_bias_grid",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
)
_RUN_CONFIG_IGNORED_KEYS = frozenset({"stages", "K_parallel", "debug"})


def _load_run_config(cell_dir: Path) -> Dict[str, Any]:
    p = cell_dir / "run_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_run_config(cfg: Dict[str, Any]) -> None:
    extra = set(cfg.keys()) - set(_RUN_CONFIG_FORWARD_KEYS) - _RUN_CONFIG_IGNORED_KEYS
    if extra:
        raise ValueError(f"run_config.json has unknown keys (refuse to guess): {sorted(extra)}")
    missing = [k for k in _RUN_CONFIG_FORWARD_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"run_config.json missing keys: {missing}")


def _assert_width_divisible(*, p_rec: int, p_last: int, k: int) -> None:
    if int(p_rec) % int(k) != 0 or int(p_last) % int(k) != 0:
        raise ValueError(
            f"K_parallel={k} requires P_rec={p_rec} and P_last={p_last} divisible by K; "
            f"got P_rec%{k}={p_rec % k}, P_last%{k}={p_last % k}."
        )


def _build_bench_command(
    *,
    cfg: Dict[str, Any],
    k_parallel: int,
    out_root: Path,
    bench_debug: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(BENCH),
        "--out_root",
        str(out_root),
        "--K_parallel",
        str(int(k_parallel)),
    ]
    for key in _RUN_CONFIG_FORWARD_KEYS:
        val = cfg[key]
        flag = f"--{key}"
        if key in (
            "seeds",
            "ood_T_multipliers",
            "ste_lr_grid",
            "ste_beta_grid",
            "cvx_beta_grid",
            "cvx_bias_grid",
        ):
            seq = val
            if not isinstance(seq, list):
                raise TypeError(f"run_config[{key!r}] must be a list, got {type(val)}")
            cmd.append(flag)
            cmd.extend([str(x) for x in seq])
        else:
            cmd.extend([flag, str(val)])
    if bench_debug:
        cmd.append("--debug")
    elif "debug" in cfg:
        if cfg["debug"]:
            cmd.append("--debug")
    return cmd


def _prepare_cell_dir(
    *,
    source_cell: Path,
    dest_dir: Path,
    k_parallel: int,
    cfg: Dict[str, Any],
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dfa_src = source_cell / "dfa.json"
    if not dfa_src.is_file():
        raise FileNotFoundError(f"Missing {dfa_src}")
    shutil.copy2(dfa_src, dest_dir / "dfa.json")
    meta = {
        "sweep": "dfa_k_parallel_from_cell",
        "source_cell_dir": str(source_cell.resolve()),
        "K_parallel": int(k_parallel),
        "reference_dfa_spec": cfg["dfa_spec"],
        "reference_T": int(cfg["T"]),
    }
    (dest_dir / "grid_cell.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="K_parallel sweep from an existing DFA sweep cell directory.")
    ap.add_argument(
        "--cell_dir",
        type=str,
        required=True,
        help="Directory containing dfa.json and run_config.json (e.g. q3_a3_seed60090).",
    )
    ap.add_argument("--k_list", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument(
        "--sweep_root",
        type=str,
        default="",
        help="Parent directory for K subfolders. Default: sweep_results/dfa_kpar_<spec>_T<T>_<ts>",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a K if dest_dir/metrics.json already exists.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Forward --debug to the bench (small data, few epochs); for quick wiring checks.",
    )
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir).expanduser().resolve()
    cfg = _load_run_config(cell_dir)
    _validate_run_config(cfg)

    dfa_spec = str(cfg["dfa_spec"])
    t_train = int(cfg["T"])
    p_rec, p_last = int(cfg["P_rec"]), int(cfg["P_last"])
    k_list: Sequence[int] = [int(k) for k in args.k_list]
    if len(k_list) != len(set(k_list)):
        raise ValueError(f"--k_list must have unique values, got {k_list}")

    for k in k_list:
        _assert_width_divisible(p_rec=p_rec, p_last=p_last, k=k)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.sweep_root).strip():
        sweep_parent = Path(str(args.sweep_root)).expanduser().resolve()
    else:
        spec_slug = dfa_spec.replace("/", "_")
        sweep_parent = (
            Path.cwd() / "sweep_results" / f"dfa_kpar_{spec_slug}_T{t_train}_ntrain{int(cfg['n_train_pre'])}_{stamp}"
        )
    sweep_parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created": stamp,
        "source_cell_dir": str(cell_dir),
        "dfa_spec": dfa_spec,
        "k_list": list(k_list),
        "runs": [],
    }

    for k in k_list:
        dest = sweep_parent / f"K{k}"
        skip_reason = None
        if args.skip_existing and (dest / "metrics.json").is_file():
            skip_reason = "metrics.json exists"
        cmd_t = _build_bench_command(cfg=cfg, k_parallel=k, out_root=dest, bench_debug=bool(args.debug))
        rec: Dict[str, Any] = {
            "dir": str(dest),
            "K_parallel": k,
            "status": "skipped" if skip_reason else ("dry_run" if args.dry_run else "pending"),
            "skip_reason": skip_reason,
            "command": cmd_t,
        }
        manifest["runs"].append(rec)
        print("---", dest.name, "---", flush=True)
        print(" ".join(cmd_t), flush=True)
        if skip_reason:
            print("skip:", skip_reason, flush=True)
            continue
        if args.dry_run:
            continue
        _prepare_cell_dir(source_cell=cell_dir, dest_dir=dest, k_parallel=k, cfg=cfg)
        r = subprocess.run(cmd_t, cwd=str(ATOMIC))
        rec["status"] = "ok" if r.returncode == 0 else f"failed_exit_{r.returncode}"
        (sweep_parent / "sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if r.returncode != 0:
            raise SystemExit(f"Bench failed for K_parallel={k} (exit {r.returncode})")

    (sweep_parent / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"[done] manifest: {sweep_parent / 'sweep_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
