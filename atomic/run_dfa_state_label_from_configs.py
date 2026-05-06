#!/usr/bin/env python3
"""
Run ``run_dfa_state_label_finetune_bench.py`` using:

* **dfa_config** — JSON with at least ``dfa_spec`` (e.g. ablation ``grid_cell.json``). Extra keys are ignored.
* **run_config** — Same shape as ``run_dfa_laststep_finetune_bench.py`` writes under ``run_config.json`` /
  ``metrics.json`` → ``run_config`` (``dfa_laststep`` recipe: splits, ``L``, ``P_*``, ``K_parallel``,
  ``last_layer_readout``, grids, etc.). **No** ``run_mode`` / variants here.

**CLI** supplies everything that only exists on the state-label bench (or that you want to override):

* ``--run-mode``, ``--pretrain-variant``, ``--finetune-variant``, ``--out-root``, ``--output-json``
* Optional: ``--lambda-sum-grid`` / ``--lambda-carry-grid``, ``--lambda-carry``, ``--verify-samples``,
  ``--batch-size``, ``--cvx-device``, ``--cvx-grid-workers``, ``--cvx-ovr-workers``, ``--blas-threads``,
  loss / time-objective flags, ``--debug``.

``last_layer_readout`` from the JSON is copied to both ``--ste-last-layer-readout`` and
``--cvx-last-layer-readout`` for the state-label bench.

Examples (from ``atomic/``)::

    # run_config = inner object from a laststep metrics file (or standalone laststep run_config.json)
    python3 run_dfa_state_label_from_configs.py \\
      --dfa_config sweep_results/.../q5_a2_dense_i0/grid_cell.json \\
      --run_config sweep_results/.../q5_a2_dense_i0/metrics.json \\
      --run-mode full \\
      --out-root sweep_results/dfa_state_label_my_run \\
      --lambda-sum-grid 0.125 1.0 4.0 10.0 \\
      --cvx-grid-workers 12

    python3 run_dfa_state_label_from_configs.py --dry_run ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ATOMIC = Path(__file__).resolve().parent
BENCH = ATOMIC / "run_dfa_state_label_finetune_bench.py"

# Keys written by run_dfa_laststep_finetune_bench.config_dump (must match that script).
_LASTSTEP_RUN_KEYS = frozenset(
    {
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
        "K_parallel",
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
        "stages",
        "debug",
    }
)

# Not forwarded as CLI flags to the state-label bench.
_SKIP_ARGV = frozenset({"stages"})

# If present in JSON, user probably meant the state-label bench JSON — reject.
_WRONG_RUN_CONFIG_KEYS = frozenset(
    {
        "run_mode",
        "pretrain_variant",
        "finetune_variant",
        "out_root",
        "output_json",
        "ste_last_layer_readout",
        "cvx_last_layer_readout",
        "lambda_carry",
        "lambda_sum_grid",
        "lambda_carry_grid",
        "lambda_sum_grid_swept_state_head",
        "lambda_carry_fixed_label_head",
        "naming",
        "evaluation_modes",
        "verify_samples",
        "cvx_device",
        "ste_sum_loss",
        "ste_carry_loss",
        "cvx_sum_loss",
        "cvx_carry_loss",
        "tf_objective",
        "ste_time_loss",
        "cvx_time_loss",
        "cvx_grid_workers",
        "cvx_ovr_workers",
        "blas_threads",
    }
)

_STATE_SINGLE = frozenset(
    {
        "dfa_spec",
        "T",
        "n_train_pre",
        "n_val_pre",
        "n_train_ft",
        "n_val_ft",
        "n_test",
        "n_test_ood",
        "verify_samples",
        "finetune_seed_offset",
        "eval_seed_offset",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
        "L",
        "P_rec",
        "P_last",
        "K_parallel",
        "beta_leak",
        "threshold",
        "ste_last_layer_readout",
        "cvx_last_layer_readout",
        "optimizer_name",
        "ste_pretrain_epochs",
        "ste_finetune_epochs",
        "batch_size",
        "lambda_carry",
        "cvx_device",
        "ste_sum_loss",
        "ste_carry_loss",
        "cvx_sum_loss",
        "cvx_carry_loss",
        "tf_objective",
        "ste_time_loss",
        "cvx_time_loss",
        "cvx_grid_workers",
        "cvx_ovr_workers",
        "blas_threads",
        "run_mode",
        "pretrain_variant",
        "finetune_variant",
        "out_root",
        "output_json",
    }
)
_STATE_LIST_FLOAT = frozenset(
    {"lambda_sum_grid", "lambda_carry_grid", "ste_lr_grid", "ste_beta_grid", "cvx_beta_grid", "cvx_bias_grid"}
)
_STATE_LIST_INT = frozenset({"seeds", "ood_T_multipliers"})


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_run_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if "run_config" in raw and isinstance(raw["run_config"], dict):
        return dict(raw["run_config"])
    return dict(raw)


def _apply_dfa_config(dfa_cfg: Mapping[str, Any], rc: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if "dfa_spec" not in dfa_cfg:
        raise KeyError('dfa_config must contain "dfa_spec".')
    dfa_spec = str(dfa_cfg["dfa_spec"])
    rc = dict(rc)
    rc["dfa_spec"] = dfa_spec
    if "T" in dfa_cfg and "T" not in rc:
        rc["T"] = int(dfa_cfg["T"])
    if "n_train" in dfa_cfg and "n_train_pre" not in rc and "n_train_ft" not in rc:
        nt = int(dfa_cfg["n_train"])
        rc["n_train_pre"] = nt
        rc["n_train_ft"] = nt
    if "n_val" in dfa_cfg and "n_val_pre" not in rc and "n_val_ft" not in rc:
        nv = int(dfa_cfg["n_val"])
        rc["n_val_pre"] = nv
        rc["n_val_ft"] = nv
    return dfa_spec, rc


def _laststep_to_state_label_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    ovl = set(raw) & _WRONG_RUN_CONFIG_KEYS
    if ovl:
        raise ValueError(
            f"run_config looks like a state-label dump, not laststep: disallowed keys {sorted(ovl)}. "
            "Use metrics/run_config from run_dfa_laststep_finetune_bench; put run_mode / out_root on the CLI."
        )
    bad = set(raw) - _LASTSTEP_RUN_KEYS
    if bad:
        raise ValueError(
            f"run_config: unknown keys {sorted(bad)} for laststep-shaped config. "
            f"Expected subset of: {sorted(_LASTSTEP_RUN_KEYS)}"
        )
    out = dict(raw)
    if "last_layer_readout" not in out:
        raise KeyError('laststep run_config must contain "last_layer_readout".')
    lro = str(out.pop("last_layer_readout"))
    out["ste_last_layer_readout"] = lro
    out["cvx_last_layer_readout"] = lro
    for k in _SKIP_ARGV:
        out.pop(k, None)
    return out


def _merge_to_argv(rc: Dict[str, Any]) -> List[str]:
    if "lambda_sum_grid" in rc and "lambda_carry_grid" in rc:
        raise ValueError("Set only one of lambda_sum_grid and lambda_carry_grid.")

    argv: List[str] = []
    for k in sorted(_STATE_SINGLE & set(rc)):
        if k in ("out_root", "output_json") and str(rc[k]).strip() == "":
            continue
        v = rc[k]
        if v is None:
            raise ValueError(f"merged config: {k!r} is null.")
        argv.extend([f"--{k.replace('_', '-')}", str(v)])

    for k in sorted(_STATE_LIST_FLOAT & set(rc)):
        seq = rc[k]
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
            raise TypeError(f"merged[{k!r}] must be a list of floats, got {type(seq).__name__}")
        if not seq:
            raise ValueError(f"merged[{k!r}] must be non-empty.")
        argv.append(f"--{k.replace('_', '-')}")
        argv.extend(str(float(x)) for x in seq)

    for k in sorted(_STATE_LIST_INT & set(rc)):
        seq = rc[k]
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
            raise TypeError(f"merged[{k!r}] must be a list of ints, got {type(seq).__name__}")
        if not seq:
            raise ValueError(f"merged[{k!r}] must be non-empty.")
        argv.append(f"--{k.replace('_', '-')}")
        argv.extend(str(int(x)) for x in seq)

    if rc.get("debug"):
        argv.append("--debug")
    return argv


def _cli_to_state_optional(ns: argparse.Namespace) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if getattr(ns, "lambda_carry", None) is not None:
        extra["lambda_carry"] = float(ns.lambda_carry)
    if ns.lambda_sum_grid is not None:
        extra["lambda_sum_grid"] = [float(x) for x in ns.lambda_sum_grid]
    if ns.lambda_carry_grid is not None:
        extra["lambda_carry_grid"] = [float(x) for x in ns.lambda_carry_grid]
    if ns.verify_samples is not None:
        extra["verify_samples"] = int(ns.verify_samples)
    if ns.batch_size is not None:
        extra["batch_size"] = int(ns.batch_size)
    if ns.cvx_device is not None:
        extra["cvx_device"] = str(ns.cvx_device)
    if ns.cvx_grid_workers is not None:
        extra["cvx_grid_workers"] = int(ns.cvx_grid_workers)
    if ns.cvx_ovr_workers is not None:
        extra["cvx_ovr_workers"] = int(ns.cvx_ovr_workers)
    if ns.blas_threads is not None:
        extra["blas_threads"] = int(ns.blas_threads)
    for k in (
        "ste_sum_loss",
        "ste_carry_loss",
        "cvx_sum_loss",
        "cvx_carry_loss",
        "tf_objective",
        "ste_time_loss",
        "cvx_time_loss",
    ):
        v = getattr(ns, k)
        if v is not None:
            extra[k] = str(v)
    if ns.debug:
        extra["debug"] = True
    return extra


def build_command(
    *,
    dfa_config_path: Path,
    run_config_path: Path,
    cli: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any]]:
    dfa_raw = _load_json(dfa_config_path)
    run_raw = _unwrap_run_config(_load_json(run_config_path))
    base = _laststep_to_state_label_dict(run_raw)
    _, merged = _apply_dfa_config(dfa_raw, base)

    merged["run_mode"] = str(cli.run_mode)
    merged["pretrain_variant"] = str(cli.pretrain_variant)
    merged["finetune_variant"] = str(cli.finetune_variant)
    if str(cli.out_root).strip():
        merged["out_root"] = str(Path(cli.out_root).expanduser().resolve())
    if str(cli.output_json).strip():
        merged["output_json"] = str(Path(cli.output_json).expanduser().resolve())

    ol = _cli_to_state_optional(cli)
    for k, v in ol.items():
        merged[k] = v

    extra_keys = set(merged) - _ALL_EMITTED_KEYS
    if extra_keys:
        raise RuntimeError(f"internal: unhandled merged keys {sorted(extra_keys)}")

    argv = [sys.executable, str(BENCH), *_merge_to_argv(merged)]
    summary = {
        "dfa_config": str(dfa_config_path.resolve()),
        "run_config": str(run_config_path.resolve()),
        "dfa_spec": str(merged["dfa_spec"]),
        "run_mode": merged["run_mode"],
        "out_root": merged.get("out_root", ""),
    }
    return argv, summary


_ALL_EMITTED_KEYS = _STATE_SINGLE | _STATE_LIST_FLOAT | _STATE_LIST_INT | {"debug"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dfa_config", type=Path, required=True, help="JSON with dfa_spec (e.g. grid_cell.json).")
    ap.add_argument(
        "--run_config",
        type=Path,
        required=True,
        help="Last-step-shaped JSON: run_config.json from dfa_laststep, or metrics.json (unwraps run_config).",
    )
    ap.add_argument(
        "--run-mode",
        dest="run_mode",
        choices=["full", "pretrain_only", "finetune_only"],
        default="full",
        help="Forwarded to run_dfa_state_label_finetune_bench.",
    )
    ap.add_argument(
        "--pretrain-variant",
        dest="pretrain_variant",
        choices=["ste", "cvx"],
        default="ste",
        help="pretrain_only / full: which head writes weights.",
    )
    ap.add_argument(
        "--finetune-variant",
        dest="finetune_variant",
        choices=["ste_from_ste", "ste_from_cvx"],
        default="ste_from_ste",
        help="finetune_only / full: which checkpoint to load before STE finetune.",
    )
    ap.add_argument("--out-root", dest="out_root", type=str, default="", help="Required for pretrain_only / finetune_only.")
    ap.add_argument("--output-json", dest="output_json", type=str, default="", help="Optional extra metrics copy path.")
    ap.add_argument("--lambda-carry", dest="lambda_carry", type=float, default=None, help="Fixed label-head weight (bench default if omitted).")
    ap.add_argument("--lambda-sum-grid", dest="lambda_sum_grid", type=float, nargs="*", default=None, help="State-head sweep.")
    ap.add_argument("--lambda-carry-grid", dest="lambda_carry_grid", type=float, nargs="*", default=None, help="Alias of --lambda-sum-grid.")
    ap.add_argument("--verify-samples", dest="verify_samples", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--cvx-device", dest="cvx_device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    ap.add_argument("--cvx-grid-workers", dest="cvx_grid_workers", type=int, default=None)
    ap.add_argument("--cvx-ovr-workers", dest="cvx_ovr_workers", type=int, default=None)
    ap.add_argument("--blas-threads", dest="blas_threads", type=int, default=None)
    ap.add_argument("--ste-sum-loss", dest="ste_sum_loss", default=None)
    ap.add_argument("--ste-carry-loss", dest="ste_carry_loss", default=None)
    ap.add_argument("--cvx-sum-loss", dest="cvx_sum_loss", default=None)
    ap.add_argument("--cvx-carry-loss", dest="cvx_carry_loss", default=None)
    ap.add_argument("--tf-objective", dest="tf_objective", default=None)
    ap.add_argument("--ste-time-loss", dest="ste_time_loss", default=None)
    ap.add_argument("--cvx-time-loss", dest="cvx_time_loss", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Print argv and exit.")
    args = ap.parse_args()
    if args.lambda_sum_grid is not None and args.lambda_carry_grid is not None:
        raise SystemExit("Pass only one of --lambda-sum-grid and --lambda-carry-grid.")

    argv, summary = build_command(dfa_config_path=args.dfa_config, run_config_path=args.run_config, cli=args)
    print(json.dumps({"invoke": summary}, indent=2), flush=True)
    if args.dry_run:
        print(" ".join(argv), flush=True)
        return
    r = subprocess.run(argv, cwd=str(ATOMIC))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
