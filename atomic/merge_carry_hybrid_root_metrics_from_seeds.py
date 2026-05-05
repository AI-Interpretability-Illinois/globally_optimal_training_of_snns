#!/usr/bin/env python3
"""Rebuild carry_hybrid root `metrics.json` + `aggregate` from all `seed_*/metrics.json` files.

Uses finite values only per scalar when aggregating, so a joint-table-rebuilt seed (NaN for
non-joint metrics) can be merged with a full bench seed without turning the mean into NaN.
If every seed is non-finite for a metric, raises (no silent fallback).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _mean_std_list_finite(vals: List[float], *, ctx: str) -> Dict[str, Any]:
    if not vals:
        raise ValueError(f"{ctx}: no finite values across seeds")
    a = np.array(vals, dtype=np.float64)
    if a.size == 1:
        return {"mean": float(a[0]), "std": 0.0, "n": 1}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": int(a.size)}


def _scalar_finite_candidates(raw: List[Any], *, ctx: str) -> List[float]:
    vals: List[float] = []
    for v in raw:
        if isinstance(v, bool):
            raise TypeError(f"{ctx}: bool value not allowed: {v!r}")
        if isinstance(v, (int, float)):
            x = float(v)
            if math.isfinite(x):
                vals.append(x)
        elif v is None:
            raise TypeError(f"{ctx}: None not allowed")
        else:
            raise TypeError(f"{ctx}: expected int/float, got {type(v).__name__}")
    return vals


def _float_dict_mean_std(dicts: List[Dict[str, Any]], *, ctx_prefix: str) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    for d in dicts[1:]:
        if d.keys() != keys:
            raise ValueError(f"{ctx_prefix}: metric key set mismatch {keys!r} vs {d.keys()!r}")
    out: Dict[str, Any] = {}
    for k in keys:
        ctx = f"{ctx_prefix}.{k}"
        cands = _scalar_finite_candidates([d[k] for d in dicts], ctx=ctx)
        out[k] = _mean_std_list_finite(cands, ctx=ctx)
    return out


def _aggregate_eval_mode_payload(
    mode_payloads: List[Dict[str, Any]], *, ctx_prefix: str
) -> Dict[str, Any]:
    if not mode_payloads:
        return {}
    ctx0 = f"{ctx_prefix}.id_metrics"
    id_metrics = _float_dict_mean_std(
        [sp["id_metrics"] for sp in mode_payloads], ctx_prefix=ctx0
    )
    ood0 = mode_payloads[0]["ood_eval"]
    for sp in mode_payloads[1:]:
        if sp["ood_eval"].keys() != ood0.keys():
            raise ValueError(f"{ctx_prefix}: ood_eval keys mismatch")
    ood_agg: Dict[str, Any] = {}
    for key in ood0:
        blocks = [sp["ood_eval"][key] for sp in mode_payloads]
        for b in blocks:
            if b["n_digits"] != blocks[0]["n_digits"] or b["n_test"] != blocks[0]["n_test"]:
                raise ValueError(f"{ctx_prefix}: ood {key} n_digits/n_test mismatch")
        ood_agg[key] = {
            "n_digits": blocks[0]["n_digits"],
            "n_test": blocks[0]["n_test"],
            "metrics": _float_dict_mean_std(
                [b["metrics"] for b in blocks], ctx_prefix=f"{ctx_prefix}.ood.{key}.metrics"
            ),
        }
    return {"id_metrics": id_metrics, "ood_eval": ood_agg}


def _aggregate_stage(stage_payloads: List[Dict[str, Any]], *, ctx_prefix: str) -> Dict[str, Any]:
    if not stage_payloads:
        return {}
    return {
        "teacher_forcing": _aggregate_eval_mode_payload(
            [sp["teacher_forcing"] for sp in stage_payloads],
            ctx_prefix=f"{ctx_prefix}.teacher_forcing",
        ),
        "autoregressive": _aggregate_eval_mode_payload(
            [sp["autoregressive"] for sp in stage_payloads],
            ctx_prefix=f"{ctx_prefix}.autoregressive",
        ),
    }


def _aggregate_lambda_sweep_finite(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        raise ValueError("no seed payloads")
    base_grid = [float(e["lambda_carry"]) for e in seed_payloads[0]["lambda_sweep"]]
    for si, sp in enumerate(seed_payloads[1:], start=1):
        g = [float(e["lambda_carry"]) for e in sp["lambda_sweep"]]
        if g != base_grid:
            raise ValueError(f"lambda_carry sweep differs: seed 0 vs seed index {si}")
    out: List[Dict[str, Any]] = []
    for idx, lc in enumerate(base_grid):
        entries = [sp["lambda_sweep"][idx] for sp in seed_payloads]
        px = f"lambda_sweep[λ={lc}]"
        out.append(
            {
                "lambda_carry": float(lc),
                "ste_pretrain": _aggregate_stage(
                    [e["ste_pretrain"] for e in entries], ctx_prefix=f"{px}.ste_pretrain"
                ),
                "cvx_from_ste_pretrain": _aggregate_stage(
                    [e["cvx_from_ste_pretrain"] for e in entries],
                    ctx_prefix=f"{px}.cvx_from_ste_pretrain",
                ),
                "ste_finetune_from_ste_pretrain_new_train": _aggregate_stage(
                    [e["ste_finetune_from_ste_pretrain_new_train"] for e in entries],
                    ctx_prefix=f"{px}.ste_finetune_from_ste_pretrain_new_train",
                ),
                "cvx_pretrain": _aggregate_stage(
                    [e["cvx_pretrain"] for e in entries], ctx_prefix=f"{px}.cvx_pretrain"
                ),
                "ste_finetune_from_cvx_pretrain_new_train": _aggregate_stage(
                    [e["ste_finetune_from_cvx_pretrain_new_train"] for e in entries],
                    ctx_prefix=f"{px}.ste_finetune_from_cvx_pretrain_new_train",
                ),
            }
        )
    return out


def _discover_seed_jsons(run_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(run_dir.glob("seed_*")):
        if not p.is_dir():
            continue
        m = re.match(r"^seed_(\d+)$", p.name)
        if not m:
            continue
        mj = p / "metrics.json"
        if mj.is_file():
            paths.append(mj)
    if not paths:
        raise SystemExit(f"no seed_*/metrics.json under {run_dir}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="carry_hybrid_* directory containing seed_*/metrics.json",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    seed_paths = _discover_seed_jsons(run_dir)
    seed_payloads: List[Dict[str, Any]] = []
    seed_ids: List[int] = []
    for mj in seed_paths:
        m = re.match(r"^seed_(\d+)$", mj.parent.name)
        if not m:
            raise RuntimeError(f"unexpected path {mj}")
        seed_ids.append(int(m.group(1)))
        seed_payloads.append(json.loads(mj.read_text(encoding="utf-8")))

    cfg_path = run_dir / "run_config.json"
    if cfg_path.is_file():
        run_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        root_m = run_dir / "metrics.json"
        if not root_m.is_file():
            raise SystemExit(f"need {cfg_path} or {root_m} for run_config template")
        run_config = json.loads(root_m.read_text(encoding="utf-8"))["run_config"]

    run_config["seeds"] = sorted(seed_ids)

    root_payload: Dict[str, Any] = {
        "run_config": run_config,
        "out_root": str(run_dir),
        "seeds": seed_payloads,
        "aggregate": {
            "n_seeds": len(seed_payloads),
            "lambda_sweep": _aggregate_lambda_sweep_finite(seed_payloads),
        },
    }

    cfg_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(root_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {run_dir / 'metrics.json'} and {cfg_path} with seeds {run_config['seeds']}")


if __name__ == "__main__":
    main()
