#!/usr/bin/env python3
"""Emit AR-only aggregate markdown tables per carry_hybrid sweep (joint / individual toks / mean-first-wrong)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STAGES: List[Tuple[str, str]] = [
    ("ste_pretrain", "STE pretrain (split A)"),
    ("cvx_from_ste_pretrain", "CVX from STE init"),
    ("ste_finetune_from_ste_pretrain_new_train", "STE finetune (from STE, split B)"),
    ("cvx_pretrain", "CVX pretrain (Gaussian)"),
    ("ste_finetune_from_cvx_pretrain_new_train", "STE finetune (from CVX, split B)"),
]

MODE_KEY = "autoregressive"


def _mean_std_list(vals: List[float]) -> Dict[str, Any]:
    a = np.array([float(x) for x in vals], dtype=np.float64)
    if a.size < 1:
        raise ValueError("mean_std: empty list")
    if a.size == 1:
        return {"mean": float(a[0]), "std": 0.0, "n": 1}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "n": int(a.size)}


def _float_dict_mean_std(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = _mean_std_list([float(v) for v in vals])
        else:
            out[k] = vals[0]
    return out


def _aggregate_eval_mode_payload(mode_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not mode_payloads:
        return {}
    id_metrics = _float_dict_mean_std([sp["id_metrics"] for sp in mode_payloads])
    ood0 = mode_payloads[0]["ood_eval"]
    ood_agg: Dict[str, Any] = {}
    for key in ood0:
        blocks = [sp["ood_eval"][key] for sp in mode_payloads]
        ood_agg[key] = {
            "n_digits": blocks[0]["n_digits"],
            "n_test": blocks[0]["n_test"],
            "metrics": _float_dict_mean_std([b["metrics"] for b in blocks]),
        }
    return {"id_metrics": id_metrics, "ood_eval": ood_agg}


def _aggregate_stage(stage_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stage_payloads:
        return {}
    return {
        "teacher_forcing": _aggregate_eval_mode_payload(
            [sp["teacher_forcing"] for sp in stage_payloads]
        ),
        "autoregressive": _aggregate_eval_mode_payload(
            [sp["autoregressive"] for sp in stage_payloads]
        ),
    }


def _aggregate_lambda_sweep(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        raise ValueError("no seed payloads for aggregation")
    base_grid = [float(e["lambda_carry"]) for e in seed_payloads[0]["lambda_sweep"]]
    for sp in seed_payloads[1:]:
        if [float(e["lambda_carry"]) for e in sp["lambda_sweep"]] != base_grid:
            raise ValueError("lambda_carry sweep order differs across seeds")
    out: List[Dict[str, Any]] = []
    for idx, lc in enumerate(base_grid):
        entries = [sp["lambda_sweep"][idx] for sp in seed_payloads]
        out.append(
            {
                "lambda_carry": float(lc),
                "ste_pretrain": _aggregate_stage([e["ste_pretrain"] for e in entries]),
                "cvx_from_ste_pretrain": _aggregate_stage(
                    [e["cvx_from_ste_pretrain"] for e in entries]
                ),
                "ste_finetune_from_ste_pretrain_new_train": _aggregate_stage(
                    [e["ste_finetune_from_ste_pretrain_new_train"] for e in entries]
                ),
                "cvx_pretrain": _aggregate_stage([e["cvx_pretrain"] for e in entries]),
                "ste_finetune_from_cvx_pretrain_new_train": _aggregate_stage(
                    [e["ste_finetune_from_cvx_pretrain_new_train"] for e in entries]
                ),
            }
        )
    return out


def _is_agg_block(v: Any) -> bool:
    return isinstance(v, dict) and "mean" in v and "std" in v and "n" in v


def _fmt_cell(v: Any, *, with_std: bool) -> str:
    if _is_agg_block(v):
        m = float(v["mean"])
        s = float(v["std"])
        n = int(v["n"])
        if math.isnan(m):
            return "nan"
        if with_std and n > 1:
            return f"{m:.6g}±{s:.6g}"
        return f"{m:.6g}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if math.isnan(x):
            return "nan"
        return f"{x:.6g}"
    raise TypeError(f"expected float or aggregate block, got {type(v).__name__}: {v!r}")


def _ood_keys_ordered(sweep_0: Dict[str, Any], stage_key: str) -> List[str]:
    ood = sweep_0[stage_key][MODE_KEY]["ood_eval"]
    return sorted(ood.keys(), key=lambda k: int(ood[k]["n_digits"]))


def _discover_carry_hybrid_dirs(sweep_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(sweep_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("carry_hybrid_"):
            continue
        if (p / "metrics.json").is_file():
            out.append(p)
    return out


def _load_sweep(root: Path) -> Tuple[List[Dict[str, Any]], str, int]:
    metrics_path = root / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "seeds" not in data:
        raise KeyError(f"{metrics_path}: missing top-level 'seeds'")

    n_seeds_cfg = len(data["seeds"])
    src: str

    agg = data.get("aggregate")
    if isinstance(agg, dict) and "lambda_sweep" in agg:
        sweep = agg["lambda_sweep"]
        src = "aggregate.lambda_sweep"
        n_seeds = int(agg.get("n_seeds", n_seeds_cfg))
    else:
        seed_payloads = [s for s in data["seeds"] if "lambda_sweep" in s]
        if not seed_payloads:
            raise ValueError(f"{metrics_path}: no seed entries with lambda_sweep")
        sweep = _aggregate_lambda_sweep(seed_payloads)
        src = "recomputed from seeds[].lambda_sweep"
        n_seeds = len(seed_payloads)

    if not sweep:
        raise ValueError(f"{metrics_path}: empty lambda_sweep")
    return sweep, src, n_seeds


def _joint_table(
    sweep: List[Dict[str, Any]],
    stage_key: str,
    stage_title: str,
    *,
    with_std: bool,
    limit_lambdas: Optional[int],
) -> str:
    sk0 = sweep[0]
    ood_keys = _ood_keys_ordered(sk0, stage_key)
    h = ["λ_carry", "ID:joint_tok"]
    for ok in ood_keys:
        nd = int(sk0[stage_key][MODE_KEY]["ood_eval"][ok]["n_digits"])
        h.append(f"OOD{nd}:joint_tok")
    lines = [f"### {stage_title}", "", "| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for i, e in enumerate(sweep):
        if limit_lambdas is not None and i >= limit_lambdas:
            break
        m = e[stage_key][MODE_KEY]
        row = [str(e["lambda_carry"]), _fmt_cell(m["id_metrics"]["joint_token_acc"], with_std=with_std)]
        for ok in ood_keys:
            row.append(_fmt_cell(m["ood_eval"][ok]["metrics"]["joint_token_acc"], with_std=with_std))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _individual_table(
    sweep: List[Dict[str, Any]],
    stage_key: str,
    stage_title: str,
    *,
    with_std: bool,
    limit_lambdas: Optional[int],
) -> str:
    sk0 = sweep[0]
    ood_keys = _ood_keys_ordered(sk0, stage_key)
    h = ["λ_carry", "ID:sum_tok", "ID:carry_tok"]
    for ok in ood_keys:
        nd = int(sk0[stage_key][MODE_KEY]["ood_eval"][ok]["n_digits"])
        h += [f"OOD{nd}:sum_tok", f"OOD{nd}:carry_tok"]
    lines = [f"### {stage_title}", "", "| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for i, e in enumerate(sweep):
        if limit_lambdas is not None and i >= limit_lambdas:
            break
        m = e[stage_key][MODE_KEY]
        row = [
            str(e["lambda_carry"]),
            _fmt_cell(m["id_metrics"]["sum_token_acc"], with_std=with_std),
            _fmt_cell(m["id_metrics"]["carry_token_acc"], with_std=with_std),
        ]
        for ok in ood_keys:
            om = m["ood_eval"][ok]["metrics"]
            row += [
                _fmt_cell(om["sum_token_acc"], with_std=with_std),
                _fmt_cell(om["carry_token_acc"], with_std=with_std),
            ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _mfw_table(
    sweep: List[Dict[str, Any]],
    stage_key: str,
    stage_title: str,
    *,
    with_std: bool,
    limit_lambdas: Optional[int],
) -> str:
    sk0 = sweep[0]
    ood_keys = _ood_keys_ordered(sk0, stage_key)
    h = ["λ_carry", "ID:1st_wrong_sum", "ID:1st_wrong_car"]
    for ok in ood_keys:
        nd = int(sk0[stage_key][MODE_KEY]["ood_eval"][ok]["n_digits"])
        h += [f"OOD{nd}:1st_wrong_sum", f"OOD{nd}:1st_wrong_car"]
    lines = [f"### {stage_title}", "", "| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for i, e in enumerate(sweep):
        if limit_lambdas is not None and i >= limit_lambdas:
            break
        m = e[stage_key][MODE_KEY]
        row = [
            str(e["lambda_carry"]),
            _fmt_cell(m["id_metrics"]["mean_first_wrong_sum_among_error_seq"], with_std=with_std),
            _fmt_cell(m["id_metrics"]["mean_first_wrong_carry_among_error_seq"], with_std=with_std),
        ]
        for ok in ood_keys:
            om = m["ood_eval"][ok]["metrics"]
            row += [
                _fmt_cell(om["mean_first_wrong_sum_among_error_seq"], with_std=with_std),
                _fmt_cell(om["mean_first_wrong_carry_among_error_seq"], with_std=with_std),
            ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _build_doc(
    run_name: str,
    sweep: List[Dict[str, Any]],
    src: str,
    n_seeds: int,
    *,
    which: str,
    with_std: bool,
    limit_lambdas: Optional[int],
) -> str:
    title_map = {
        "joint": ("joint_token_acc (AR aggregate)", _joint_table),
        "individual": ("sum_token_acc & carry_token_acc (AR aggregate)", _individual_table),
        "mfw": (
            "mean_first_wrong_sum / mean_first_wrong_carry among error seq (AR aggregate)",
            _mfw_table,
        ),
    }
    doc_title, table_fn = title_map[which]
    parts = [
        f"# {doc_title}",
        "",
        f"- Run: `{run_name}`",
        f"- Eval mode: **{MODE_KEY}**",
        f"- Cross-seed source: {src}",
        f"- Seeds: **{n_seeds}**",
    ]
    if with_std and n_seeds > 1:
        parts.append("- Cells: `mean±std` when multiple seeds; `mean` only when a single seed.")
    elif n_seeds <= 1:
        parts.append("- Cells: `mean` (single seed; std omitted).")
    parts.append("")
    for st_key, st_title in STAGES:
        parts.append(table_fn(sweep, st_key, st_title, with_std=with_std, limit_lambdas=limit_lambdas))
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep_root",
        type=Path,
        default=Path(__file__).resolve().parent / "sweep_results",
        help="Directory containing carry_hybrid_* folders",
    )
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Single carry_hybrid directory (default: all under sweep_root)",
    )
    ap.add_argument(
        "--mean_only",
        action="store_true",
        help="Show only mean (ignore std even with multiple seeds)",
    )
    ap.add_argument(
        "--limit_lambdas",
        type=int,
        default=None,
        help="Debug: only first N λ_carry rows per table",
    )
    args = ap.parse_args()

    sweep_root = args.sweep_root.expanduser().resolve()
    if args.run_dir is not None:
        dirs = [args.run_dir.expanduser().resolve()]
    else:
        dirs = _discover_carry_hybrid_dirs(sweep_root)

    if not dirs:
        raise SystemExit(f"no carry_hybrid_* with metrics.json under {sweep_root}")

    with_std = not args.mean_only

    out_map = {
        "joint": "AR_aggregate_joint_token_acc.md",
        "individual": "AR_aggregate_individual_token_accs.md",
        "mfw": "AR_aggregate_mean_first_wrong.md",
    }

    for d in dirs:
        sweep, src, n_seeds = _load_sweep(d)
        run_name = d.name
        for key, filename in out_map.items():
            body = _build_doc(
                run_name,
                sweep,
                src,
                n_seeds,
                which=key,
                with_std=with_std,
                limit_lambdas=args.limit_lambdas,
            )
            out_path = d / filename
            out_path.write_text(body, encoding="utf-8")
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
