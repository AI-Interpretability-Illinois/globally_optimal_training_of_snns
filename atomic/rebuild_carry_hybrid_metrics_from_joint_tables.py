#!/usr/bin/env python3
"""
Rebuild carry_hybrid `metrics.json` (and per-seed `seed_*/metrics.json`) from existing
`joint_token_acc_matrix_teacher_forcing_seed*.txt` and
`joint_token_acc_matrix_autoregressive_seed*.txt` files.

Only `joint_token_acc` is recovered from the tables. All other scalar metrics are set
to NaN so downstream code does not silently misread invented values.

Uses the same `run_config` layout as `run_arithmetic_add_carry_finetune_bench.py`
(default: copy from a reference `run_config.json`, patch `arith_base` and `seeds`).
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SECTION_RE = re.compile(r"^### λ_carry = ([0-9.]+)\s*$")
ROW_RE = re.compile(r"^\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")

METHOD_TO_STAGE = {
    "STE Pretrain": "ste_pretrain",
    "CVX from STE Pretrain": "cvx_from_ste_pretrain",
    "STE Finetune (from STE)": "ste_finetune_from_ste_pretrain_new_train",
    "CVX Pretrain": "cvx_pretrain",
    "STE Finetune (from CVX)": "ste_finetune_from_cvx_pretrain_new_train",
}

METRIC_KEYS = [
    "sum_token_acc",
    "carry_token_acc",
    "joint_token_acc",
    "joint_seq_acc",
    "sum_seq_acc",
    "carry_seq_acc",
    "mean_first_wrong_sum_among_error_seq",
    "mean_first_wrong_carry_among_error_seq",
]

_STAGES = [
    "ste_pretrain",
    "cvx_from_ste_pretrain",
    "ste_finetune_from_ste_pretrain_new_train",
    "cvx_pretrain",
    "ste_finetune_from_cvx_pretrain_new_train",
]


def _norm_method(cell: str) -> str:
    return cell.strip().replace("*", "").strip()


def _metrics_from_joint(joint: float) -> Dict[str, float]:
    z = float("nan")
    return {
        "sum_token_acc": z,
        "carry_token_acc": z,
        "joint_token_acc": float(joint),
        "joint_seq_acc": z,
        "sum_seq_acc": z,
        "carry_seq_acc": z,
        "mean_first_wrong_sum_among_error_seq": z,
        "mean_first_wrong_carry_among_error_seq": z,
    }


def _eval_block(
    id_j: float,
    ood10: float,
    ood25: float,
    ood50: float,
    *,
    n_test_ood: int,
) -> Dict[str, Any]:
    return {
        "id_metrics": _metrics_from_joint(id_j),
        "ood_eval": {
            "n_digits_10": {"n_digits": 10, "n_test": int(n_test_ood), "metrics": _metrics_from_joint(ood10)},
            "n_digits_25": {"n_digits": 25, "n_test": int(n_test_ood), "metrics": _metrics_from_joint(ood25)},
            "n_digits_50": {"n_digits": 50, "n_test": int(n_test_ood), "metrics": _metrics_from_joint(ood50)},
        },
    }


def _parse_joint_table(path: Path) -> List[Tuple[float, Dict[str, Tuple[float, float, float, float]]]]:
    """
    Returns list of (lambda_carry, dict stage_key -> (id, ood10, ood25, ood50)).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: List[Tuple[float, Dict[str, Tuple[float, float, float, float]]]] = []
    i = 0
    cur_lc: float | None = None
    cur_block: Dict[str, Tuple[float, float, float, float]] | None = None
    in_table = False

    def flush() -> None:
        nonlocal cur_lc, cur_block, in_table
        if cur_lc is not None and cur_block is not None:
            if len(cur_block) != len(METHOD_TO_STAGE):
                raise ValueError(
                    f"{path}: expected {len(METHOD_TO_STAGE)} methods for λ_carry={cur_lc}, got {sorted(cur_block.keys())!r}"
                )
            out.append((float(cur_lc), cur_block))
        cur_lc = None
        cur_block = None
        in_table = False

    while i < len(lines):
        msec = SECTION_RE.match(lines[i])
        if msec:
            flush()
            cur_lc = float(msec.group(1))
            cur_block = {}
            in_table = False
            i += 1
            continue
        line = lines[i]
        if cur_lc is not None and line.strip().startswith("|") and "Method" in line:
            in_table = True
            i += 1
            continue
        if cur_lc is not None and in_table and line.strip().startswith("|") and "---" in line:
            i += 1
            continue
        if cur_lc is not None and in_table:
            if re.match(r"^\|[\s:—\-]+\|", line):
                i += 1
                continue
            rm = ROW_RE.match(line)
            if rm:
                raw_m, id_s, o10, o25, o50 = rm.group(1), rm.group(2), rm.group(3), rm.group(4), rm.group(5)
                stage = METHOD_TO_STAGE[_norm_method(raw_m)]
                if stage in cur_block:
                    raise ValueError(f"{path}: duplicate method row {raw_m!r} for λ_carry={cur_lc}")
                cur_block[stage] = (float(id_s), float(o10), float(o25), float(o50))
            elif line.strip() == "" or line.startswith("###"):
                in_table = False
        i += 1
    flush()
    if not out:
        raise ValueError(f"{path}: no λ_carry sections parsed")
    return out


def _mean_std_list(vals: List[float]) -> Dict[str, float]:
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
        "teacher_forcing": _aggregate_eval_mode_payload([sp["teacher_forcing"] for sp in stage_payloads]),
        "autoregressive": _aggregate_eval_mode_payload([sp["autoregressive"] for sp in stage_payloads]),
    }


def _aggregate_lambda_sweep(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
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
                "cvx_from_ste_pretrain": _aggregate_stage([e["cvx_from_ste_pretrain"] for e in entries]),
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


def _merge_tf_ar(
    tf_rows: List[Tuple[float, Dict[str, Tuple[float, float, float, float]]]],
    ar_rows: List[Tuple[float, Dict[str, Tuple[float, float, float, float]]]],
    *,
    n_test_ood: int,
) -> List[Dict[str, Any]]:
    if len(tf_rows) != len(ar_rows):
        raise ValueError(f"TF has {len(tf_rows)} λ blocks, AR has {len(ar_rows)}")
    lambda_sweep: List[Dict[str, Any]] = []
    nan_diag = {"primal_value": float("nan"), "dual_value": float("nan"), "gap": float("nan")}
    empty_sel: Dict[str, Any] = {"reconstructed_from_joint_tables_only": True}

    for (lc_tf, d_tf), (lc_ar, d_ar) in zip(tf_rows, ar_rows):
        if float(lc_tf) != float(lc_ar):
            raise ValueError(f"λ_carry mismatch: TF {lc_tf} vs AR {lc_ar}")
        if set(d_tf.keys()) != set(d_ar.keys()):
            raise ValueError(f"method set mismatch at λ={lc_tf}")
        entry: Dict[str, Any] = {"lambda_carry": float(lc_tf)}
        for st in _STAGES:
            id_tf, o10_tf, o25_tf, o50_tf = d_tf[st]
            id_ar, o10_ar, o25_ar, o50_ar = d_ar[st]
            tf_eval = _eval_block(id_tf, o10_tf, o25_tf, o50_tf, n_test_ood=n_test_ood)
            ar_eval = _eval_block(id_ar, o10_ar, o25_ar, o50_ar, n_test_ood=n_test_ood)

            if st in ("cvx_from_ste_pretrain", "cvx_pretrain"):
                entry[st] = {
                    "selected_params": deepcopy(empty_sel),
                    "teacher_forcing_pretrain_test_metrics": {k: tf_eval["id_metrics"][k] for k in METRIC_KEYS},
                    "diagnostics": deepcopy(nan_diag),
                    "teacher_forcing": tf_eval,
                    "autoregressive": ar_eval,
                }
            else:
                entry[st] = {
                    "selected_params": deepcopy(empty_sel),
                    "teacher_forcing": tf_eval,
                    "autoregressive": ar_eval,
                }
        lambda_sweep.append(entry)
    return lambda_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="carry_hybrid_* directory containing seed_* subfolders",
    )
    ap.add_argument(
        "--reference_run_config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sweep_results"
        / "carry_hybrid_b3_d5_20260424_031252"
        / "run_config.json",
        help="Template run_config.json (same hyperparams as the original sweep except arith_base).",
    )
    ap.add_argument("--arith_base", type=int, default=0, help="If >0, override run_config arith_base.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    seed_dirs = sorted(run_dir.glob("seed_*"), key=lambda p: p.name)
    if not seed_dirs:
        raise SystemExit(f"no seed_* under {run_dir}")

    local_cfg = run_dir / "run_config.json"
    if local_cfg.is_file():
        run_config = json.loads(local_cfg.read_text(encoding="utf-8"))
    else:
        ref_cfg_path = args.reference_run_config.expanduser().resolve()
        if not ref_cfg_path.is_file():
            raise SystemExit(f"reference run_config missing: {ref_cfg_path}")
        run_config = json.loads(ref_cfg_path.read_text(encoding="utf-8"))

    seeds: List[int] = []
    seed_payloads: List[Dict[str, Any]] = []

    for sdir in seed_dirs:
        m = re.match(r"^seed_(\d+)$", sdir.name)
        if not m:
            continue
        seed = int(m.group(1))
        tf_path = sdir / f"joint_token_acc_matrix_teacher_forcing_seed{seed}.txt"
        ar_path = sdir / f"joint_token_acc_matrix_autoregressive_seed{seed}.txt"
        if not tf_path.is_file() or not ar_path.is_file():
            print(
                f"skip {sdir.name}: missing {tf_path.name} or {ar_path.name}",
                flush=True,
            )
            continue

        tf_rows = _parse_joint_table(tf_path)
        ar_rows = _parse_joint_table(ar_path)
        n_ood = int(run_config["n_test_ood"])
        lambda_sweep = _merge_tf_ar(tf_rows, ar_rows, n_test_ood=n_ood)

        pre_seed = seed
        ft_seed = seed + int(run_config["finetune_seed_offset"])
        eval_seed = seed + int(run_config["eval_seed_offset"])
        seed_payload = {
            "seed": seed,
            "split_seeds": {
                "pretrain_train": pre_seed + 11,
                "pretrain_val": pre_seed + 29,
                "finetune_train": ft_seed + 11,
                "finetune_val": ft_seed + 29,
                "eval_test": eval_seed + 47,
            },
            "lambda_sweep": lambda_sweep,
            "reconstruction_note": (
                "Rebuilt from joint_token_acc matrix .txt files; non-joint metrics are NaN; "
                "selected_params and CVX diagnostics are placeholders."
            ),
        }
        seeds.append(seed)
        seed_payloads.append(seed_payload)
        out_seed_json = sdir / "metrics.json"
        out_seed_json.write_text(json.dumps(seed_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"wrote {out_seed_json}", flush=True)

    if not seeds:
        raise SystemExit(f"no complete seed folders (need TF+AR joint tables) under {run_dir}")
    run_config["seeds"] = seeds
    if int(args.arith_base) > 0:
        run_config["arith_base"] = int(args.arith_base)
    elif "carry_hybrid_b" in run_dir.name:
        mb = re.search(r"_b(\d+)_", run_dir.name)
        if mb:
            run_config["arith_base"] = int(mb.group(1))

    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'run_config.json'}", flush=True)

    root_payload = {
        "run_config": run_config,
        "out_root": str(run_dir),
        "seeds": seed_payloads,
        "aggregate": {
            "n_seeds": len(seed_payloads),
            "lambda_sweep": _aggregate_lambda_sweep(seed_payloads),
        },
        "reconstruction_note": (
            "Root metrics.json rebuilt from joint-token tables only; see per-seed reconstruction_note."
        ),
    }
    (run_dir / "metrics.json").write_text(json.dumps(root_payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
