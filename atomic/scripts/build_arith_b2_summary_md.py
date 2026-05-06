#!/usr/bin/env python3
"""
Render one ``summary_all_seeds.md`` per matching sweep directory under
``atomic/sweep_results/``.

Match: ``arithmetic_add_b2_len_gen_L3_traind5_K2_2026*`` (the "K2" sibling of the K4
length-generalization runs).

For each match we read::
    <dir>/summary_all_seeds.json  -> list[ {seed, ood_eval{n_digits_X{ste,cvx{...}}}, ste_sel, cvx_sel} ]
    <dir>/run_config.json         -> hyperparameter dump

and write ``<dir>/summary_all_seeds.md`` containing:
- a header with relevant config knobs,
- a per-seed table of selected hyperparameters,
- **In-distribution (ID test)** — when every ``stages[*][vk]`` has ``id_metrics``: one combined table
  per metric type listing **all** pipeline variants (``vk`` in run order), seed rows + mean ± std per variant.
- one block per OOD ``n_digits_X`` (legacy top-level ``ood_eval``) with **all** variants (not only the
  ``ste`` / ``cvx`` shorthand), same layout:
    * "top-level metrics" table (rows: seed × variant, plus mean ± std across seeds per variant),
    * "per 5-timestep block token_acc" table (same).

Numbers are 4-decimal floats; counts stay as ints; std is ``ddof=1`` (sample std).
Floats that are missing (e.g. lr=null) render as ``-``.
Aggregate rows: **sample mean ± sample std (ddof=1) across seeds** for each metric; with a
single seed the std is omitted (undefined). Derived ``sum_token_acc`` / ``final_carry_acc``
must be present for every seed or the script raises (so aggregates are never over a ragged subset).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_PATTERN = "arithmetic_add_b*_len_gen_*"


def _fmt_float(x: Any, digits: int = 4) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    return f"{int(x)}"


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Sample mean and **sample** std (``ddof=1``) across ``values``.

    With a single value, std is ``None`` (undefined); do not display ``± 0``.
    """
    vs = [float(v) for v in values if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], None
    n = len(vs)
    mu = sum(vs) / n
    var = sum((v - mu) ** 2 for v in vs) / (n - 1)
    return mu, math.sqrt(var)


def _require_sum_carry_metrics(
    sum_only: Optional[float],
    final_carry: Optional[float],
    *,
    seed: Any,
    ctx: str,
) -> Tuple[float, float]:
    """Require derivable sum-pos and carry metrics so each seed contributes one float to aggregates."""
    if sum_only is None or final_carry is None:
        raise ValueError(
            f"{ctx}: need sum_token_acc and final_carry for seed={seed!r}; "
            f"got sum_only={sum_only!r} final_carry={final_carry!r}"
        )
    return float(sum_only), float(final_carry)


def _fmt_mean_pm_std(mu: Optional[float], sd: Optional[float], *, digits: int = 4) -> str:
    """Format aggregate across seeds: ``mean``, or ``mean ± std`` when ``len(seeds) > 1``."""
    if mu is None:
        return "-"
    if sd is None:
        return _fmt_float(mu, digits=digits)
    return f"{_fmt_float(mu, digits=digits)} ± {_fmt_float(sd, digits=digits)}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Sequence[str]) -> str:
    if len(headers) != len(aligns):
        raise ValueError("headers/aligns length mismatch")
    sep = []
    for a in aligns:
        if a == "r":
            sep.append("---:")
        elif a == "l":
            sep.append(":---")
        elif a == "c":
            sep.append(":---:")
        else:
            raise ValueError(f"bad align {a!r}")
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(sep) + "|")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row width mismatch: {row}")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _config_summary(config: Dict[str, Any], summary: List[Dict[str, Any]]) -> str:
    seeds = [int(s["seed"]) for s in summary]
    bits = [
        ("base", config.get("arith_base")),
        ("op", config.get("arith_op")),
        ("L", config.get("L")),
        ("P_rec", config.get("P_rec")),
        ("P_last", config.get("P_last")),
        ("K_parallel", config.get("K_parallel")),
        ("n_train", config.get("n_train")),
        ("n_val", config.get("n_val")),
        ("ste_epochs", config.get("ste_epochs")),
        ("cvx_epochs", config.get("cvx_epochs")),
        ("loss_type", config.get("loss_type")),
        ("cvx_method", config.get("cvx_method")),
    ]
    parts = []
    for k, v in bits:
        if v is None:
            continue
        parts.append(f"**{k}**={v}")
    parts.append(f"**seeds**={seeds}")
    return ", ".join(parts)


def _selected_hp_table(summary: List[Dict[str, Any]]) -> str:
    rows: List[List[str]] = []
    for entry in summary:
        ste_sel = entry.get("ste_sel") or {}
        cvx_sel = entry.get("cvx_sel") or {}
        rows.append([
            _fmt_int(entry["seed"]),
            _fmt_float(ste_sel.get("lr"), digits=6),
            _fmt_float(ste_sel.get("beta"), digits=6),
            _fmt_float(cvx_sel.get("lr"), digits=6),
            _fmt_float(cvx_sel.get("beta"), digits=6),
            _fmt_float(cvx_sel.get("bias"), digits=6),
        ])
    return _md_table(
        headers=["seed", "ste lr", "ste beta", "cvx lr", "cvx beta", "cvx bias"],
        rows=rows,
        aligns=["r", "r", "r", "r", "r", "r"],
    )


def _ood_keys_sorted(summary: List[Dict[str, Any]]) -> List[str]:
    """Return ``n_digits_X`` keys ordered by integer X."""
    if not summary:
        return []
    keys = list((summary[0].get("ood_eval") or {}).keys())
    for entry in summary[1:]:
        if list((entry.get("ood_eval") or {}).keys()) != keys:
            raise ValueError("OOD setting keys differ across seeds")

    def _ord(k: str) -> int:
        if not k.startswith("n_digits_"):
            raise ValueError(f"Unexpected ood key {k!r}; expected 'n_digits_X'.")
        return int(k.split("_")[-1])

    return sorted(keys, key=_ord)


def _stage_keys_ordered(summary: List[Dict[str, Any]]) -> List[str]:
    """Preserve order from the first seed; require identical stage key sets across seeds."""
    if not summary:
        return []
    st0 = summary[0].get("stages")
    if not isinstance(st0, dict) or not st0:
        return []
    keys = list(st0.keys())
    key_set = set(keys)
    for entry in summary[1:]:
        st = entry["stages"]
        if not isinstance(st, dict):
            raise TypeError(f"seed {entry['seed']}: stages must be dict, got {type(st)}")
        if set(st.keys()) != key_set:
            raise ValueError(
                f"stage keys differ: seed {summary[0]['seed']} {sorted(key_set)} vs "
                f"seed {entry['seed']} {sorted(st.keys())}"
            )
    return keys


def _pipeline_variant_keys(summary: List[Dict[str, Any]]) -> List[str]:
    """OOD / combined tables: full ``stages`` key order, or legacy ``ste`` / ``cvx`` only."""
    keys = _stage_keys_ordered(summary)
    if keys:
        return keys
    return ["ste", "cvx"]


def _can_render_id_tables(summary: List[Dict[str, Any]]) -> bool:
    keys = _stage_keys_ordered(summary)
    if not keys:
        return False
    st0 = summary[0]["stages"]
    for vk in keys:
        if not isinstance(st0[vk].get("id_metrics"), dict):
            return False
    return True


def _top_level_table(summary: List[Dict[str, Any]], ood_key: str) -> str:
    """Per-seed (seed × variant) top-level metrics, plus mean ± std across seeds per variant.

    Columns include sum-only metrics so we can compare directly to the carry/state
    hybrid bench's ``sum_token_acc`` (the carry head's MSD-carry-out token is
    surfaced separately as ``final_carry_acc``).
    """
    headers = [
        "seed",
        "variant",
        "T",
        "token_acc",
        "sum_token_acc",
        "final_carry_acc",
        "seq_acc",
        "n_seq_w_err",
        "mean_first_wrong",
        "std_first_wrong",
    ]
    variant_keys = _pipeline_variant_keys(summary)
    rows: List[List[str]] = []

    accum: Dict[str, Dict[str, List[float]]] = {
        vk: {
            "token_acc": [],
            "sum_token_acc": [],
            "final_carry_acc": [],
            "seq_acc": [],
            "n_seq_w_err": [],
            "mean_first_wrong": [],
            "std_first_wrong": [],
        }
        for vk in variant_keys
    }

    for entry in summary:
        block = entry["ood_eval"][ood_key]
        for vk in variant_keys:
            sub = block[vk]
            sum_only, _sum_seq, final_carry = _extract_sum_only_metrics(sub)
            sum_only, final_carry = _require_sum_carry_metrics(
                sum_only,
                final_carry,
                seed=entry["seed"],
                ctx=f"ood_eval[{ood_key!r}] variant={vk!r}",
            )
            rows.append([
                _fmt_int(entry["seed"]),
                vk,
                _fmt_int(sub["n_timesteps"]),
                _fmt_float(sub["token_acc"]),
                _fmt_float(sum_only),
                _fmt_float(final_carry),
                _fmt_float(sub["seq_acc"]),
                _fmt_int(sub["n_sequences_with_any_error"]),
                _fmt_float(sub["mean_first_wrong_timestep_among_wrong"]),
                _fmt_float(sub["std_first_wrong_timestep_among_wrong"]),
            ])
            accum[vk]["token_acc"].append(float(sub["token_acc"]))
            accum[vk]["sum_token_acc"].append(sum_only)
            accum[vk]["final_carry_acc"].append(final_carry)
            accum[vk]["seq_acc"].append(float(sub["seq_acc"]))
            accum[vk]["n_seq_w_err"].append(float(sub["n_sequences_with_any_error"]))
            accum[vk]["mean_first_wrong"].append(float(sub["mean_first_wrong_timestep_among_wrong"]))
            accum[vk]["std_first_wrong"].append(float(sub["std_first_wrong_timestep_among_wrong"]))

    for vk in variant_keys:
        n_steps = int(summary[0]["ood_eval"][ood_key][vk]["n_timesteps"])
        mu_tok, sd_tok = _mean_std(accum[vk]["token_acc"])
        mu_sum, sd_sum = _mean_std(accum[vk]["sum_token_acc"])
        mu_fc, sd_fc = _mean_std(accum[vk]["final_carry_acc"])
        mu_seq, sd_seq = _mean_std(accum[vk]["seq_acc"])
        mu_err, sd_err = _mean_std(accum[vk]["n_seq_w_err"])
        mu_fw, sd_fw = _mean_std(accum[vk]["mean_first_wrong"])
        mu_fws, sd_fws = _mean_std(accum[vk]["std_first_wrong"])
        rows.append([
            "**mean ± std**",
            f"**{vk}**",
            _fmt_int(n_steps),
            _fmt_mean_pm_std(mu_tok, sd_tok),
            _fmt_mean_pm_std(mu_sum, sd_sum),
            _fmt_mean_pm_std(mu_fc, sd_fc),
            _fmt_mean_pm_std(mu_seq, sd_seq),
            _fmt_mean_pm_std(mu_err, sd_err, digits=1),
            _fmt_mean_pm_std(mu_fw, sd_fw),
            _fmt_mean_pm_std(mu_fws, sd_fws),
        ])
    return _md_table(
        headers=headers,
        rows=rows,
        aligns=["r", "l", "r", "r", "r", "r", "r", "r", "r", "r"],
    )


def _block_t_range(b: Dict[str, Any]) -> Tuple[int, int]:
    """Tolerate both schemas:
    - older runs:  ``timestep_start`` / ``timestep_end_inclusive``
    - newer runs:  ``timestep_start_0based`` / ``timestep_end_0based_inclusive``
    """
    if "timestep_start" in b and "timestep_end_inclusive" in b:
        return int(b["timestep_start"]), int(b["timestep_end_inclusive"])
    if "timestep_start_0based" in b and "timestep_end_0based_inclusive" in b:
        return int(b["timestep_start_0based"]), int(b["timestep_end_0based_inclusive"])
    raise KeyError(f"Per-block dict has no timestep_start/end keys; got keys={list(b.keys())}")


def _extract_sum_only_metrics(method_block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(sum_token_acc, sum_seq_acc, final_carry_token_acc)`` for one
    OOD method block (i.e. ``summary[seed].ood_eval[n_digits_X].{ste|cvx}``).

    "Sum only" means restricted to per-column sum-digit positions (timesteps
    ``0..n_digits-1``); MSD carry-out at the final timestep is reported separately.

    Two paths:
    1. Newer schema includes a ``rule_context`` block — read the precomputed
       ``sum_token_acc_over_sum_positions`` / ``sum_seq_acc_over_sum_positions`` /
       ``final_carry_token_acc`` directly.
    2. Older schema: derive from per-block ``token_acc`` if and only if the
       last per-block entry holds exactly the 1-token MSD carry-out (``n_total == n_digits + 1``
       and ``last_block`` covers a single timestep at index ``n_total - 1``).
       In that case::
           final_carry_token_acc = last_block.token_acc
           sum_token_acc = (token_acc * n_total - final_carry_token_acc) / (n_total - 1)

       Sequence accuracy on sum-only positions can't be derived from these
       aggregates (it needs the raw matches) → returns None.
    """
    rc = method_block.get("rule_context")
    if isinstance(rc, dict):
        return (
            float(rc.get("sum_token_acc_over_sum_positions")) if rc.get("sum_token_acc_over_sum_positions") is not None else None,
            float(rc.get("sum_seq_acc_over_sum_positions")) if rc.get("sum_seq_acc_over_sum_positions") is not None else None,
            float(rc.get("final_carry_token_acc")) if rc.get("final_carry_token_acc") is not None else None,
        )

    n_total_raw = method_block.get("n_timesteps")
    blocks = method_block.get("per_five_timestep_blocks")
    overall_acc = method_block.get("token_acc")
    if n_total_raw is None or not isinstance(blocks, list) or not blocks or overall_acc is None:
        return None, None, None
    n_total = int(n_total_raw)
    if n_total < 2:
        return None, None, None
    last = blocks[-1]
    last_t0, last_t1 = _block_t_range(last)
    if last_t0 != last_t1 or last_t1 != n_total - 1:
        # Last block is not a 1-token MSD-carry block — refuse to fabricate.
        return None, None, None
    final_carry = float(last["token_acc"])
    sum_only = float((float(overall_acc) * n_total - final_carry) / (n_total - 1))
    return sum_only, None, final_carry


def _block_columns(block_list: Sequence[Dict[str, Any]]) -> List[str]:
    """Format ``block i (t<a>-t<b>)`` column headers in input order."""
    cols: List[str] = []
    for b in block_list:
        idx = int(b["block_index"])
        t0, t1 = _block_t_range(b)
        if t0 == t1:
            cols.append(f"blk {idx} (t{t0})")
        else:
            cols.append(f"blk {idx} (t{t0}-t{t1})")
    return cols


def _block_table(summary: List[Dict[str, Any]], ood_key: str) -> str:
    """Per-seed per-variant per-block token_acc, mean/std across seeds per variant."""
    variant_keys = _pipeline_variant_keys(summary)
    first_blocks = summary[0]["ood_eval"][ood_key][variant_keys[0]]["per_five_timestep_blocks"]
    n_blocks = len(first_blocks)
    headers = ["seed", "variant", *_block_columns(first_blocks)]
    rows: List[List[str]] = []

    accum: Dict[str, List[List[float]]] = {vk: [[] for _ in range(n_blocks)] for vk in variant_keys}

    for entry in summary:
        block = entry["ood_eval"][ood_key]
        for vk in variant_keys:
            blocks = block[vk]["per_five_timestep_blocks"]
            if len(blocks) != n_blocks:
                raise ValueError(
                    f"per_five_timestep_blocks length mismatch for seed={entry['seed']} ood={ood_key} variant={vk!r}"
                )
            row = [_fmt_int(entry["seed"]), vk]
            for j, b in enumerate(blocks):
                row.append(_fmt_float(b["token_acc"]))
                accum[vk][j].append(float(b["token_acc"]))
            rows.append(row)

    for vk in variant_keys:
        agg_row = ["**mean ± std**", f"**{vk}**"]
        for j in range(n_blocks):
            mu, sd = _mean_std(accum[vk][j])
            agg_row.append(_fmt_mean_pm_std(mu, sd))
        rows.append(agg_row)

    return _md_table(headers=headers, rows=rows, aligns=["r", "l", *(["r"] * n_blocks)])


def _id_all_variants_top_level_table(
    summary: List[Dict[str, Any]], variant_keys: Sequence[str]
) -> str:
    """In-distribution test: one row per (seed, variant) plus mean ± std across seeds per variant."""
    headers = [
        "seed",
        "variant",
        "T",
        "token_acc",
        "sum_token_acc",
        "final_carry_acc",
        "seq_acc",
        "n_seq_w_err",
        "mean_first_wrong",
        "std_first_wrong",
    ]
    rows: List[List[str]] = []
    accum: Dict[str, Dict[str, List[float]]] = {
        vk: {
            "token_acc": [],
            "sum_token_acc": [],
            "final_carry_acc": [],
            "seq_acc": [],
            "n_seq_w_err": [],
            "mean_first_wrong": [],
            "std_first_wrong": [],
        }
        for vk in variant_keys
    }

    for entry in summary:
        for vk in variant_keys:
            sub = entry["stages"][vk]["id_metrics"]
            sum_only, _sum_seq, final_carry = _extract_sum_only_metrics(sub)
            sum_only, final_carry = _require_sum_carry_metrics(
                sum_only,
                final_carry,
                seed=entry["seed"],
                ctx=f"stages[{vk!r}].id_metrics",
            )
            rows.append([
                _fmt_int(entry["seed"]),
                vk,
                _fmt_int(sub["n_timesteps"]),
                _fmt_float(sub["token_acc"]),
                _fmt_float(sum_only),
                _fmt_float(final_carry),
                _fmt_float(sub["seq_acc"]),
                _fmt_int(sub["n_sequences_with_any_error"]),
                _fmt_float(sub["mean_first_wrong_timestep_among_wrong"]),
                _fmt_float(sub["std_first_wrong_timestep_among_wrong"]),
            ])
            accum[vk]["token_acc"].append(float(sub["token_acc"]))
            accum[vk]["sum_token_acc"].append(sum_only)
            accum[vk]["final_carry_acc"].append(final_carry)
            accum[vk]["seq_acc"].append(float(sub["seq_acc"]))
            accum[vk]["n_seq_w_err"].append(float(sub["n_sequences_with_any_error"]))
            accum[vk]["mean_first_wrong"].append(float(sub["mean_first_wrong_timestep_among_wrong"]))
            accum[vk]["std_first_wrong"].append(float(sub["std_first_wrong_timestep_among_wrong"]))

    for vk in variant_keys:
        n_steps = int(summary[0]["stages"][vk]["id_metrics"]["n_timesteps"])
        mu_tok, sd_tok = _mean_std(accum[vk]["token_acc"])
        mu_sum, sd_sum = _mean_std(accum[vk]["sum_token_acc"])
        mu_fc, sd_fc = _mean_std(accum[vk]["final_carry_acc"])
        mu_seq, sd_seq = _mean_std(accum[vk]["seq_acc"])
        mu_err, sd_err = _mean_std(accum[vk]["n_seq_w_err"])
        mu_fw, sd_fw = _mean_std(accum[vk]["mean_first_wrong"])
        mu_fws, sd_fws = _mean_std(accum[vk]["std_first_wrong"])
        rows.append([
            "**mean ± std**",
            f"**{vk}**",
            _fmt_int(n_steps),
            _fmt_mean_pm_std(mu_tok, sd_tok),
            _fmt_mean_pm_std(mu_sum, sd_sum),
            _fmt_mean_pm_std(mu_fc, sd_fc),
            _fmt_mean_pm_std(mu_seq, sd_seq),
            _fmt_mean_pm_std(mu_err, sd_err, digits=1),
            _fmt_mean_pm_std(mu_fw, sd_fw),
            _fmt_mean_pm_std(mu_fws, sd_fws),
        ])
    return _md_table(
        headers=headers,
        rows=rows,
        aligns=["r", "l", "r", "r", "r", "r", "r", "r", "r", "r"],
    )


def _id_all_variants_block_table(
    summary: List[Dict[str, Any]], variant_keys: Sequence[str]
) -> str:
    """Per-block ID token_acc for every variant."""
    first_blocks = summary[0]["stages"][variant_keys[0]]["id_metrics"]["per_five_timestep_blocks"]
    n_blocks = len(first_blocks)
    headers = ["seed", "variant", *_block_columns(first_blocks)]
    rows: List[List[str]] = []
    accum: Dict[str, List[List[float]]] = {vk: [[] for _ in range(n_blocks)] for vk in variant_keys}

    for entry in summary:
        for vk in variant_keys:
            blocks = entry["stages"][vk]["id_metrics"]["per_five_timestep_blocks"]
            if len(blocks) != n_blocks:
                raise ValueError(
                    f"ID per_five_timestep_blocks length mismatch for seed={entry['seed']!r} variant={vk!r}"
                )
            row = [_fmt_int(entry["seed"]), vk]
            for j, b in enumerate(blocks):
                row.append(_fmt_float(b["token_acc"]))
                accum[vk][j].append(float(b["token_acc"]))
            rows.append(row)

    for vk in variant_keys:
        agg_row = ["**mean ± std**", f"**{vk}**"]
        for j in range(n_blocks):
            mu, sd = _mean_std(accum[vk][j])
            agg_row.append(_fmt_mean_pm_std(mu, sd))
        rows.append(agg_row)

    return _md_table(headers=headers, rows=rows, aligns=["r", "l", *(["r"] * n_blocks)])


def _build_md(run_dir: Path) -> str:
    summary_path = run_dir / "summary_all_seeds.json"
    config_path = run_dir / "run_config.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary_all_seeds.json under {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing run_config.json under {run_dir}")

    with summary_path.open() as f:
        summary = json.load(f)
    with config_path.open() as f:
        config = json.load(f)
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"summary_all_seeds.json malformed (expected non-empty list) at {summary_path}")

    parts: List[str] = []
    parts.append(f"# {run_dir.name}\n")
    parts.append(_config_summary(config, summary) + "\n")
    parts.append("## Selected hyperparameters per seed\n")
    parts.append(_selected_hp_table(summary) + "\n")
    if _can_render_id_tables(summary):
        vks = _pipeline_variant_keys(summary)
        parts.append(
            "## In-distribution (ID test at train digit length)\n"
            "The **variant** column lists every pipeline stage (same keys as ``stages`` in "
            "``summary_all_seeds.json``), in run order.\n"
        )
        parts.append("### Top-level metrics\n")
        parts.append(_id_all_variants_top_level_table(summary, vks) + "\n")
        parts.append("### Per 5-timestep block token_acc\n")
        parts.append(_id_all_variants_block_table(summary, vks) + "\n")
    for ood_key in _ood_keys_sorted(summary):
        parts.append(f"## OOD: {ood_key}\n")
        parts.append("### Top-level metrics\n")
        parts.append(_top_level_table(summary, ood_key) + "\n")
        parts.append("### Per 5-timestep block token_acc\n")
        parts.append(_block_table(summary, ood_key) + "\n")
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render summary_all_seeds.md per length-gen sweep dir."
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=(
            "Glob pattern under atomic/sweep_results/. Default matches every length-gen base. "
            "Use a stricter pattern (e.g. 'arithmetic_add_b2_len_gen_L3_traind5_K2_2026*') to narrow."
        ),
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    sweeps = repo / "sweep_results"
    if not sweeps.exists():
        raise SystemExit(f"sweep_results not found: {sweeps}")
    matches = sorted(p for p in sweeps.glob(args.pattern) if p.is_dir())
    if not matches:
        raise SystemExit(f"No directories matched {args.pattern!r} under {sweeps}")
    written: List[str] = []
    skipped: List[str] = []
    for d in matches:
        if not (d / "summary_all_seeds.json").exists():
            skipped.append(f"{d.name} (no summary_all_seeds.json)")
            continue
        md_text = _build_md(d)
        out_path = d / "summary_all_seeds.md"
        out_path.write_text(md_text)
        written.append(str(out_path))
    print(f"wrote {len(written)} markdown files:")
    for p in written:
        print(f"  {p}")
    if skipped:
        print(f"skipped {len(skipped)} directories:")
        for s in skipped:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
