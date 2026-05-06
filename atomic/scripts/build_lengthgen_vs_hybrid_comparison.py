#!/usr/bin/env python3
"""
For each ``atomic/sweep_results/arithmetic_add_b{B}_len_gen_*`` directory, write
``comparison_to_hybrid_sum_token_acc.md`` next to it that compares its **sum-position-only**
token accuracy against the corresponding ``carry_hybrid_b{B}_d{n_digits_train}_*`` run.

Length-gen has 1 next-token head (mixes sum digits and MSD carry-out); ``sum_token_acc`` is
restricted to the sum positions (timesteps ``0..n_digits-1``) — directly comparable to the
hybrid bench's ``sum_token_acc`` head metric.

The hybrid bench sweeps ``lambda_carry``; this script reports two reductions per cell:
- ``λ=1.0`` (the canonical balanced weight),
- ``best-λ`` (max ``sum_token_acc`` mean across the lambda grid; the chosen ``lambda_carry``
  is shown in parens for traceability).

Hybrid bench sweeps ``lambda_carry``; this script reports two reductions per cell:
- ``λ=1.0`` (the canonical balanced weight),
- ``best-λ`` (max ``sum_token_acc`` mean across the lambda grid; the chosen ``lambda_carry``
  is shown in parens for traceability).

Only **autoregressive (AR)** hybrid eval is included (model uses its own predicted carry as the
next-step input). Teacher-forcing (TF) rows are omitted.

Numeric format: ``mean ± std`` to 4 decimals; missing cells render as ``-``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


HYBRID_STAGES: Tuple[str, ...] = (
    "ste_pretrain",
    "cvx_from_ste_pretrain",
    "ste_finetune_from_ste_pretrain_new_train",
    "cvx_pretrain",
    "ste_finetune_from_cvx_pretrain_new_train",
)
HYBRID_STAGE_LABELS: Dict[str, str] = {
    "ste_pretrain": "STE pretrain (split A)",
    "cvx_from_ste_pretrain": "CVX from STE init",
    "ste_finetune_from_ste_pretrain_new_train": "STE finetune (from STE, split B)",
    "cvx_pretrain": "CVX pretrain (Gaussian)",
    "ste_finetune_from_cvx_pretrain_new_train": "STE finetune (from CVX, split B)",
}
EVAL_MODES: Tuple[str, ...] = ("autoregressive",)

# Length-gen stage keys (when --variants=full5). Legacy/minimal runs only have
# `ste_pretrain` and `cvx_pretrain` (mirrored as ``ste`` / ``cvx`` in older summaries).
LENGTHGEN_STAGES: Tuple[str, ...] = (
    "ste_pretrain",
    "cvx_from_ste_pretrain",
    "ste_finetune_from_ste_pretrain_new_train",
    "cvx_pretrain",
    "ste_finetune_from_cvx_pretrain_new_train",
)
LENGTHGEN_STAGE_LABELS: Dict[str, str] = {
    "ste_pretrain": "len-gen STE pretrain",
    "cvx_from_ste_pretrain": "len-gen CVX from STE init",
    "ste_finetune_from_ste_pretrain_new_train": "len-gen STE finetune (from STE)",
    "cvx_pretrain": "len-gen CVX pretrain (Gaussian)",
    "ste_finetune_from_cvx_pretrain_new_train": "len-gen STE finetune (from CVX)",
}
# Legacy summary keys (minimal variant): ``ste`` -> ste_pretrain, ``cvx`` -> cvx_pretrain.
LENGTHGEN_LEGACY_KEY_FOR_STAGE: Dict[str, str] = {
    "ste_pretrain": "ste",
    "cvx_pretrain": "cvx",
}


# ---------------------------------------------------------------------------
# helpers (some intentionally repeat ideas from build_arith_b2_summary_md.py to
# keep this script standalone)
# ---------------------------------------------------------------------------


def _fmt_mean_std(mu: Optional[float], sd: Optional[float], digits: int = 4) -> str:
    if mu is None:
        return "-"
    if sd is None:
        return f"{mu:.{digits}f}"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], 0.0
    n = len(vs)
    mu = sum(vs) / n
    var = sum((v - mu) ** 2 for v in vs) / (n - 1)
    return mu, math.sqrt(var)


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
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row width mismatch: {row}")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# length-gen sum-only extraction (rule_context if available, else derive from
# per-block token_acc when last block holds exactly the 1-token MSD carry-out)
# ---------------------------------------------------------------------------


def _block_t_range(b: Dict[str, Any]) -> Tuple[int, int]:
    if "timestep_start" in b and "timestep_end_inclusive" in b:
        return int(b["timestep_start"]), int(b["timestep_end_inclusive"])
    if "timestep_start_0based" in b and "timestep_end_0based_inclusive" in b:
        return int(b["timestep_start_0based"]), int(b["timestep_end_0based_inclusive"])
    raise KeyError(f"per-block dict has no timestep_start/end keys; got keys={list(b.keys())}")


def _lengthgen_sum_token_acc_for_method(method_block: Dict[str, Any]) -> Optional[float]:
    """Return single-seed length-gen ``token_acc`` matching hybrid ``sum_token_acc``.

    Both metrics average ``(pred == target_tokens).mean()`` over ``(N, T)`` with
    ``T = n_digits + 1`` (5 sum digits + 1 MSD carry-out). Hybrid's ``sum_token_acc``
    *includes* the MSD-carry-out position because ``y_sum = target_tokens`` covers all
    T positions. So the apples-to-apples length-gen value is its raw ``token_acc``,
    not a "sum-only" subset.
    """
    overall = method_block.get("token_acc")
    if overall is None:
        return None
    return float(overall)


def _lengthgen_resolve_stage_block(seed_entry: Dict[str, Any], ood_key: str, stage_key: str) -> Optional[Dict[str, Any]]:
    """Look up an OOD method block for the given length-gen stage in a seed payload.

    Tolerant of two layouts:
    - newer (variants=full5): ``ood_eval[ood_key][stage_key]`` and/or
      ``stages[stage_key].ood_eval[ood_key]``.
    - legacy (variants=minimal): ``ood_eval[ood_key]['ste'|'cvx']`` (only ``ste_pretrain``
      and ``cvx_pretrain`` are available).
    """
    ood_block = (seed_entry.get("ood_eval") or {}).get(ood_key)
    if isinstance(ood_block, dict) and stage_key in ood_block:
        return ood_block[stage_key]
    legacy = LENGTHGEN_LEGACY_KEY_FOR_STAGE.get(stage_key)
    if legacy is not None and isinstance(ood_block, dict) and legacy in ood_block:
        return ood_block[legacy]
    stages = seed_entry.get("stages")
    if isinstance(stages, dict) and stage_key in stages:
        ood = (stages[stage_key].get("ood_eval") or {})
        if isinstance(ood, dict) and ood_key in ood:
            return ood[ood_key]
    return None


def _lengthgen_resolve_id_block(seed_entry: Dict[str, Any], stage_key: str) -> Optional[Dict[str, Any]]:
    """ID-test block for a length-gen stage. Newer runs put it under
    ``stages[stage_key].id_metrics``; legacy runs implicitly used the OOD entry
    keyed by the in-distribution ``n_digits_train`` length, which the caller passes
    via :func:`_lengthgen_resolve_stage_block` with that key.
    """
    stages = seed_entry.get("stages")
    if isinstance(stages, dict) and stage_key in stages:
        m = stages[stage_key].get("id_metrics")
        if isinstance(m, dict):
            return m
    return None


def _lengthgen_aggregate_per_ood(
    summary: List[Dict[str, Any]], ood_key: str, method_or_stage: str
) -> Tuple[Optional[float], Optional[float], int]:
    """Return ``(mean, std, n_seeds_used)`` of length-gen sum-only token_acc.

    ``method_or_stage`` is one of:
    - legacy method keys: ``ste`` (==ste_pretrain), ``cvx`` (==cvx_pretrain).
    - new stage keys: ``ste_pretrain``, ``cvx_from_ste_pretrain``,
      ``ste_finetune_from_ste_pretrain_new_train``, ``cvx_pretrain``,
      ``ste_finetune_from_cvx_pretrain_new_train``.
    """
    vals: List[float] = []
    for entry in summary:
        method_block = _lengthgen_resolve_stage_block(entry, ood_key, str(method_or_stage))
        if method_block is None:
            continue
        v = _lengthgen_sum_token_acc_for_method(method_block)
        if v is None:
            continue
        vals.append(float(v))
    mu, sd = _mean_std(vals)
    return mu, sd, len(vals)


def _lengthgen_aggregate_id(
    summary: List[Dict[str, Any]], stage_key: str, lg_id_key: Optional[str]
) -> Tuple[Optional[float], Optional[float], int]:
    """Aggregate length-gen ID sum_token_acc for a stage, preferring stages[*].id_metrics
    when present and falling back to ood_eval[lg_id_key][stage|legacy].
    """
    vals: List[float] = []
    for entry in summary:
        block = _lengthgen_resolve_id_block(entry, stage_key)
        if block is None and lg_id_key is not None:
            block = _lengthgen_resolve_stage_block(entry, lg_id_key, stage_key)
        if block is None:
            continue
        v = _lengthgen_sum_token_acc_for_method(block)
        if v is None:
            continue
        vals.append(float(v))
    mu, sd = _mean_std(vals)
    return mu, sd, len(vals)


# ---------------------------------------------------------------------------
# hybrid sum_token_acc extraction: ``aggregate.lambda_sweep`` already aggregates
# across seeds for us
# ---------------------------------------------------------------------------


def _hybrid_sum_token_acc_at_lambda(
    hybrid_lambda_entry: Dict[str, Any],
    stage: str,
    eval_mode: str,
    ood_key: Optional[str],
    *,
    is_id_column: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """Pull (mean, std) of ``sum_token_acc`` from the hybrid aggregate for one lambda row.

    ``is_id_column=True`` reads from ``id_metrics`` and ignores ``ood_key``.
    Otherwise we read ``ood_eval[ood_key]['metrics']``; if ``ood_key`` is None or absent,
    return ``(None, None)`` (column not present on the hybrid side).
    """
    stage_block = hybrid_lambda_entry.get(stage)
    if not isinstance(stage_block, dict):
        return None, None
    mode_block = stage_block.get(eval_mode)
    if not isinstance(mode_block, dict):
        return None, None
    if is_id_column:
        m = mode_block.get("id_metrics")
    else:
        if ood_key is None:
            return None, None
        ood = mode_block.get("ood_eval") or {}
        m = (ood.get(ood_key) or {}).get("metrics")
    if not isinstance(m, dict):
        return None, None
    s = m.get("sum_token_acc")
    if not isinstance(s, dict):
        return None, None
    mu = s.get("mean")
    sd = s.get("std")
    return (None if mu is None else float(mu), None if sd is None else float(sd))


def _hybrid_pick_lambda_one_oh(
    hybrid_aggregate_lambda_sweep: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the entry whose ``lambda_carry == 1.0`` (within 1e-9), else None."""
    for entry in hybrid_aggregate_lambda_sweep:
        try:
            if abs(float(entry["lambda_carry"]) - 1.0) <= 1e-9:
                return entry
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _hybrid_best_lambda_cell(
    hybrid_aggregate_lambda_sweep: Sequence[Dict[str, Any]],
    stage: str,
    eval_mode: str,
    ood_key: Optional[str],
    *,
    is_id_column: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Across the lambda sweep, return ``(best_mean, best_std, best_lambda)`` for one cell."""
    best: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)
    for entry in hybrid_aggregate_lambda_sweep:
        try:
            lc = float(entry["lambda_carry"])
        except (KeyError, TypeError, ValueError):
            continue
        mu, sd = _hybrid_sum_token_acc_at_lambda(
            entry, stage, eval_mode, ood_key, is_id_column=is_id_column
        )
        if mu is None:
            continue
        if best[0] is None or float(mu) > float(best[0]):
            best = (float(mu), None if sd is None else float(sd), lc)
    return best


# ---------------------------------------------------------------------------
# pairing length-gen runs to a hybrid run by base + n_digits_train
# ---------------------------------------------------------------------------


def _find_hybrid_run(sweeps_root: Path, base: int, n_digits_train: int) -> Optional[Path]:
    candidates = sorted(sweeps_root.glob(f"carry_hybrid_b{int(base)}_d{int(n_digits_train)}_*"))
    valid: List[Path] = []
    for c in candidates:
        if (c / "metrics.json").exists():
            valid.append(c)
    if not valid:
        return None
    if len(valid) > 1:
        # Newer timestamps sort last; pick the most recent. Print all candidates so the user
        # can override by reorganizing the directory structure if needed.
        print(
            f"[pair] multiple hybrid candidates for base={base} d={n_digits_train}; using newest: "
            f"{valid[-1].name}; others: {[c.name for c in valid[:-1]]}",
            file=sys.stderr,
        )
    return valid[-1]


def _length_gen_ood_to_hybrid_ood(
    lengthgen_ood_keys: Sequence[str],
    hybrid_aggregate_lambda_sweep: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], List[str]]:
    """Build a list of ``(lengthgen_ood_key | None, hybrid_ood_key | None)`` pairs that the
    comparison table will iterate over, plus a list of column display labels.

    The returned pairs include:
      - ID at the training length: lengthgen key=``n_digits_<n_digits_train>`` if present;
        hybrid key=None (uses ``id_metrics``). If neither side has it, omit.
      - OOD at lengths present in both sides.
      - OOD lengths only on one side (renders ``-`` on the missing side).
    Pairs preserve numeric order on the digit length.
    """
    # Collect lengths from both sides.
    lengthgen_lengths: Dict[int, str] = {}
    for k in lengthgen_ood_keys:
        if not k.startswith("n_digits_"):
            continue
        lengthgen_lengths[int(k.split("_")[-1])] = k

    hybrid_ood_keys_observed: List[str] = []
    if hybrid_aggregate_lambda_sweep:
        ts0 = hybrid_aggregate_lambda_sweep[0]
        # Pick any stage/mode that has ood_eval to enumerate its keys; they're consistent.
        for stage in HYBRID_STAGES:
            sb = ts0.get(stage)
            if not isinstance(sb, dict):
                continue
            for mode in EVAL_MODES:
                mb = sb.get(mode)
                if isinstance(mb, dict) and isinstance(mb.get("ood_eval"), dict):
                    hybrid_ood_keys_observed = list(mb["ood_eval"].keys())
                    break
            if hybrid_ood_keys_observed:
                break

    hybrid_lengths: Dict[int, str] = {}
    for k in hybrid_ood_keys_observed:
        if not k.startswith("n_digits_"):
            continue
        hybrid_lengths[int(k.split("_")[-1])] = k

    all_lengths = sorted(set(lengthgen_lengths.keys()) | set(hybrid_lengths.keys()))
    pairs: List[Tuple[Optional[str], Optional[str]]] = []
    headers: List[str] = []
    for L in all_lengths:
        lg_k = lengthgen_lengths.get(L)
        hy_k = hybrid_lengths.get(L)
        pairs.append((lg_k, hy_k))
        headers.append(f"n={L}")
    return pairs, headers


# ---------------------------------------------------------------------------
# build the markdown
# ---------------------------------------------------------------------------


def _build_comparison_md(
    *,
    lengthgen_dir: Path,
    hybrid_dir: Path,
) -> str:
    lengthgen_summary_path = lengthgen_dir / "summary_all_seeds.json"
    lengthgen_run_config_path = lengthgen_dir / "run_config.json"
    hybrid_metrics_path = hybrid_dir / "metrics.json"

    lengthgen_summary = json.loads(lengthgen_summary_path.read_text())
    lengthgen_rc = json.loads(lengthgen_run_config_path.read_text())
    hybrid_payload = json.loads(hybrid_metrics_path.read_text())
    hybrid_rc = hybrid_payload.get("run_config", {})
    hybrid_lambda_sweep = (hybrid_payload.get("aggregate") or {}).get("lambda_sweep", [])
    hybrid_seed_list = [int(s["seed"]) for s in hybrid_payload.get("seeds", [])]
    lengthgen_seed_list = [int(s["seed"]) for s in lengthgen_summary]
    # Detect which length-gen stages have any data in this summary; minimal-variant runs
    # only carry ste_pretrain and cvx_pretrain (under legacy ``ste``/``cvx`` keys).
    available_lg_stages: List[str] = []
    if lengthgen_summary:
        sample_seed = lengthgen_summary[0]
        sample_ood_key = next(iter((sample_seed.get("ood_eval") or {})), None)
        for stage_key in LENGTHGEN_STAGES:
            present = False
            if sample_ood_key is not None and isinstance(sample_seed.get("ood_eval"), dict):
                present = (
                    stage_key in sample_seed["ood_eval"][sample_ood_key]
                    or LENGTHGEN_LEGACY_KEY_FOR_STAGE.get(stage_key) in sample_seed["ood_eval"][sample_ood_key]
                )
            if not present and isinstance(sample_seed.get("stages"), dict):
                present = stage_key in sample_seed["stages"]
            if present:
                available_lg_stages.append(stage_key)
    if not available_lg_stages:
        # Fall back to the legacy two-stage layout if probing failed (older summaries).
        available_lg_stages = ["ste_pretrain", "cvx_pretrain"]

    # Decide column structure: include ID column when length-gen has it either as
    # ``stages[*].id_metrics`` (full5/post-2026-04 layout) OR as an explicit
    # ``ood_eval[n_digits_<n_train>]`` block (legacy minimal layout). Hybrid always has
    # ``id_metrics`` per stage / mode, so the column is meaningful whenever length-gen
    # exposes either one.
    n_train_digits = int(lengthgen_rc.get("n_digits_train", 0)) or None
    lg_id_key = f"n_digits_{n_train_digits}" if n_train_digits is not None else None
    sample_seed_for_id = lengthgen_summary[0]
    sample_stages = sample_seed_for_id.get("stages") if isinstance(sample_seed_for_id, dict) else None
    has_id_via_stages = False
    if isinstance(sample_stages, dict):
        for sk in available_lg_stages:
            block = (sample_stages.get(sk) or {}).get("id_metrics")
            if isinstance(block, dict) and block.get("token_acc") is not None:
                has_id_via_stages = True
                break
    has_id_via_ood_key = bool(lg_id_key and lg_id_key in sample_seed_for_id.get("ood_eval", {}))
    has_id_pair = has_id_via_stages or has_id_via_ood_key

    pairs, ood_col_headers = _length_gen_ood_to_hybrid_ood(
        list(lengthgen_summary[0].get("ood_eval", {}).keys()),
        hybrid_lambda_sweep,
    )

    # Build OOD-only pairs after stripping the ID key from the length-gen OOD list (so we
    # don't double-count the in-distribution column below).
    pairs_ood, ood_col_headers = _length_gen_ood_to_hybrid_ood(
        [k for k in lengthgen_summary[0].get("ood_eval", {}).keys() if k != lg_id_key],
        hybrid_lambda_sweep,
    )

    # Each entry in col_specs is (lg_key_or_None, hy_ood_key_or_None, is_id_column, header_label).
    col_specs: List[Tuple[Optional[str], Optional[str], bool, str]] = []
    if has_id_pair:
        col_specs.append((lg_id_key, None, True, f"ID (n={n_train_digits})"))
    for (lg_k, hy_k), label in zip(pairs_ood, ood_col_headers):
        col_specs.append((lg_k, hy_k, False, label))
    col_headers = [c[3] for c in col_specs]

    # ---- header / preamble
    lines: List[str] = []
    lines.append(f"# {lengthgen_dir.name} vs hybrid carry bench (sum_token_acc)\n")
    lines.append(
        f"- Length-gen run: `{lengthgen_dir.name}` "
        f"(arith_base={lengthgen_rc.get('arith_base')}, L={lengthgen_rc.get('L')}, "
        f"K_parallel={lengthgen_rc.get('K_parallel')}, n_digits_train={n_train_digits}, "
        f"n_train={lengthgen_rc.get('n_train')}, "
        f"loss_type={lengthgen_rc.get('loss_type')}, "
        f"cvx_method={lengthgen_rc.get('cvx_method')}, "
        f"variants={lengthgen_rc.get('variants', 'minimal')}, "
        f"mask_carry_in={lengthgen_rc.get('mask_carry_in', False)}, "
        f"seeds={lengthgen_seed_list})\n"
    )
    lines.append(
        f"- Hybrid run:    `{hybrid_dir.name}` "
        f"(arith_base={hybrid_rc.get('arith_base')}, L={hybrid_rc.get('L')}, "
        f"K_parallel={hybrid_rc.get('K_parallel')}, n_digits={hybrid_rc.get('n_digits')}, "
        f"ste_last_layer_readout={hybrid_rc.get('ste_last_layer_readout')}, "
        f"cvx_last_layer_readout={hybrid_rc.get('cvx_last_layer_readout')}, "
        f"lambda_carry_grid={[float(e['lambda_carry']) for e in hybrid_lambda_sweep]}, "
        f"seeds={hybrid_seed_list})\n"
    )
    lines.append(
        "- Metric: **sum_token_acc** in the hybrid bench averages "
        "``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where "
        "``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the "
        "MSD carry-out at the final step. The matching length-gen quantity is its raw "
        "``token_acc`` (single-head, same target sequence, same averaging) — both metrics "
        "are mathematically identical.\n"
    )
    lines.append(
        "- Hybrid eval: **AR** (autoregressive rollout — model feeds its own predicted carry "
        "into the next step). Teacher-forcing (TF) hybrid numbers are not listed here.\n"
    )
    lines.append(
        "- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid "
        "uses its own seed list (they are independent — see headers above).\n"
    )

    # ---- length-gen rows (no λ, no eval mode distinction; one number per OOD)
    lg_rows: List[List[str]] = []
    for stage_key in available_lg_stages:
        cells: List[str] = []
        for lg_k, _hy_k, is_id, _hdr in col_specs:
            if is_id:
                # Prefer stages[stage_key].id_metrics when present; fall back to OOD entry
                # at the in-distribution n_digits length.
                mu, sd, _n = _lengthgen_aggregate_id(lengthgen_summary, stage_key, lg_id_key)
                cells.append(_fmt_mean_std(mu, sd))
                continue
            if lg_k is None:
                cells.append("-")
                continue
            mu, sd, _n = _lengthgen_aggregate_per_ood(lengthgen_summary, lg_k, stage_key)
            cells.append(_fmt_mean_std(mu, sd))
        lg_rows.append([LENGTHGEN_STAGE_LABELS[stage_key], *cells])
    lines.append("## Length-gen token_acc (== hybrid sum_token_acc; full T positions)\n")
    lines.append(_md_table(["length-gen stage", *col_headers], lg_rows, ["l", *(["r"] * len(col_headers))]) + "\n")

    # ---- hybrid table at λ=1.0
    one_oh = _hybrid_pick_lambda_one_oh(hybrid_lambda_sweep)
    hybrid_one_rows: List[List[str]] = []
    if one_oh is not None:
        for stage in HYBRID_STAGES:
            for mode in EVAL_MODES:
                cells: List[str] = []
                for _lg_k, hy_k, is_id, _hdr in col_specs:
                    mu, sd = _hybrid_sum_token_acc_at_lambda(
                        one_oh, stage, mode, hy_k, is_id_column=is_id
                    )
                    cells.append(_fmt_mean_std(mu, sd))
                hybrid_one_rows.append([
                    HYBRID_STAGE_LABELS[stage], *cells
                ])
    lines.append("## Hybrid sum_token_acc at λ_carry = 1.0 (AR eval)\n")
    if one_oh is None:
        lines.append("_λ_carry=1.0 not present in hybrid lambda grid — skipping this table._\n")
    else:
        lines.append(_md_table(
            ["hybrid stage", *col_headers],
            hybrid_one_rows,
            ["l", *(["r"] * len(col_headers))],
        ) + "\n")

    # ---- hybrid table picking best λ per cell (max sum_token_acc mean)
    hybrid_best_rows: List[List[str]] = []
    for stage in HYBRID_STAGES:
        for mode in EVAL_MODES:
            cells: List[str] = []
            for _lg_k, hy_k, is_id, _hdr in col_specs:
                mu, sd, lam = _hybrid_best_lambda_cell(
                    hybrid_lambda_sweep, stage, mode, hy_k, is_id_column=is_id
                )
                if mu is None:
                    cells.append("-")
                    continue
                lam_str = f" (λ={lam:g})" if lam is not None else ""
                cells.append(_fmt_mean_std(mu, sd) + lam_str)
            hybrid_best_rows.append([
                HYBRID_STAGE_LABELS[stage], *cells
            ])
    lines.append("## Hybrid sum_token_acc — best λ_carry per cell (AR eval; selection on max mean)\n")
    lines.append(_md_table(
        ["hybrid stage", *col_headers],
        hybrid_best_rows,
        ["l", *(["r"] * len(col_headers))],
    ) + "\n")

    return "\n".join(lines) + "\n"


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sweeps = repo / "sweep_results"
    if not sweeps.exists():
        raise SystemExit(f"sweep_results not found: {sweeps}")
    written: List[str] = []
    skipped: List[str] = []
    for lengthgen_dir in sorted(sweeps.glob("arithmetic_add_b*_len_gen_*")):
        if not lengthgen_dir.is_dir():
            continue
        lengthgen_summary = lengthgen_dir / "summary_all_seeds.json"
        lengthgen_rc_path = lengthgen_dir / "run_config.json"
        if not lengthgen_summary.exists() or not lengthgen_rc_path.exists():
            skipped.append(f"{lengthgen_dir.name} (missing summary_all_seeds.json or run_config.json)")
            continue
        rc = json.loads(lengthgen_rc_path.read_text())
        base = int(rc.get("arith_base"))
        n_train_digits = int(rc.get("n_digits_train", 0))
        if base <= 0 or n_train_digits <= 0:
            skipped.append(f"{lengthgen_dir.name} (bad arith_base/n_digits_train in run_config.json)")
            continue
        hybrid_dir = _find_hybrid_run(sweeps, base, n_train_digits)
        if hybrid_dir is None:
            skipped.append(
                f"{lengthgen_dir.name} (no carry_hybrid_b{base}_d{n_train_digits}_* with metrics.json)"
            )
            continue
        md_text = _build_comparison_md(
            lengthgen_dir=lengthgen_dir,
            hybrid_dir=hybrid_dir,
        )
        out_path = lengthgen_dir / "comparison_to_hybrid_sum_token_acc.md"
        out_path.write_text(md_text)
        written.append(str(out_path))

    print(f"wrote {len(written)} comparison files:")
    for p in written:
        print(f"  {p}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
