#!/usr/bin/env python3
"""
Walk DFA sweep result directories and write ``token_metrics_summary.md`` next to each
``metrics.json``.

For every pipeline stage in ``run_config.stages``, the markdown contains:
- per-seed **ID** and **OOD** **last_step_acc** (the bench's sequence-level last-step token
  accuracy — the primary "overall" accuracy reported for DFA last-step tasks),
- a trailing **mean ± std** row across seeds (sample std, ``ddof=1``; single-seed runs omit std).

OOD columns follow ``T_train`` / ``T_ood`` / multiplier from the JSON keys (sorted by ``T_ood``).

Default scope: each ``atomic/sweep_results/dfa_random_ablation_*`` directory, processing
every immediate child subdirectory that contains ``metrics.json``.

Examples::

    cd atomic
    python3 scripts/build_dfa_token_metrics_md.py
    python3 scripts/build_dfa_token_metrics_md.py --sweep_dir sweep_results/dfa_random_ablation_T9_ntrain5000_20260505_131956
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vs = [float(v) for v in values]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], None
    n = len(vs)
    mu = sum(vs) / n
    var = sum((v - mu) ** 2 for v in vs) / (n - 1)
    return mu, math.sqrt(var)


def _fmt_float(x: Any, *, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{digits}f}"


def _fmt_mean_pm(mu: Optional[float], sd: Optional[float], *, digits: int = 4) -> str:
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
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(sep) + "|",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row width mismatch: {row} vs {len(headers)}")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _ood_keys_sorted(ood_eval: Dict[str, Any]) -> List[str]:
    keys = list(ood_eval.keys())

    def _t_ood(k: str) -> int:
        b = ood_eval[k]
        if not isinstance(b, dict):
            raise TypeError(f"ood_eval[{k!r}] must be dict")
        return int(b["T_ood"])

    return sorted(keys, key=_t_ood)


def _ood_col_label(ood_eval: Dict[str, Any], key: str) -> str:
    b = ood_eval[key]
    if not isinstance(b, dict):
        raise TypeError(f"ood_eval[{key!r}] must be dict")
    train = int(b["T_train"])
    mult = int(b["multiplier"])
    todd = int(b["T_ood"])
    return f"OOD T={todd} (train T={train}, ×{mult})"


def _id_last_step(stage_block: Dict[str, Any]) -> float:
    idt = stage_block["id_test"]
    if not isinstance(idt, dict):
        raise TypeError("id_test must be dict")
    v = idt["last_step_acc"]
    return float(v)


def _ood_last_step(ood_block: Dict[str, Any]) -> float:
    v = ood_block["last_step_acc"]
    return float(v)


def _build_md(cell_dir: Path, payload: Dict[str, Any]) -> str:
    rc = payload["run_config"]
    if not isinstance(rc, dict):
        raise TypeError("metrics.json: run_config must be dict")
    stages_order: List[str] = list(rc["stages"])
    seeds_payload = payload["seeds"]
    if not isinstance(seeds_payload, list) or not seeds_payload:
        raise ValueError("metrics.json: seeds must be non-empty list")

    sample_stages = seeds_payload[0]["stages"]
    if not isinstance(sample_stages, dict):
        raise TypeError("first seed: stages must be dict")

    ood_eval0: Dict[str, Any] = {}
    for sk in stages_order:
        if sk not in sample_stages:
            raise KeyError(f"stage {sk!r} missing in seed 0 stages")
        oe = sample_stages[sk].get("ood_eval")
        if isinstance(oe, dict) and oe:
            ood_eval0 = oe
            break
    if not ood_eval0:
        raise ValueError("no non-empty ood_eval found in first seed stages")

    ood_keys = _ood_keys_sorted(ood_eval0)
    ood_headers = [_ood_col_label(ood_eval0, k) for k in ood_keys]

    lines: List[str] = []
    lines.append(f"# DFA token metrics (`last_step_acc`) — `{cell_dir.name}`\n")
    lines.append(
        f"- **dfa_spec**: `{rc.get('dfa_spec')}` | **T_train**: {rc.get('T')} | "
        f"**n_test** (ID): {rc.get('n_test')} | **n_test** (OOD): {rc.get('n_test_ood')}\n"
    )
    lines.append(
        f"- **Stages**: {', '.join(stages_order)}\n"
    )
    lines.append(
        "- **Metric**: `last_step_acc` = fraction of sequences with correct **last-step** "
        "prediction (ID test split and full-length OOD sequences).\n"
    )

    for stage_key in stages_order:
        lines.append(f"## `{stage_key}`\n")
        id_vals: List[float] = []
        ood_cols: Dict[str, List[float]] = {k: [] for k in ood_keys}

        headers = ["seed", "ID last_step_acc", *ood_headers]
        rows: List[List[str]] = []

        for sp in seeds_payload:
            seed = int(sp["seed"])
            st = sp["stages"]
            if not isinstance(st, dict):
                raise TypeError(f"seed {seed}: stages must be dict")
            if stage_key not in st:
                raise KeyError(f"seed {seed}: missing stage {stage_key!r}")
            sb = st[stage_key]
            if not isinstance(sb, dict):
                raise TypeError(f"seed {seed} stage {stage_key!r}: must be dict")

            id_acc = _id_last_step(sb)
            id_vals.append(id_acc)

            oe = sb.get("ood_eval")
            if not isinstance(oe, dict):
                raise TypeError(f"seed {seed} stage {stage_key!r}: ood_eval must be dict")

            if list(_ood_keys_sorted(oe)) != ood_keys:
                raise ValueError(
                    f"seed {seed} stage {stage_key!r}: OOD keys {list(oe.keys())} "
                    f"!= first-seed reference {ood_keys}"
                )

            row = [str(seed), _fmt_float(id_acc)]
            for ok in ood_keys:
                ood_cols[ok].append(_ood_last_step(oe[ok]))
                row.append(_fmt_float(_ood_last_step(oe[ok])))
            rows.append(row)

        mu_id, sd_id = _mean_std(id_vals)
        agg = ["**mean ± std**", _fmt_mean_pm(mu_id, sd_id)]
        for ok in ood_keys:
            mu, sd = _mean_std(ood_cols[ok])
            agg.append(_fmt_mean_pm(mu, sd))
        rows.append(agg)

        aligns = ["r", "r", *(["r"] * len(ood_keys))]
        lines.append(_md_table(headers, rows, aligns) + "\n")

    return "\n".join(lines) + "\n"


def _iter_cell_metrics_paths(
    sweeps_root: Path,
    *,
    sweep_glob: str,
) -> List[Path]:
    """Return paths ``.../cell_dir/metrics.json`` for standard DFA ablation layout."""
    out: List[Path] = []
    for sweep_dir in sorted(sweeps_root.glob(sweep_glob)):
        if not sweep_dir.is_dir():
            continue
        for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
            mp = cell_dir / "metrics.json"
            if mp.is_file():
                out.append(mp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Write token_metrics_summary.md per DFA cell.")
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--sweeps_root",
        type=Path,
        default=repo / "sweep_results",
        help="Directory containing dfa_random_ablation_* (default: atomic/sweep_results).",
    )
    ap.add_argument(
        "--sweep_glob",
        type=str,
        default="dfa_random_ablation_*",
        help="Glob under sweeps_root for sweep parent directories.",
    )
    ap.add_argument(
        "--sweep_dir",
        type=Path,
        default=None,
        help="If set, only process immediate children of this directory (instead of sweep_glob).",
    )
    ap.add_argument(
        "--output_name",
        type=str,
        default="token_metrics_summary.md",
        help="Markdown filename written next to each metrics.json.",
    )
    args = ap.parse_args()

    if args.sweep_dir is not None:
        sweep_dir = args.sweep_dir.resolve()
        if not sweep_dir.is_dir():
            raise SystemExit(f"not a directory: {sweep_dir}")
        metrics_paths = []
        for cell_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
            mp = cell_dir / "metrics.json"
            if mp.is_file():
                metrics_paths.append(mp)
    else:
        root = args.sweeps_root.resolve()
        if not root.is_dir():
            raise SystemExit(f"sweeps_root not found: {root}")
        metrics_paths = _iter_cell_metrics_paths(root, sweep_glob=args.sweep_glob)

    if not metrics_paths:
        raise SystemExit("No metrics.json files found (check --sweeps_root / --sweep_dir / --sweep_glob).")

    written: List[str] = []
    for mp in metrics_paths:
        cell_dir = mp.parent
        payload = json.loads(mp.read_text())
        md = _build_md(cell_dir, payload)
        out_path = cell_dir / args.output_name
        out_path.write_text(md)
        written.append(str(out_path))

    print(f"wrote {len(written)} files:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
