#!/usr/bin/env python3
"""Regenerate PERF_TABLES_*.md and latex_AR_stats.tex for a given seed (same layout as seed_0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

STAGES: List[Tuple[str, str]] = [
    ("ste_pretrain", "STE pretrain (split A)"),
    ("cvx_from_ste_pretrain", "CVX from STE init"),
    ("ste_finetune_from_ste_pretrain_new_train", "STE finetune (from STE, split B)"),
    ("cvx_pretrain", "CVX pretrain (Gaussian)"),
    ("ste_finetune_from_cvx_pretrain_new_train", "STE finetune (from CVX, split B)"),
]
METRIC_ORDER = [
    "joint_token_acc",
    "joint_seq_acc",
    "sum_token_acc",
    "carry_token_acc",
    "sum_seq_acc",
    "carry_seq_acc",
    "mean_first_wrong_sum_among_error_seq",
    "mean_first_wrong_carry_among_error_seq",
]
SHORT = {
    "joint_token_acc": "joint_tok",
    "joint_seq_acc": "joint_seq",
    "sum_token_acc": "sum_tok",
    "carry_token_acc": "carry_tok",
    "sum_seq_acc": "sum_seq",
    "carry_seq_acc": "carry_seq",
    "mean_first_wrong_sum_among_error_seq": "1st_wrong_sum",
    "mean_first_wrong_carry_among_error_seq": "1st_wrong_car",
}
MODES = [("teacher_forcing", "TF"), ("autoregressive", "AR")]


def _get_m(d: Dict[str, Any], *path: str) -> Any:
    for k in path:
        d = d[k]
    return d


def build_perf_markdown(data: Dict[str, Any], run_tag: str, seed: int) -> str:
    sweep = data["lambda_sweep"]
    ood_keys = list(sweep[0][STAGES[0][0]][MODES[0][0]]["ood_eval"].keys())
    out: List[str] = []
    if int(seed) < 0:
        out.append(f"# `aggregate` (cross-seed mean) — `{run_tag}`")
    else:
        out.append(f"# `seed_{seed}` performance — `{run_tag}`")
    out.append("- Source: `metrics.json`")
    out.append("- Training stages (5 checkpoints, each with TF + AR eval).")
    out.append("")

    for mode_key, mode_label in MODES:
        out.append(f"## {mode_label} (in-distribution `X_test` + OOD by `n_digits`)\n")
        for stage_key, stage_title in STAGES:
            out.append(f"### {stage_title}\n")
            h = ["λ_carry"] + [f"ID:{SHORT.get(m, m)}" for m in METRIC_ORDER]
            for ok in ood_keys:
                nd = int(_get_m(sweep[0], stage_key, mode_key, "ood_eval", ok, "n_digits"))
                h += [f"OOD{nd}:{SHORT.get(m, m)}" for m in METRIC_ORDER]
            out.append("| " + " | ".join(h) + " |")
            out.append("|" + "|".join(["---"] * len(h)) + "|")
            for e in sweep:
                lc = e["lambda_carry"]
                row = [str(lc)]
                im = e[stage_key][mode_key]["id_metrics"]
                row += [f"{im[k]:.6g}" for k in METRIC_ORDER]
                for ok in ood_keys:
                    om = e[stage_key][mode_key]["ood_eval"][ok]["metrics"]
                    row += [f"{om[k]:.6g}" for k in METRIC_ORDER]
                out.append("| " + " | ".join(row) + " |")
            out.append("")

    out.append("---\n## Abbreviated (joint_tok, joint_seq) only\n")
    for mode_key, mode_label in MODES:
        out.append(f"### {mode_label}\n")
        h = [
            "λ_carry",
            "stage",
            "ID joint_tok",
            "ID joint_seq",
            "10 joint_tok",
            "10 joint_seq",
            "20 joint_tok",
            "20 joint_seq",
            "50 joint_tok",
            "50 joint_seq",
        ]
        out.append("| " + " | ".join(h) + " |")
        out.append("|" + "|".join(["---"] * len(h)) + "|")
        for e in sweep:
            lc = e["lambda_carry"]
            for stage_key, stage_title in STAGES:
                m = e[stage_key][mode_key]
                row = [str(lc), stage_key, f"{m['id_metrics']['joint_token_acc']:.6g}", f"{m['id_metrics']['joint_seq_acc']:.6g}"]
                for ok in ood_keys:
                    mm = m["ood_eval"][ok]["metrics"]
                    row += [f"{mm['joint_token_acc']:.6g}", f"{mm['joint_seq_acc']:.6g}"]
                out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def fl(x: float) -> str:
    return f"{float(x):.6g}"


def tex_stage(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def row_ar(
    sweep: List[Dict[str, Any]], lc: float, stg: str
) -> Tuple[Tuple[float, float, float, float], ...]:
    b = [e for e in sweep if float(e["lambda_carry"]) == float(lc)][0]
    m = b[stg]["autoregressive"]
    idm = m["id_metrics"]
    o10 = m["ood_eval"]["n_digits_10"]["metrics"]
    o20 = m["ood_eval"]["n_digits_20"]["metrics"]
    o50 = m["ood_eval"]["n_digits_50"]["metrics"]

    mfs = (idm["mean_first_wrong_sum_among_error_seq"], o10["mean_first_wrong_sum_among_error_seq"], o20["mean_first_wrong_sum_among_error_seq"], o50["mean_first_wrong_sum_among_error_seq"])
    mfc = (idm["mean_first_wrong_carry_among_error_seq"], o10["mean_first_wrong_carry_among_error_seq"], o20["mean_first_wrong_carry_among_error_seq"], o50["mean_first_wrong_carry_among_error_seq"])
    seqj = (idm["joint_seq_acc"], o10["joint_seq_acc"], o20["joint_seq_acc"], o50["joint_seq_acc"])
    seqs = (idm["sum_seq_acc"], o10["sum_seq_acc"], o20["sum_seq_acc"], o50["sum_seq_acc"])
    seqc = (idm["carry_seq_acc"], o10["carry_seq_acc"], o20["carry_seq_acc"], o50["carry_seq_acc"])
    tokj = (idm["joint_token_acc"], o10["joint_token_acc"], o20["joint_token_acc"], o50["joint_token_acc"])
    toks = (idm["sum_token_acc"], o10["sum_token_acc"], o20["sum_token_acc"], o50["sum_token_acc"])
    tokc = (idm["carry_token_acc"], o10["carry_token_acc"], o20["carry_token_acc"], o50["carry_token_acc"])
    return mfs, mfc, seqj, seqs, seqc, tokj, toks, tokc


def _is_mean_std_obj(v: Any) -> bool:
    return bool(
        isinstance(v, dict) and "mean" in v and "std" in v and "n" in v
    )


def _unwrap_metric_val(v: Any) -> float:
    if _is_mean_std_obj(v):
        return float(v["mean"])
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    raise TypeError(f"expected float or mean/std block, got {type(v).__name__} {v!r}")


def _demean_eval_mode(em: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id_metrics": {k: _unwrap_metric_val(t) for k, t in em["id_metrics"].items()},
        "ood_eval": {
            ok: {
                "n_digits": o["n_digits"],
                "n_test": o["n_test"],
                "metrics": {k: _unwrap_metric_val(t) for k, t in o["metrics"].items()},
            }
            for ok, o in em["ood_eval"].items()
        },
    }


def _demean_aggregate_sweep(agg_sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_sweep: List[Dict[str, Any]] = []
    for e in agg_sweep:
        row: Dict[str, Any] = {"lambda_carry": float(e["lambda_carry"])}
        for stg, _t in STAGES:
            st = e[stg]
            row[stg] = {
                "teacher_forcing": _demean_eval_mode(st["teacher_forcing"]),
                "autoregressive": _demean_eval_mode(st["autoregressive"]),
            }
        out_sweep.append(row)
    return {"lambda_sweep": out_sweep}


def _header_mfw(s_label: str, cap: str, run_line: str) -> List[str]:
    return [
        r"% \usepackage{longtable,booktabs}",
        r"\begin{longtable}{@{}r l *{8}{r}@{}}",
        rf"\caption{{Autoregressive: mean first wrong (sum and carry) among erroneous sequences. {run_line}, {cap}.}}\label{{tab:hybrid-ar-mfw-{s_label}}}\\",
        r"\toprule",
        r"$\lambda_{\mathrm{carry}}$ & Stage &",
        r"\multicolumn{4}{c}{mean wrong step (sum)} & \multicolumn{4}{c}{mean wrong step (carry)} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"& & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"$\lambda_{\mathrm{carry}}$ & Stage &",
        r"\multicolumn{4}{c}{sum} & \multicolumn{4}{c}{carry} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"& & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 \\",
        r"\midrule",
        r"\endhead",
        r"\endfoot",
        r"\endlastfoot",
    ]


def _header_seq_tok(
    which: str,
    s_label: str,
    cap: str,
    run_line: str,
    *,
    ncols: int = 12,
) -> List[str]:
    if which == "seq":
        cap_short = "sequence accuracies"
        jk, sk, ck = "joint\\_seq", "sum\\_seq", "carry\\_seq"
        lbl = f"hybrid-ar-seq-{s_label}"
    else:
        cap_short = "token accuracies"
        jk, sk, ck = "joint\\_tok", "sum\\_tok", "carry\\_tok"
        lbl = f"hybrid-ar-tok-{s_label}"
    return [
        r"% \usepackage{longtable,booktabs}",
        rf"\begin{{longtable}}{{@{{}}r l *{{{ncols}}}{{r}}@{{}}}}",
        rf"\caption{{Autoregressive: {cap_short}. {run_line}, {cap}.}}\label{{tab:{lbl}}}\\",
        r"\toprule",
        r"$\lambda_{\mathrm{carry}}$ & Stage &",
        rf"\multicolumn{{4}}{{c}}{{{jk}}} & \multicolumn{{4}}{{c}}{{{sk}}} & \multicolumn{{4}}{{c}}{{{ck}}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}",
        r"& & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"$\lambda_{\mathrm{carry}}$ & Stage &",
        r"\multicolumn{4}{c}{joint} & \multicolumn{4}{c}{sum} & \multicolumn{4}{c}{carry} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}",
        r"& & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 & ID & 10 & 20 & 50 \\",
        r"\midrule",
        r"\endhead",
        r"\endfoot",
        r"\endlastfoot",
    ]


def build_latex_ar_v2(
    data: Dict[str, Any],
    run_tag: str,
    seed: int,
    *,
    s_label: Optional[str] = None,
    run_line: Optional[str] = None,
) -> str:
    sweep = data["lambda_sweep"]
    stages = [s[0] for s in STAGES]
    lc_order = [float(e["lambda_carry"]) for e in sweep]
    s_lbl = s_label if s_label is not None else f"s{int(seed)}"
    run_ln = run_line if run_line is not None else f"Seed~{int(seed)}"
    cap = f"\\texttt{{{run_tag.replace('_', r'\_')}}}"

    parts: List[str] = []
    # Table 1: mfw
    parts.extend(_header_mfw(s_lbl, cap, run_ln))
    for lc in lc_order:
        for stg in stages:
            mfs, mfc, *_r = row_ar(sweep, lc, stg)
            a, b, c, d = mfs
            e, f, g, h = mfc
            parts.append(
                f"{lc} & {tex_stage(stg)} & {fl(a)} & {fl(b)} & {fl(c)} & {fl(d)} & {fl(e)} & {fl(f)} & {fl(g)} & {fl(h)} \\\\"
            )
    parts.append(r"\end{longtable}")
    parts.append("")
    parts.append(r"%% --- %%")
    parts.append("")

    # Table 2: seq
    parts.extend(_header_seq_tok("seq", s_lbl, cap, run_ln))
    for lc in lc_order:
        for stg in stages:
            _mfs, _mfc, seqj, seqs, seqc, _tj, _ts, _tc = row_ar(sweep, lc, stg)
            ja, j10, j20, j50 = seqj
            sa, s10, s20, s50 = seqs
            ca, c10, c20, c50 = seqc
            parts.append(
                f"{lc} & {tex_stage(stg)} & {fl(ja)} & {fl(j10)} & {fl(j20)} & {fl(j50)} & "
                f"{fl(sa)} & {fl(s10)} & {fl(s20)} & {fl(s50)} & "
                f"{fl(ca)} & {fl(c10)} & {fl(c20)} & {fl(c50)} \\\\"
            )
    parts.append(r"\end{longtable}")
    parts.append("")
    parts.append(r"%% --- token accuracies (AR) --- %%")
    parts.append("")

    # Table 3: tok
    parts.extend(_header_seq_tok("tok", s_lbl, cap, run_ln))
    for lc in lc_order:
        for stg in stages:
            _1, _2, _3, _4, _5, tokj, toks, tokc = row_ar(sweep, lc, stg)
            ja, j10, j20, j50 = tokj
            sa, s10, s20, s50 = toks
            ca, c10, c20, c50 = tokc
            parts.append(
                f"{lc} & {tex_stage(stg)} & {fl(ja)} & {fl(j10)} & {fl(j20)} & {fl(j50)} & "
                f"{fl(sa)} & {fl(s10)} & {fl(s20)} & {fl(s50)} & "
                f"{fl(ca)} & {fl(c10)} & {fl(c20)} & {fl(c50)} \\\\"
            )
    parts.append(r"\end{longtable}")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build PERF_TABLES_*.md and latex_AR_stats.tex from seed metrics or from aggregate in root metrics.json."
    )
    ap.add_argument("--sweep_dir", type=Path, required=True)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, read seed_*/metrics.json for this seed and write into that seed_*/",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Read top-level metrics.json, use aggregate.lambda_sweep (mean over seeds) into *_aggregate.*",
    )
    args = ap.parse_args()
    d = args.sweep_dir.resolve()
    run_tag = d.name
    if args.seed is None and not args.aggregate:
        ap.error("Provide at least one of --seed N and/or --aggregate")

    if args.seed is not None:
        p_metrics = d / f"seed_{args.seed}" / "metrics.json"
        if not p_metrics.is_file():
            raise SystemExit(f"Missing {p_metrics}")
        data = json.loads(p_metrics.read_text())
        (d / f"seed_{args.seed}").mkdir(parents=True, exist_ok=True)

        md = build_perf_markdown(data, run_tag, args.seed)
        p_md = d / f"seed_{args.seed}" / f"PERF_TABLES_seed{args.seed}.md"
        p_md.write_text(md)
        print(f"Wrote {p_md} ({md.count(chr(10))} lines)")

        tex = build_latex_ar_v2(data, run_tag, args.seed)
        p_tex = d / f"seed_{args.seed}" / "latex_AR_stats.tex"
        p_tex.write_text(tex)
        print(f"Wrote {p_tex} ({len(tex.splitlines())} lines)")

    if args.aggregate:
        p_root = d / "metrics.json"
        if not p_root.is_file():
            raise SystemExit(f"Missing {p_root}")
        root = json.loads(p_root.read_text())
        agg = root.get("aggregate")
        if not agg or "lambda_sweep" not in agg:
            raise SystemExit("metrics.json has no aggregate.lambda_sweep (run a multi-seed hybrid sweep to produce it).")
        n_s = int(agg["n_seeds"])
        data_agg = _demean_aggregate_sweep(agg["lambda_sweep"])
        run_line = f"Cross-seed mean; $n_{{\\mathrm{{seeds}}}}={n_s}$ (table entries: mean of runs, std not shown)"

        md = build_perf_markdown(data_agg, run_tag, -1)
        p_md = d / "PERF_TABLES_aggregate.md"
        p_md.write_text(md)
        print(f"Wrote {p_md} ({md.count(chr(10))} lines)")

        tex = build_latex_ar_v2(
            data_agg,
            run_tag,
            0,
            s_label="agg",
            run_line=run_line,
        )
        p_tex = d / "latex_AR_stats_aggregate.tex"
        p_tex.write_text(tex)
        print(f"Wrote {p_tex} ({len(tex.splitlines())} lines)")


if __name__ == "__main__":
    main()
