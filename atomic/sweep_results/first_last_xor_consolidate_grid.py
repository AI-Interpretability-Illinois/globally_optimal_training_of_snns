#!/usr/bin/env python3
"""
Aggregate first_last_xor Simple Run Summary .md files into L x T tables.
Reads sweep_results/first_last_xor_{L}_{T}/*.md and writes a consolidated .md report.

L=3, T=11 has no .md here — uses *.json: STE = mean ± std over ste_only runs;
CVX = single best test_last_step_acc over cvx_only runs (metric_aggregate.json skipped).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = "first_last_xor_"
T_GRID = [3, 6, 8, 11, 14]
L_GRID = [3, 5, 10, 15]

SKIP_JSON_NAMES = frozenset({"metric_aggregate.json"})


def parse_test_accs(md_text: str) -> dict[str, float]:
    """Extract STE and CVX test accuracies from Simple Run Summary table."""
    lines = md_text.splitlines()
    out: dict[str, float] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            continue
        key = parts[0]
        if key == "STE":
            acc_s = parts[-1]
            out["STE"] = float(acc_s)
        elif key.startswith("CVX"):
            acc_s = parts[-1]
            out["CVX"] = float(acc_s)
    if "STE" not in out or "CVX" not in out:
        keys = set(out.keys())
        raise ValueError(f"Missing STE/CVX row in summary table (got keys {keys})")
    return out


def parse_simple_json(path: Path) -> tuple[str, float] | None:
    """Return (\"STE\"|\"CVX\", acc) from simple sweep json, or None if skipped."""
    if path.name in SKIP_JSON_NAMES:
        return None
    data = json.loads(path.read_text())
    side = data["simple_side"]
    if side == "ste_only":
        return "STE", float(data["ste"]["test_last_step_acc"])
    if side == "cvx_only":
        return "CVX", float(data["cvx_gaussian"]["test_last_step_acc"])
    raise ValueError(f"Unexpected simple_side={side!r} in {path}")


def collect_cell_l3_t11_from_json(d: Path) -> dict:
    """No .md: aggregate from json. CVX = best run only (user request)."""
    ste_vals: list[float] = []
    cvx_vals: list[float] = []
    for js_path in sorted(d.glob("*.json")):
        got = parse_simple_json(js_path)
        if got is None:
            continue
        kind, acc = got
        if kind == "STE":
            ste_vals.append(acc)
        else:
            cvx_vals.append(acc)
    if not ste_vals:
        raise ValueError(f"No ste_only JSON in {d}")
    if not cvx_vals:
        raise ValueError(f"No cvx_only JSON in {d}")
    best_cvx = max(cvx_vals)
    return {
        "ste": ste_vals,
        "cvx": [best_cvx],
        "cvx_best_of_n": len(cvx_vals),
    }


def collect_cell(L: int, T: int) -> dict:
    """
    Keys: ste (list[float]), cvx (list[float]), optional cvx_best_of_n (int).
    """
    d = ROOT / f"{PREFIX}{L}_{T}"
    empty = {"ste": [], "cvx": [], "cvx_best_of_n": None}
    if not d.is_dir():
        return empty
    md_files = sorted(d.glob("*.md"))
    if md_files:
        ste: list[float] = []
        cvx: list[float] = []
        for md_path in md_files:
            accs = parse_test_accs(md_path.read_text())
            ste.append(accs["STE"])
            cvx.append(accs["CVX"])
        return {"ste": ste, "cvx": cvx, "cvx_best_of_n": None}
    if L == 3 and T == 11:
        return collect_cell_l3_t11_from_json(d)
    return empty


def fmt_mean_std(xs: list[float]) -> str:
    if not xs:
        return "—"
    m = statistics.mean(xs)
    if len(xs) == 1:
        return f"{m:.4f} ± 0.0000 (n=1)"
    s = statistics.stdev(xs)
    return f"{m:.4f} ± {s:.4f} (n={len(xs)})"


def fmt_cvx_cell(cd: dict) -> str:
    if not cd["cvx"]:
        return "—"
    n_best = cd.get("cvx_best_of_n")
    if n_best is not None:
        assert len(cd["cvx"]) == 1
        return f"{cd['cvx'][0]:.4f} (best of {n_best} CVX json runs)"
    return fmt_mean_std(cd["cvx"])


def extend_pooled(cd: dict, ste_all: list[float], cvx_all: list[float]) -> None:
    ste_all.extend(cd["ste"])
    cvx_all.extend(cd["cvx"])


def main() -> None:
    cells: dict[tuple[int, int], dict] = {}
    for L in L_GRID:
        for T in T_GRID:
            cells[(L, T)] = collect_cell(L, T)

    lines: list[str] = []
    lines.append("# first_last_xor — consolidated test accuracy (.md summaries + L=3,T=11 JSON)")
    lines.append("")
    lines.append(
        f"Grid: **L** (readout/hidden width) ∈ {L_GRID}, **T** (sequence length) ∈ {T_GRID}."
    )
    lines.append(
        "Each cell is **mean ± sample std** over all `.md` runs in "
        "`sweep_results/first_last_xor_{L}_{T}/` when present; `(n=…)` is the number of runs."
    )
    lines.append(
        "**L=3, T=11:** no `.md` in this repo — values from `*.json` using `test_last_step_acc`: "
        "STE = mean ± std over all `ste_only` files; **CVX = best** accuracy among all `cvx_only` files "
        f"(`metric_aggregate.json` skipped)."
    )
    lines.append("")
    lines.append(
        "**Note:** No `first_last_xor_*_3` dirs and no L=3,T=8 `.md` here — those cells stay empty."
    )
    lines.append("")

    for model_key, title, fmt_fn in (
        ("ste", "STE", lambda cd: fmt_mean_std(cd["ste"])),
        ("cvx", "CVX", fmt_cvx_cell),
    ):
        lines.append(f"## {title} — test accuracy")
        lines.append("")
        hdr = "| L \\ T | " + " | ".join(str(T) for T in T_GRID) + " |"
        sep = "|" + "|".join(["---:"] + [":---:"] * len(T_GRID)) + "|"
        lines.append(hdr)
        lines.append(sep)
        for L in L_GRID:
            row = [str(L)]
            for T in T_GRID:
                cd = cells[(L, T)]
                row.append(fmt_fn(cd))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    t_set = ", ".join(str(t) for t in T_GRID)
    lines.append(f"## Per-layer aggregate over T ∈ {{{t_set}}}")
    lines.append("")
    lines.append(
        "Pooled **all** run-level accuracies that feed each cell (for L=3,T=11 CVX, only the **single best** CVX value). "
        "Mean ± sample std over the pooled lists."
    )
    lines.append("")
    hdr = "| L | STE | CVX | pooled n (STE / CVX) |"
    sep = "|---:|---:|---:|---:|"
    lines.append(hdr)
    lines.append(sep)
    for L in L_GRID:
        ste_all: list[float] = []
        cvx_all: list[float] = []
        for T in T_GRID:
            extend_pooled(cells[(L, T)], ste_all, cvx_all)
        ste_s = fmt_mean_std(ste_all) if ste_all else "—"
        cvx_s = fmt_mean_std(cvx_all) if cvx_all else "—"
        lines.append(
            f"| {L} | {ste_s} | {cvx_s} | {len(ste_all)} / {len(cvx_all)} |"
        )

    out_path = ROOT / "first_last_xor_LxT_accuracy_consolidated.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
