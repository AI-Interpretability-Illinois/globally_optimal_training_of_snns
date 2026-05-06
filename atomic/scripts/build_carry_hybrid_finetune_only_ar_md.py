#!/usr/bin/env python3
"""Emit AR-only markdown summary from seed_*/metrics_finetune_only.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "metrics_json",
        type=Path,
        help="Path to metrics_finetune_only.json",
    )
    ap.add_argument(
        "--out_md",
        type=Path,
        default=None,
        help="Output .md path (default: alongside JSON as *_AR_summary.md)",
    )
    args = ap.parse_args()
    p = args.metrics_json.expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))

    OOD_KEYS = [
        ("n_digits_10", "OOD ×2 (10 digits)"),
        ("n_digits_25", "OOD ×5 (25 digits)"),
        ("n_digits_50", "OOD ×10 (50 digits)"),
    ]

    def fmt(x: float) -> str:
        return f"{float(x):.4f}"

    lines: list[str] = []
    lines.append("# Carry hybrid `finetune_only` — autoregressive summary")
    lines.append("")
    lines.append(f"- **Source:** `{p}`")
    lines.append(f"- **Seed:** {data.get('seed')}")
    lines.append(f"- **Finetune variant:** `{data.get('finetune_variant')}`")
    lines.append(
        f"- **Loaded pretrain:** `{data['lambda_sweep'][0].get('loaded_pretrain_variant')}` (checkpoint before STE fine-tune)"
    )
    lines.append("")
    lines.append(
        "Training uses **5-digit** ID adds; OOD columns are **10 / 25 / 50** digits (×2 / ×5 / ×10 vs train length)."
    )
    lines.append("")

    best_j, best_lc = -1.0, None
    for e in data["lambda_sweep"]:
        j = float(e["pretrain_eval"]["autoregressive"]["id_metrics"]["joint_token_acc"])
        if j > best_j:
            best_j, best_lc = j, float(e["lambda_carry"])

    lines.append(
        f"**Best CVX-only (pretrain eval) AR ID `joint_token_acc`:** {best_j:.6f} at **`lambda_carry = {best_lc}`**."
    )
    lines.append("")

    for phase_key, phase_title in [
        ("pretrain_eval", "A. CVX pretrained checkpoint (before STE fine-tune)"),
        ("ste_finetune", "B. After STE fine-tune from that CVX init"),
    ]:
        lines.append(f"## {phase_title}")
        lines.append("")

        def emit_table(keys: list[tuple[str, str]], title_suffix: str) -> None:
            lines.append(f"### {title_suffix}")
            h = ["`lambda_carry`"]
            for _, lbl in keys:
                h.append(f"ID `{lbl}`")
            for _ok, olabel in OOD_KEYS:
                for _, lbl in keys:
                    h.append(f"{olabel} `{lbl}`")
            lines.append("| " + " | ".join(h) + " |")
            lines.append("| " + " | ".join(["---"] * len(h)) + " |")
            for e in data["lambda_sweep"]:
                lc = e["lambda_carry"]
                ar = e[phase_key]["autoregressive"]
                idm = ar["id_metrics"]
                ood = ar["ood_eval"]
                cells = [str(lc)]
                for k, _ in keys:
                    cells.append(fmt(idm[k]))
                for ok, _ in OOD_KEYS:
                    m = ood[ok]["metrics"]
                    for k, _ in keys:
                        cells.append(fmt(m[k]))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        emit_table(
            [("sum_token_acc", "sum tok"), ("carry_token_acc", "carry tok"), ("joint_token_acc", "joint tok")],
            "Token accuracy",
        )
        emit_table(
            [("joint_seq_acc", "joint seq"), ("sum_seq_acc", "sum seq"), ("carry_seq_acc", "carry seq")],
            "Sequence accuracy",
        )
        emit_table(
            [
                ("mean_first_wrong_sum_among_error_seq", "mfw sum"),
                ("mean_first_wrong_carry_among_error_seq", "mfw carry"),
            ],
            "Mean first wrong (among sequences with errors)",
        )

    lines.append("---")
    lines.append("")
    lines.append("*AR = `autoregressive` rollout; teacher forcing not shown.*")

    out = args.out_md
    if out is None:
        out = p.parent / (p.stem + "_AR_summary.md")
    else:
        out = out.expanduser().resolve()
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
