#!/usr/bin/env python3
"""
Parse snn_logs.txt and generate:
1. A summary table: task, T, L, CVX test acc, SNN test acc (mean ± std over seeds)
2. Training curves per task: overlay CVX (with beta, lr) and SNN curves
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path(__file__).resolve().parent / "snn_logs.txt"
OUT_DIR = Path(__file__).resolve().parent / "plots"
TABLE_PATH = Path(__file__).resolve().parent / "results_table.md"


def parse_log(path: Path):
    """Parse log file into runs. Each run has task, T, L, loss?, and list of seeds with curves and test accs."""
    text = path.read_text()
    lines = text.splitlines()

    re_task = re.compile(r"\[info\] task=(\S+)\s+T=(\d+)\s+L=(\d+)")
    re_retrain = re.compile(r"\[seed\s+(\d+)\]\s+re-train CVX best for logging:\s+beta=(\S+)\s+lr=([\d.e-]+)")
    re_cvx = re.compile(r"\[CVX\]\s+ep=(\d+)/(\d+)\s+train_score=([\d.]+)\s+val_score=([\d.]+)")
    re_snn = re.compile(r"\[STE-SNN\]\s+ep=(\d+)/(\d+)\s+train_score=([\d.]+)\s+val_score=([\d.]+)")
    re_test = re.compile(
        r"\[seed\s+(\d+)\]\s+CVX test_acc=([\d.]+)\s+\(val=[\d.]+, beta_l1=([^,]+), lr=([^)]+)\)\s+\|\s+STE-SNN test_acc=([\d.]+)"
    )

    runs = []
    i = 0
    current_run = None
    while i < len(lines):
        line = lines[i]
        m_task = re_task.search(line)
        if m_task:
            task, T, L = m_task.group(1), int(m_task.group(2)), int(m_task.group(3))
            ctx_start = max(0, i - 30)
            ctx_end = min(len(lines), i + 15)
            loss = "hinge"
            for j in range(ctx_start, ctx_end):
                if "loss=ce" in lines[j]:
                    loss = "ce"
                    break
                if "loss=hinge" in lines[j]:
                    loss = "hinge"
                    break
            key = (task, T, L, loss)
            if current_run is not None and (current_run["task"], current_run["T"], current_run["L"], current_run["loss"]) != (task, T, L, loss):
                if current_run["seeds"]:
                    runs.append(current_run)
                current_run = None
            if current_run is None:
                current_run = {"task": task, "T": T, "L": L, "loss": loss, "seeds": []}
            run = current_run
            i += 1
            while i < len(lines):
                ln = lines[i]
                m_retrain = re_retrain.search(ln)
                m_test = re_test.search(ln)
                m_other_task = re_task.search(ln)
                if m_other_task:
                    ot, oT, oL = m_other_task.group(1), int(m_other_task.group(2)), int(m_other_task.group(3))
                    other_ctx_start = max(0, i - 30)
                    other_ctx_end = min(len(lines), i + 15)
                    other_loss = "hinge"
                    for j in range(other_ctx_start, other_ctx_end):
                        if "loss=ce" in lines[j]:
                            other_loss = "ce"
                            break
                        if "loss=hinge" in lines[j]:
                            other_loss = "hinge"
                            break
                    if (ot, oT, oL, other_loss) != (run["task"], run["T"], run["L"], run["loss"]):
                        break
                if m_retrain:
                    seed_id, beta, lr = m_retrain.group(1), m_retrain.group(2), m_retrain.group(3)
                    cvx_ep, cvx_train, cvx_val = [], [], []
                    i += 1
                    while i < len(lines) and re_cvx.match(lines[i]):
                        mm = re_cvx.match(lines[i])
                        cvx_ep.append(int(mm.group(1)))
                        cvx_train.append(float(mm.group(3)))
                        cvx_val.append(float(mm.group(4)))
                        i += 1
                    snn_ep, snn_train, snn_val = [], [], []
                    while i < len(lines) and re_snn.match(lines[i]):
                        mm = re_snn.match(lines[i])
                        snn_ep.append(int(mm.group(1)))
                        snn_train.append(float(mm.group(3)))
                        snn_val.append(float(mm.group(4)))
                        i += 1
                    cvx_acc = snn_acc = None
                    if i < len(lines):
                        m_t = re_test.search(lines[i])
                        if m_t and m_t.group(1) == seed_id:
                            cvx_acc = float(m_t.group(2))
                            snn_acc = float(m_t.group(5))
                            i += 1
                    run["seeds"].append({
                        "seed": seed_id,
                        "beta": beta,
                        "lr": lr,
                        "cvx_epoch": np.array(cvx_ep) if cvx_ep else None,
                        "cvx_train": np.array(cvx_train) if cvx_train else None,
                        "cvx_val": np.array(cvx_val) if cvx_val else None,
                        "snn_epoch": np.array(snn_ep) if snn_ep else None,
                        "snn_train": np.array(snn_train) if snn_train else None,
                        "snn_val": np.array(snn_val) if snn_val else None,
                        "cvx_test_acc": cvx_acc,
                        "snn_test_acc": snn_acc,
                    })
                    continue
                if m_test and run["seeds"] and run["seeds"][-1].get("cvx_test_acc") is None:
                    run["seeds"][-1]["cvx_test_acc"] = float(m_test.group(2))
                    run["seeds"][-1]["snn_test_acc"] = float(m_test.group(5))
                    run["seeds"][-1]["beta"] = m_test.group(3)
                    run["seeds"][-1]["lr"] = m_test.group(4)
                    i += 1
                    continue
                i += 1
            continue
        i += 1
    if current_run is not None and current_run["seeds"]:
        runs.append(current_run)
    return runs


def build_table(runs):
    """Aggregate runs by (task, T, L) and compute mean ± std test acc."""
    key_to_seeds = defaultdict(list)
    for r in runs:
        key = (r["task"], r["T"], r["L"], r.get("loss", ""))
        for s in r["seeds"]:
            if s.get("cvx_test_acc") is not None and s.get("snn_test_acc") is not None:
                key_to_seeds[key].append((s["cvx_test_acc"], s["snn_test_acc"]))

    rows = []
    for (task, T, L, loss), accs in sorted(key_to_seeds.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if not accs:
            continue
        cvx_accs = [a[0] for a in accs]
        snn_accs = [a[1] for a in accs]
        cvx_mean, cvx_std = np.mean(cvx_accs), np.std(cvx_accs)
        snn_mean, snn_std = np.mean(snn_accs), np.std(snn_accs)
        loss_suffix = f" ({loss})" if loss else ""
        rows.append({
            "task": task + loss_suffix,
            "T": T,
            "L": L,
            "cvx_mean": cvx_mean,
            "cvx_std": cvx_std,
            "snn_mean": snn_mean,
            "snn_std": snn_std,
            "n_seeds": len(accs),
        })
    return rows


def write_table_md(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("| Task | T | L | CVX test acc | SNN test acc |\n")
        f.write("|------|---|---|---------------|---------------|\n")
        for r in rows:
            cvx_str = f"{r['cvx_mean']:.4f} ± {r['cvx_std']:.4f}" if r["n_seeds"] > 1 else f"{r['cvx_mean']:.4f}"
            snn_str = f"{r['snn_mean']:.4f} ± {r['snn_std']:.4f}" if r["n_seeds"] > 1 else f"{r['snn_mean']:.4f}"
            f.write(f"| {r['task']} | {r['T']} | {r['L']} | {cvx_str} | {snn_str} |\n")
    print(f"Wrote {path}")


def plot_training_curves(runs, out_dir: Path):
    """One figure per (task, T, L): overlay CVX (with beta, lr) and SNN curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Group runs by (task, T, L) and collect all seeds' curves
    key_to_run = {}
    for r in runs:
        key = (r["task"], r["T"], r["L"])
        if key not in key_to_run:
            key_to_run[key] = {"task": r["task"], "T": r["T"], "L": r["L"], "seeds": []}
        key_to_run[key]["seeds"].extend(r["seeds"])

    for (task, T, L), data in key_to_run.items():
        seeds = [
            s for s in data["seeds"]
            if s.get("cvx_epoch") is not None
            and len(s["cvx_epoch"]) > 0
            and s.get("snn_epoch") is not None
            and len(s["snn_epoch"]) > 0
        ]
        if not seeds:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        # Plot train accuracy - overlay CVX with beta/lr and SNN
        for idx, s in enumerate(seeds):
            beta, lr = s.get("beta", "?"), s.get("lr", "?")
            label_cvx = f"CVX β={beta} lr={lr}" if len(seeds) > 1 else f"CVX (β={beta}, lr={lr})"
            ax.plot(s["cvx_epoch"], s["cvx_train"], alpha=0.8, label=label_cvx)
        for idx, s in enumerate(seeds):
            ax.plot(s["snn_epoch"], s["snn_train"], "--", alpha=0.9, color="C1", label="SNN" if idx == 0 else None)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train accuracy")
        ax.set_title(f"{task}  T={T}  L={L}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        safe = re.sub(r"[^\w-]", "_", f"{task}_T{T}_L{L}")[:60]
        fig.savefig(out_dir / f"train_{safe}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"train_{safe}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved train_{safe}.pdf, train_{safe}.png")


def main():
    runs = parse_log(LOG_PATH)
    print(f"Parsed {len(runs)} run(s), total seeds: {sum(len(r['seeds']) for r in runs)}")
    rows = build_table(runs)
    write_table_md(rows, TABLE_PATH)
    plot_training_curves(runs, OUT_DIR)


if __name__ == "__main__":
    main()
