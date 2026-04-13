#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_PATH = Path("/Users/hima_3114/Desktop/Paper_1/experiments/cvx_snn_vs_ste_arithmetic_seq_results.csv")
OUT_PATH = Path("/Users/hima_3114/Desktop/Paper_1/experiments/snn_generalized_pt2/base2_add_digits8_seq_acc_vs_sizes.png")

# Must match the sweep launcher order.
TRAINS = [512, 2304, 4096, 5888, 7680]
TESTS = [256, 512, 768, 1024, 1280]


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    rows = []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == "arith_seq::add::base2::digits8":
                rows.append(row)

    if not rows:
        raise ValueError("No rows found for dataset=arith_seq::add::base2::digits8.")

    # Keep up to 5 latest rows, matching the 5-step sweep progression.
    rows = rows[-5:]
    n_points = len(rows)

    x_labels = []
    cvx_train_seq = []
    cvx_test_seq = []
    ste_train_seq = []
    ste_test_seq = []

    for i, row in enumerate(rows):
        n_train = TRAINS[i]
        n_test = TESTS[i]
        n_val = n_train // 5
        x_labels.append(f"({n_train},{n_val},{n_test})")

        cvx_train_seq.append(float(row["cvx_seq_train_acc_mean"]))
        cvx_test_seq.append(float(row["cvx_seq_test_acc_mean"]))
        ste_train_seq.append(float(row["ste_seq_train_acc_mean"]))
        ste_test_seq.append(float(row["ste_seq_test_acc_mean"]))

    x = list(range(n_points))

    plt.figure(figsize=(12, 6))
    plt.plot(x, cvx_train_seq, linestyle="-", linewidth=2.2, marker="o", label="CVX train seq acc")
    plt.plot(x, cvx_test_seq, linestyle="-", linewidth=2.2, marker="o", label="CVX test seq acc")
    plt.plot(x, ste_train_seq, linestyle="--", linewidth=2.2, marker="s", label="STE train seq acc")
    plt.plot(x, ste_test_seq, linestyle="--", linewidth=2.2, marker="s", label="STE test seq acc")

    plt.xticks(x, x_labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("(training,val,test)-size")
    plt.ylabel("Sequence Accuracy")
    plt.title("base 2 addition of 8 digits")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180)
    print(f"saved plot: {OUT_PATH}")


if __name__ == "__main__":
    main()
