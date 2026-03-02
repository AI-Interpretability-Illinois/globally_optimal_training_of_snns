#!/usr/bin/env python3
"""
Single-node 8×A100 parallel grid search over (L, T, init_method, beta_dist).

Assumes you already have the node allocated. Just run:
  python run_two_step_xor_grid.py [--num-gpus 8] [--out-dir results_grid]

Each config is pinned to a GPU round-robin via CUDA_VISIBLE_DEVICES.
8 workers run concurrently (one per GPU), pulling from a shared queue.
Results saved as JSON per config + final summary CSV.

Options:
  --dry-run        Print configs without running
  --collect-only   Skip running, just aggregate existing results
  --num-gpus N     Number of GPUs (default: 8)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

# ── Grid axes ──────────────────────────────────────────────
T_LIST = [2, 6, 10, 18, 26, 42, 54, 70, 86, 106]
L_LIST = [3, 5, 9, 15, 23, 33, 45, 59, 77, 97, 119]
INIT_METHODS = ["random", "micheli", "lognormal", "lsuv"]
BETA_DISTS = ["fixed", "het_loguniform", "het_uniform", "het_bimodal"]

# Reference widths at T=6
T_REF = 6
P_IN_BASE = 2000
P_REC_BASE = 2000
P_LAST_BASE = 5000

# Fixed training hyperparams
SEEDS = [0]
EPOCHS = 200
LR_GRID = [1e-3]
LAST_LAYER_READOUT = "membrane"


def scale_factor(T: int) -> float:
    return max(T, T_REF) / min(T, T_REF) if T > T_REF else 1.0


def scaled_widths(T: int):
    f = scale_factor(T)
    return (
        max(1, round(P_IN_BASE * f)),
        max(1, round(P_REC_BASE * f)),
        max(1, round(P_LAST_BASE * f)),
    )


def build_configs() -> list[dict]:
    configs = []
    idx = 0
    for L, T, init_m, beta_d in product(L_LIST, T_LIST, INIT_METHODS, BETA_DISTS):
        P_in, P_rec, P_last = scaled_widths(T)
        configs.append({
            "id": idx,
            "L": L,
            "T": T,
            "P_in": P_in,
            "P_rec": P_rec,
            "P_last": P_last,
            "init_method": init_m,
            "beta_dist": beta_d,
        })
        idx += 1
    return configs


def run_one_config(cfg: dict, gpu_id: int, out_dir: str, script_dir: str) -> dict:
    """Run a single config on a specific GPU. Called in a worker process."""
    L = cfg["L"]
    T = cfg["T"]
    P_in = cfg["P_in"]
    P_rec = cfg["P_rec"]
    P_last = cfg["P_last"]
    init_method = cfg["init_method"]
    beta_dist = cfg["beta_dist"]
    config_id = cfg["id"]

    result_name = f"L{L}_T{T}_{init_method}_{beta_dist}.json"
    result_path = os.path.join(out_dir, "results", result_name)

    # Skip if already completed successfully
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                existing = json.load(f)
            if existing.get("cvx_test") is not None:
                return existing
        except (json.JSONDecodeError, KeyError):
            pass  # re-run corrupt results

    snn_py = os.path.join(script_dir, "snn_p2.py")
    metrics_path = os.path.join(out_dir, "results", f"metrics_{result_name}")

    cmd = [
        sys.executable, snn_py,
        "--task", "two_step_xor_seq",
        "--T", str(T),
        "--L", str(L),
        "--P_in", str(P_in),
        "--P_rec", str(P_rec),
        "--P_last", str(P_last),
        "--init_method", init_method,
        "--beta_dist", beta_dist,
        "--last_layer_readout", LAST_LAYER_READOUT,
        "--seeds", *[str(s) for s in SEEDS],
        "--epochs", str(EPOCHS),
        "--lr_grid", *[str(v) for v in LR_GRID],
        "--log_train",
        "--verbose_patterns",
        "--save_metrics_path", metrics_path,
        "--device", f"cuda",
    ]

    # Pin to exactly one GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0

    lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
    killed = lines[-1].strip() == "kill" if lines else False

    # Parse scores from stdout
    cvx_test = None
    ste_test = None
    for line in reversed(lines):
        if "CVX" in line and "test_acc=" in line:
            try:
                cvx_test = float(line.split("test_acc=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "STE" in line and "test_acc=" in line:
            try:
                ste_test = float(line.split("test_acc=")[1].split()[0].rstrip(")"))
            except (IndexError, ValueError):
                pass
        if cvx_test is not None and ste_test is not None:
            break

    result = {
        "config_id": config_id,
        "L": L,
        "T": T,
        "P_in": P_in,
        "P_rec": P_rec,
        "P_last": P_last,
        "init_method": init_method,
        "beta_dist": beta_dist,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "cvx_test": cvx_test,
        "ste_test": ste_test,
        "killed": killed,
        "gpu_id": gpu_id,
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Write stdout log
    log_path = os.path.join(out_dir, "logs", f"{config_id:04d}_L{L}_T{T}_{init_method}_{beta_dist}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(proc.stdout or "")

    return result


def print_summary(results: list[dict]):
    """Print final summary table."""
    import numpy as np

    valid = [r for r in results if r.get("cvx_test") is not None]
    failed = [r for r in results if r.get("cvx_test") is None]

    print(f"\n{'='*90}")
    print(f"COMPLETED: {len(valid)}  |  FAILED: {len(failed)}  |  TOTAL: {len(results)}")
    print(f"{'='*90}")

    if not valid:
        return

    # ── Best per (L, T) ──
    by_lt = defaultdict(list)
    for r in valid:
        by_lt[(r["L"], r["T"])].append(r)

    print(f"\nBEST config per (L, T) by cvx_test:")
    print(f"{'L':>4} {'T':>4} {'init':>12} {'beta_dist':>16} {'cvx':>8} {'ste':>8} {'time':>7}")
    print("-" * 70)

    for (L, T) in sorted(by_lt.keys()):
        best = max(by_lt[(L, T)], key=lambda r: r["cvx_test"])
        ste = f"{best['ste_test']:.4f}" if best.get("ste_test") else "N/A"
        print(f"{L:>4} {T:>4} {best['init_method']:>12} {best['beta_dist']:>16} "
              f"{best['cvx_test']:>8.4f} {ste:>8} {best.get('elapsed_seconds',0):>6.0f}s")

    # ── Aggregate by beta_dist ──
    above = [r for r in valid if r["cvx_test"] > 0.55]
    print(f"\nMean cvx_test by beta_dist (configs with cvx > 55%, n={len(above)}):")
    by_beta = defaultdict(list)
    for r in above:
        by_beta[r["beta_dist"]].append(r["cvx_test"])
    for bd in BETA_DISTS:
        vals = by_beta.get(bd, [])
        if vals:
            arr = np.array(vals)
            print(f"  {bd:>16}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(vals)})")
        else:
            print(f"  {bd:>16}: —")

    # ── Aggregate by init_method ──
    print(f"\nMean cvx_test by init_method (configs with cvx > 55%):")
    by_init = defaultdict(list)
    for r in above:
        by_init[r["init_method"]].append(r["cvx_test"])
    for im in INIT_METHODS:
        vals = by_init.get(im, [])
        if vals:
            arr = np.array(vals)
            print(f"  {im:>12}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(vals)})")
        else:
            print(f"  {im:>12}: —")

    # ── Critical T threshold: at what T does each beta_dist collapse? ──
    print(f"\nMean cvx_test by (T, beta_dist) — collapse threshold:")
    print(f"{'T':>4}", end="")
    for bd in BETA_DISTS:
        print(f"  {bd:>16}", end="")
    print()
    print("-" * (4 + 18 * len(BETA_DISTS)))

    by_t_beta = defaultdict(list)
    for r in valid:
        by_t_beta[(r["T"], r["beta_dist"])].append(r["cvx_test"])

    for T in T_LIST:
        print(f"{T:>4}", end="")
        for bd in BETA_DISTS:
            vals = by_t_beta.get((T, bd), [])
            if vals:
                arr = np.array(vals)
                m = arr.mean()
                marker = " ✗" if m < 0.55 else ""
                print(f"  {m:>14.4f}{marker}", end="")
            else:
                print(f"  {'—':>16}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="8×A100 parallel grid: (L, T, init_method, beta_dist) for two_step_xor_seq"
    )
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs on this node (default: 8)")
    parser.add_argument("--out-dir", type=str, default="results_grid",
                        help="Output directory for results/logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--collect-only", action="store_true",
                        help="Skip running, just collect + summarize existing results")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    configs = build_configs()
    print(f"Grid: {len(L_LIST)} L × {len(T_LIST)} T × {len(INIT_METHODS)} init × "
          f"{len(BETA_DISTS)} beta = {len(configs)} configs")
    print(f"GPUs: {args.num_gpus}  |  Output: {out_dir}")

    # Save configs CSV
    csv_path = os.path.join(out_dir, "configs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=configs[0].keys())
        writer.writeheader()
        writer.writerows(configs)

    if args.dry_run:
        print("\n[dry-run] First 16 configs (one full T=2 block):")
        for c in configs[:16]:
            gpu = c["id"] % args.num_gpus
            print(f"  GPU{gpu} | id={c['id']:>4} L={c['L']:>3} T={c['T']:>3} "
                  f"init={c['init_method']:<10} beta={c['beta_dist']:<16} "
                  f"P=({c['P_in']},{c['P_rec']},{c['P_last']})")
        print(f"  ... ({len(configs)} total)")
        return

    if args.collect_only:
        results = []
        results_dir = os.path.join(out_dir, "results")
        for fname in sorted(os.listdir(results_dir)):
            if fname.startswith("L") and fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    results.append(json.load(f))
        print_summary(results)
        # Save summary CSV
        if results:
            keys = ["L", "T", "init_method", "beta_dist", "cvx_test", "ste_test",
                    "killed", "elapsed_seconds", "P_in", "P_rec", "P_last", "gpu_id"]
            csv_out = os.path.join(out_dir, "summary.csv")
            with open(csv_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            print(f"\nCSV → {csv_out}")
        return

    # ── Run all configs across GPUs ──
    num_gpus = args.num_gpus
    all_results = []
    done = 0
    t_start = time.time()

    # ProcessPoolExecutor with num_gpus workers.
    # Each submitted task gets gpu_id = config_id % num_gpus.
    # The executor handles queuing: 8 workers pull from ~1760 tasks.
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_cfg = {}
        for cfg in configs:
            gpu_id = cfg["id"] % num_gpus
            future = executor.submit(run_one_config, cfg, gpu_id, out_dir, script_dir)
            future_to_cfg[future] = cfg

        for future in as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            done += 1
            try:
                result = future.result()
                all_results.append(result)
                cvx = result.get("cvx_test")
                cvx_str = f"{cvx:.4f}" if cvx is not None else "FAIL"
                elapsed_total = time.time() - t_start
                rate = done / elapsed_total * 3600
                eta_hr = (len(configs) - done) / rate if rate > 0 else 0
                print(f"[{done:>4}/{len(configs)}] "
                      f"L={cfg['L']:>3} T={cfg['T']:>3} "
                      f"init={cfg['init_method']:<10} beta={cfg['beta_dist']:<16} "
                      f"cvx={cvx_str}  GPU{result.get('gpu_id','')}  "
                      f"{result.get('elapsed_seconds',0):.0f}s  "
                      f"(ETA {eta_hr:.1f}hr)",
                      flush=True)
            except Exception as e:
                all_results.append({
                    **cfg, "cvx_test": None, "ste_test": None, "error": str(e)
                })
                print(f"[{done:>4}/{len(configs)}] FAILED "
                      f"L={cfg['L']} T={cfg['T']} "
                      f"init={cfg['init_method']} beta={cfg['beta_dist']}: {e}",
                      flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal wall time: {total_time/3600:.2f} hours")

    # ── Summary ──
    print_summary(all_results)

    # Save CSV
    keys = ["L", "T", "init_method", "beta_dist", "cvx_test", "ste_test",
            "killed", "elapsed_seconds", "P_in", "P_rec", "P_last", "gpu_id"]
    csv_out = os.path.join(out_dir, "summary.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV → {csv_out}")

    # Save full JSON
    json_out = os.path.join(out_dir, "summary.json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON → {json_out}")


if __name__ == "__main__":
    main()