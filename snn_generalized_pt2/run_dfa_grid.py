#!/usr/bin/env python3
"""
DFA Test Bench grid runner — 8×A100 parallel.

Sweeps: DFA_TASKS × T_LIST × INIT_METHODS × BETA_DISTS
Each config runs both CVX and STE on snn_p2.py with --task dfa:<name>.

Usage:
  python run_dfa_grid.py                      # run full grid
  python run_dfa_grid.py --dry-run             # preview configs
  python run_dfa_grid.py --collect-only         # aggregate results
  python run_dfa_grid.py --num-gpus 4          # fewer GPUs
  python run_dfa_grid.py --tasks tomita_3 parity_5  # subset of tasks
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

# ── Grid axes ──────────────────────────────────────────────

# DFA tasks (from dfa_tasks.py built-in library)
DFA_TASKS_DEFAULT = [
    # Tomita grammars (classic)
    "tomita_1", "tomita_2", "tomita_3", "tomita_4",
    "tomita_5", "tomita_6", "tomita_7",
    # Parity / modular counting
    "parity_2", "parity_3", "parity_5",
    # XOR
    "first_last_xor",
    # Substring containment (SL class)
    "contains_00", "contains_010",
    "forbids_000",
    # Bracket matching
    "dyck1_d3",
    # Larger alphabet
    "mod3_sigma4", "mod5_sigma4",
]

T_LIST = [6, 12, 20, 30, 50]
L_LIST = [2, 3, 5]  # lighter L sweep for DFA bench
INIT_METHODS = ["random", "lsuv"]  # focused: most & least structured
BETA_DISTS = ["fixed", "het_loguniform", "het_bimodal"]

# Fixed widths (DFA tasks have small d_in, don't need massive width scaling)
P_IN = 2000
P_REC = 2000
P_LAST = 5000

SEEDS = [0]
EPOCHS = 200
LR_GRID = [1e-3]

# ── Loss type per task ────────────────────────────────────
# All DFA tasks are binary accept/reject → "hinge".
# Override here for any non-standard tasks.
TASK_LOSS = {
    # Default for all DFA builtins is "hinge" (binary classification).
    # Add overrides if you ever wire in regression or multiclass DFAs:
    # "some_regression_dfa": "squared",
    # "some_multiclass_dfa": "ce",
}
DEFAULT_LOSS = "hinge"


def loss_for_task(dfa_name: str) -> str:
    """Return the correct --loss flag for a given DFA task."""
    return TASK_LOSS.get(dfa_name, DEFAULT_LOSS)


def build_configs(tasks: list[str]) -> list[dict]:
    configs = []
    idx = 0
    for task, T, L, init_m, beta_d in product(tasks, T_LIST, L_LIST, INIT_METHODS, BETA_DISTS):
        configs.append({
            "id": idx,
            "task": f"dfa:{task}",
            "dfa_name": task,
            "T": T,
            "L": L,
            "P_in": P_IN,
            "P_rec": P_REC,
            "P_last": P_LAST,
            "init_method": init_m,
            "beta_dist": beta_d,
            "loss": loss_for_task(task),
        })
        idx += 1
    return configs


def run_one_config(cfg: dict, gpu_id: int, out_dir: str, script_dir: str) -> dict:
    """Run a single DFA config on a pinned GPU."""
    task = cfg["task"]
    dfa_name = cfg["dfa_name"]
    T = cfg["T"]
    L = cfg["L"]
    init_method = cfg["init_method"]
    beta_dist = cfg["beta_dist"]
    config_id = cfg["id"]

    result_name = f"{dfa_name}_T{T}_L{L}_{init_method}_{beta_dist}.json"
    result_path = os.path.join(out_dir, "results", result_name)

    # Skip if done
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                existing = json.load(f)
            if existing.get("cvx_test") is not None:
                return existing
        except (json.JSONDecodeError, KeyError):
            pass

    snn_py = os.path.join(script_dir, "snn_p2.py")
    metrics_path = os.path.join(out_dir, "results", f"metrics_{result_name}")

    cmd = [
        sys.executable, snn_py,
        "--task", task,
        "--T", str(T),
        "--L", str(L),
        "--P_in", str(cfg["P_in"]),
        "--P_rec", str(cfg["P_rec"]),
        "--P_last", str(cfg["P_last"]),
        "--loss", cfg["loss"],
        "--init_method", init_method,
        "--beta_dist", beta_dist,
        "--last_layer_readout", "membrane",
        "--seeds", *[str(s) for s in SEEDS],
        "--epochs", str(EPOCHS),
        "--lr_grid", *[str(v) for v in LR_GRID],
        "--log_train",
        "--verbose_patterns",
        "--save_metrics_path", metrics_path,
        "--device", "cuda",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=script_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env,
    )
    elapsed = time.time() - t0

    lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
    killed = lines[-1].strip() == "kill" if lines else False

    cvx_test = None
    ste_test = None
    # snn_p2.py prints test_acc= for hinge/ce, test_negMSE= for squared
    score_keys = ["test_acc=", "test_negMSE="]
    for line in reversed(lines):
        if "CVX" in line and cvx_test is None:
            for sk in score_keys:
                if sk in line:
                    try: cvx_test = float(line.split(sk)[1].split()[0])
                    except: pass
                    break
        if "STE" in line and ste_test is None:
            for sk in score_keys:
                if sk in line:
                    try: ste_test = float(line.split(sk)[1].split()[0].rstrip(")"))
                    except: pass
                    break
        if cvx_test is not None and ste_test is not None:
            break

    result = {
        "config_id": config_id,
        "dfa_name": dfa_name,
        "task": task,
        "T": T,
        "L": L,
        "loss": cfg["loss"],
        "P_in": cfg["P_in"],
        "P_rec": cfg["P_rec"],
        "P_last": cfg["P_last"],
        "init_method": init_method,
        "beta_dist": beta_dist,
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

    log_path = os.path.join(out_dir, "logs", f"{config_id:04d}_{dfa_name}_T{T}_L{L}_{init_method}_{beta_dist}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(proc.stdout or "")

    return result


def print_summary(results: list[dict], tasks: list[str]):
    import numpy as np

    valid = [r for r in results if r.get("cvx_test") is not None]
    failed = [r for r in results if r.get("cvx_test") is None]

    print(f"\n{'='*95}")
    print(f"COMPLETED: {len(valid)}  |  FAILED: {len(failed)}  |  TOTAL: {len(results)}")
    print(f"{'='*95}")

    if not valid:
        return

    # ── Best config per (dfa, T) ──
    by_dt = defaultdict(list)
    for r in valid:
        by_dt[(r["dfa_name"], r["T"])].append(r)

    print(f"\nBEST config per (DFA, T) by cvx_test:")
    print(f"{'DFA':<16} {'T':>4} {'L':>3} {'init':>10} {'beta':>16} {'cvx':>8} {'ste':>8} {'Δ':>7}")
    print("-" * 80)

    for (dfa, T) in sorted(by_dt.keys()):
        best = max(by_dt[(dfa, T)], key=lambda r: r["cvx_test"])
        ste = best.get("ste_test")
        ste_str = f"{ste:.4f}" if ste else "N/A"
        delta = best["cvx_test"] - ste if ste else 0
        marker = " ✓" if best["cvx_test"] > 0.55 else " ✗"
        print(f"{dfa:<16} {T:>4} {best['L']:>3} {best['init_method']:>10} "
              f"{best['beta_dist']:>16} {best['cvx_test']:>8.4f} {ste_str:>8} "
              f"{delta:>+7.4f}{marker}")

    # ── Heatmap: DFA × T (best cvx_test over init/beta) ──
    print(f"\n{'='*95}")
    print("CVX test_acc heatmap: DFA × T (best over L/init/beta)")
    T_vals = sorted(set(r["T"] for r in valid))
    dfa_names = sorted(set(r["dfa_name"] for r in valid))

    print(f"{'DFA':<16}", end="")
    for T in T_vals:
        print(f"  T={T:>3}", end="")
    print()
    print("-" * (16 + 7 * len(T_vals)))

    for dfa in dfa_names:
        print(f"{dfa:<16}", end="")
        for T in T_vals:
            vals = [r["cvx_test"] for r in valid if r["dfa_name"] == dfa and r["T"] == T]
            if vals:
                best = max(vals)
                marker = "✗" if best < 0.55 else " "
                print(f"  {best:>.3f}{marker}", end="")
            else:
                print(f"     — ", end="")
        print()

    # ── Aggregate: which beta_dist helps most? ──
    print(f"\nMean cvx_test by beta_dist (across all DFAs with cvx > 55%):")
    above = [r for r in valid if r["cvx_test"] > 0.55]
    by_beta = defaultdict(list)
    for r in above:
        by_beta[r["beta_dist"]].append(r["cvx_test"])
    for bd in sorted(by_beta.keys()):
        arr = np.array(by_beta[bd])
        print(f"  {bd:>16}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")

    # ── CVX vs STE win rate ──
    cvx_wins = sum(1 for r in valid if r.get("ste_test") and r["cvx_test"] > r["ste_test"])
    ste_wins = sum(1 for r in valid if r.get("ste_test") and r["ste_test"] > r["cvx_test"])
    ties = sum(1 for r in valid if r.get("ste_test") and abs(r["cvx_test"] - r["ste_test"]) < 0.005)
    total_comp = sum(1 for r in valid if r.get("ste_test"))
    print(f"\nCVX vs STE: CVX wins {cvx_wins}/{total_comp}, "
          f"STE wins {ste_wins}/{total_comp}, ties(±0.5%)={ties}")


def main():
    parser = argparse.ArgumentParser(
        description="DFA Test Bench: 8×A100 parallel grid over DFA tasks × T × init × beta"
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="results_dfa_grid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of DFA tasks (default: all built-in)")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else DFA_TASKS_DEFAULT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    configs = build_configs(tasks)
    # Show loss mapping
    losses_used = sorted(set(c["loss"] for c in configs))
    print(f"Grid: {len(tasks)} DFAs × {len(T_LIST)} T × {len(L_LIST)} L × "
          f"{len(INIT_METHODS)} init × {len(BETA_DISTS)} beta = {len(configs)} configs")
    print(f"DFAs: {tasks}")
    print(f"Losses: {losses_used} (per-task assignment via TASK_LOSS, default={DEFAULT_LOSS})")
    print(f"GPUs: {args.num_gpus}  |  Output: {out_dir}")

    csv_path = os.path.join(out_dir, "configs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=configs[0].keys())
        writer.writeheader()
        writer.writerows(configs)

    if args.dry_run:
        print(f"\n[dry-run] First 12 configs:")
        for c in configs[:12]:
            gpu = c["id"] % args.num_gpus
            print(f"  GPU{gpu} | {c['dfa_name']:<16} T={c['T']:>3} L={c['L']} "
                  f"loss={c['loss']:<7} init={c['init_method']:<10} beta={c['beta_dist']}")
        print(f"  ... ({len(configs)} total)")
        return

    if args.collect_only:
        results = []
        results_dir = os.path.join(out_dir, "results")
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".json") and not fname.startswith("metrics_"):
                with open(os.path.join(results_dir, fname)) as f:
                    results.append(json.load(f))
        print_summary(results, tasks)
        if results:
            keys = ["dfa_name", "T", "L", "loss", "init_method", "beta_dist",
                    "cvx_test", "ste_test", "killed", "elapsed_seconds"]
            csv_out = os.path.join(out_dir, "summary.csv")
            with open(csv_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            print(f"\nCSV → {csv_out}")
        return

    # ── Run ──
    num_gpus = args.num_gpus
    all_results = []
    done = 0
    t_start = time.time()

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
                eta = (len(configs) - done) / rate if rate > 0 else 0
                print(f"[{done:>4}/{len(configs)}] "
                      f"{cfg['dfa_name']:<16} T={cfg['T']:>3} L={cfg['L']} "
                      f"init={cfg['init_method']:<10} beta={cfg['beta_dist']:<16} "
                      f"cvx={cvx_str}  GPU{result.get('gpu_id','')} "
                      f"{result.get('elapsed_seconds',0):.0f}s  "
                      f"(ETA {eta:.1f}hr)", flush=True)
            except Exception as e:
                all_results.append({**cfg, "cvx_test": None, "ste_test": None, "error": str(e)})
                print(f"[{done:>4}/{len(configs)}] FAILED {cfg['dfa_name']} T={cfg['T']}: {e}",
                      flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal wall time: {total_time/3600:.2f} hours")

    print_summary(all_results, tasks)

    keys = ["dfa_name", "T", "L", "loss", "init_method", "beta_dist",
            "cvx_test", "ste_test", "killed", "elapsed_seconds",
            "P_in", "P_rec", "P_last", "gpu_id"]
    csv_out = os.path.join(out_dir, "summary.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV → {csv_out}")

    json_out = os.path.join(out_dir, "summary.json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON → {json_out}")


if __name__ == "__main__":
    main()