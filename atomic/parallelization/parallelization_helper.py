from __future__ import annotations

import argparse
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ATOMIC_DIR = SCRIPT_DIR.parent


def discover_run_scripts() -> list[Path]:
    return sorted(ATOMIC_DIR.glob("run_*.py"))


def _sides_for_mode(mode: str) -> list[str]:
    if mode == "both":
        return ["cvx_only", "ste_only"]
    return [mode]


def _is_cvx_side(side: str) -> bool:
    return side == "cvx_only"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch run_* sweep scripts in parallel (local or Slurm).")
    parser.add_argument("--mode", choices=("cvx_only", "ste_only", "both"), default="both")
    parser.add_argument(
        "--execution",
        choices=("slurm", "local"),
        default="slurm",
        help="slurm submits sbatch jobs; local runs with subprocess pool.",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=[],
        help="Optional script basenames (e.g. run_mnist_t_sweep_simple_and_ste_parallel.py). Default: all run_*.py.",
    )
    parser.add_argument("--max_local_workers", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")

    # CPU/CVX job placement.
    parser.add_argument("--cpu_partition", type=str, default="cpu-preempt")
    parser.add_argument("--cpu_time", type=str, default="2-00:00:00")
    parser.add_argument("--cpu_nodelist", type=str, default="cn137")
    parser.add_argument("--cpu_cpus_per_task", type=int, default=16)
    parser.add_argument("--cpu_mem", type=str, default="64G")

    # GPU/SNN job placement.
    parser.add_argument("--gpu_partition", type=str, default="gpuA100x4-interactive")
    parser.add_argument("--gpu_time", type=str, default="1:00:00")
    parser.add_argument("--gpu_nodelist", type=str, default="gpua001")
    parser.add_argument("--gpu_gres", type=str, default="gpu:1")
    parser.add_argument("--gpu_cpus_per_task", type=int, default=8)
    parser.add_argument("--gpu_mem", type=str, default="48G")
    return parser


def resolve_scripts(selected: Iterable[str]) -> list[Path]:
    all_scripts = discover_run_scripts()
    if not selected:
        return all_scripts
    by_name = {p.name: p for p in all_scripts}
    resolved: list[Path] = []
    for name in selected:
        if name not in by_name:
            raise ValueError(f"Unknown script '{name}'. Available: {sorted(by_name.keys())}")
        resolved.append(by_name[name])
    return resolved


def _job_name(script: Path, side: str) -> str:
    base = script.stem.replace("run_", "").replace("_simple_and_ste_parallel", "")
    suffix = "cvx" if side == "cvx_only" else "ste"
    return f"{base}_{suffix}"


def _build_python_cmd(script: Path, side: str) -> list[str]:
    return ["python3", script.name, "--side", side]


def _build_sbatch_cmd(args: argparse.Namespace, script: Path, side: str) -> list[str]:
    is_cvx = _is_cvx_side(side)
    partition = args.cpu_partition if is_cvx else args.gpu_partition
    time_lim = args.cpu_time if is_cvx else args.gpu_time
    nodelist = args.cpu_nodelist if is_cvx else args.gpu_nodelist
    cpus = args.cpu_cpus_per_task if is_cvx else args.gpu_cpus_per_task
    mem = args.cpu_mem if is_cvx else args.gpu_mem
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--time",
        time_lim,
        "--cpus-per-task",
        str(cpus),
        "--mem",
        mem,
        "--job-name",
        _job_name(script, side),
    ]
    if nodelist:
        sbatch.extend(["--nodelist", nodelist])
    if not is_cvx:
        sbatch.extend(["--gres", args.gpu_gres])

    py_cmd = _build_python_cmd(script, side)
    wrap = f"cd {shlex.quote(str(ATOMIC_DIR))} && {' '.join(shlex.quote(tok) for tok in py_cmd)}"
    sbatch.extend(["--wrap", wrap])
    return sbatch


def submit_slurm(args: argparse.Namespace, jobs: list[tuple[Path, str]]) -> None:
    print(f"Submitting {len(jobs)} jobs to Slurm...")
    for script, side in jobs:
        cmd = _build_sbatch_cmd(args, script, side)
        print(" ".join(shlex.quote(t) for t in cmd))
        if args.dry_run:
            continue
        out = subprocess.check_output(cmd, text=True).strip()
        print(f"Submitted {script.name} [{side}] -> job_id={out}")


def run_local(args: argparse.Namespace, jobs: list[tuple[Path, str]]) -> None:
    print(f"Running {len(jobs)} jobs locally with max_workers={args.max_local_workers}...")

    def _run_one(item: tuple[Path, str]) -> tuple[str, int]:
        script, side = item
        cmd = _build_python_cmd(script, side)
        if args.dry_run:
            print(" ".join(shlex.quote(t) for t in cmd))
            return f"{script.name}:{side}", 0
        proc = subprocess.run(cmd, cwd=ATOMIC_DIR, check=False)
        return f"{script.name}:{side}", int(proc.returncode)

    with ThreadPoolExecutor(max_workers=max(1, args.max_local_workers)) as pool:
        futures = [pool.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futures):
            label, code = fut.result()
            print(f"{label} -> exit={code}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scripts = resolve_scripts(args.scripts)

    jobs: list[tuple[Path, str]] = []
    for script in scripts:
        for side in _sides_for_mode(args.mode):
            jobs.append((script, side))

    if args.execution == "slurm":
        submit_slurm(args, jobs)
    else:
        run_local(args, jobs)


if __name__ == "__main__":
    main()
