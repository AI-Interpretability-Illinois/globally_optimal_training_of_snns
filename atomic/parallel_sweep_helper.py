from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from queue import Empty
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch

from .simple_testing import _cvx_last_step_acc, _load_dataset_from_args, _set_seed, _ste_last_step_acc
from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


@dataclass
class SweepTask:
    task_type: Literal["ste", "cvx"]
    beta: float
    lr: float
    bias: float


@dataclass
class SweepResult:
    task_type: Literal["ste", "cvx"]
    beta: float
    lr: float
    bias: float
    score: float
    val_loss: float
    test_loss: float
    test_last_step_acc: float
    extra: Dict[str, Any]


_WORKER_DATA: Dict[str, Any] | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_DEVICE: torch.device | None = None


def _parse_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _allocated_cpu_cores() -> int | None:
    # Prefer scheduler allocations when present.
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = _parse_int_env(key)
        if val is not None and val > 0:
            return val
    return None


def _visible_cuda_count() -> int | None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis == "":
        return None
    if vis == "-1":
        return 0
    parts = [p.strip() for p in vis.split(",") if p.strip() != ""]
    if len(parts) == 0:
        return 0
    return len(parts)


def _allocated_gpu_slots() -> int | None:
    # SLURM values can be plain counts or lists like "0,1,2,3".
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        raw = os.environ.get(key, "").strip()
        if raw == "":
            continue
        if raw.isdigit():
            return int(raw)
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if len(parts) > 0 and all(part.isdigit() for part in parts):
            return len(parts)
    return None


def _total_system_memory_gb() -> float | None:
    """Best-effort RAM detection without extra dependencies."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            return float(page_size * phys_pages) / float(1024**3)
    except (ValueError, OSError, AttributeError):
        return None
    return None


def _detected_gpu_slots() -> int:
    alloc_gpu = _allocated_gpu_slots()
    if alloc_gpu is not None:
        return max(0, int(alloc_gpu))
    vis_count = _visible_cuda_count()
    if vis_count is not None:
        return max(0, int(vis_count))
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple MPS behaves like a single logical accelerator slot.
        return 1
    return 0


def _auto_worker_counts(
    *,
    gpu_jobs: int,
    cpu_jobs: int,
    cvx_method: str,
) -> Tuple[int, int, Dict[str, Any]]:
    gpu_slots = _detected_gpu_slots()
    alloc_cores = _allocated_cpu_cores()
    cores = int(alloc_cores if alloc_cores is not None else (os.cpu_count() or 2))
    mem_gb = _total_system_memory_gb()

    # One heavy job per GPU slot by default.
    gpu_workers = min(gpu_slots, gpu_jobs)
    if gpu_workers == 0 and gpu_jobs > 0:
        # CPU fallback for "GPU" queue when no accelerators exist.
        gpu_workers = 1

    if cpu_jobs <= 0:
        cpu_workers = 0
    else:
        reserve_cores = 2
        usable_cores = max(1, cores - reserve_cores)
        per_worker_gb = 8.0 if cvx_method == "cvx" else 4.0
        max_by_mem = usable_cores if mem_gb is None else max(1, int(mem_gb // per_worker_gb))
        cpu_workers = min(cpu_jobs, usable_cores, max_by_mem, 16)
        cpu_workers = max(1, cpu_workers)

    stats = {
        "detected_gpu_slots": gpu_slots,
        "detected_cpu_cores": cores,
        "slurm_allocated_cpu_cores": alloc_cores,
        "slurm_allocated_gpu_slots": _allocated_gpu_slots(),
        "cuda_visible_devices_count": _visible_cuda_count(),
        "detected_mem_gb": mem_gb,
        "mem_budget_per_cpu_worker_gb": 8.0 if cvx_method == "cvx" else 4.0,
    }
    return gpu_workers, cpu_workers, stats


def _resolve_device(worker_kind: str, worker_id: int, num_gpus: int) -> torch.device:
    if worker_kind == "gpu":
        if torch.cuda.is_available() and num_gpus > 0:
            cuda_count = int(torch.cuda.device_count())
            if cuda_count > 0:
                return torch.device(f"cuda:{worker_id % cuda_count}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _worker_init(worker_kind: str, worker_id: int, num_gpus: int, args_dict: Dict[str, Any]) -> None:
    global _WORKER_DATA, _WORKER_ARGS, _WORKER_DEVICE
    _WORKER_ARGS = argparse.Namespace(**args_dict)
    _WORKER_DEVICE = _resolve_device(worker_kind=worker_kind, worker_id=worker_id, num_gpus=num_gpus)
    _set_seed(int(_WORKER_ARGS.seed))
    _WORKER_DATA = _load_dataset_from_args(_WORKER_ARGS)


def _run_one_task(task: SweepTask) -> SweepResult:
    if _WORKER_DATA is None or _WORKER_ARGS is None or _WORKER_DEVICE is None:
        raise RuntimeError("Worker not initialized.")

    x_train = _WORKER_DATA["x_train"]
    y_train = _WORKER_DATA["y_train"]
    x_val = _WORKER_DATA["x_val"]
    y_val = _WORKER_DATA["y_val"]
    x_test = _WORKER_DATA["x_test"]
    y_test = _WORKER_DATA["y_test"]
    num_classes = int(_WORKER_DATA["num_classes"])
    d_in = int(_WORKER_DATA["d_in"])

    # Keep deterministic behavior per-task.
    _set_seed(int(_WORKER_ARGS.seed))

    if task.task_type == "ste":
        ste_out = ste_solve(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            model_cfg=SteModelConfig(
                d_in=d_in,
                num_classes=num_classes,
                L=int(_WORKER_ARGS.L),
                P_rec=int(_WORKER_ARGS.P_rec),
                P_last=int(_WORKER_ARGS.P_last),
                last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
            ),
            solve_cfg=SteSolveConfig(
                loss_name=str(_WORKER_ARGS.loss_type),
                optimizer_name=str(_WORKER_ARGS.optimizer_name),
                lr=float(task.lr),
                epochs=int(_WORKER_ARGS.ste_epochs),
                batch_size=None if int(_WORKER_ARGS.batch_size) == -1 else int(_WORKER_ARGS.batch_size),
                weight_decay=0.0,
                beta_path_reg=float(task.beta),
            ),
            device=_WORKER_DEVICE,
        )
        val_loss = float(ste_out.best_losses["val_loss"])
        score = float(val_loss + task.beta)
        return SweepResult(
            task_type="ste",
            beta=float(task.beta),
            lr=float(task.lr),
            bias=0.0,
            score=score,
            val_loss=val_loss,
            test_loss=float(ste_out.best_losses["test_loss"]),
            test_last_step_acc=float(_ste_last_step_acc(ste_out.model, x_test, y_test)),
            extra={"train_objective": float(ste_out.final_train_objective)},
        )

    init_cfg = InitializationConfig(
        mode="gaussian",
        seed=int(_WORKER_ARGS.seed),
        L=int(_WORKER_ARGS.L),
        P_rec=int(_WORKER_ARGS.P_rec),
        P_last=int(_WORKER_ARGS.P_last),
        feature_count=int(_WORKER_ARGS.P_last),
        last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
        bias=float(task.bias),
    )
    run_device = torch.device("cpu") if str(_WORKER_ARGS.cvx_method) == "cvx" else _WORKER_DEVICE
    cvx_out = cvx_solve(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        init_cfg=init_cfg,
        solve_cfg=SolveConfig(
            method=str(_WORKER_ARGS.cvx_method),
            loss_name=str(_WORKER_ARGS.loss_type),
            beta=float(task.beta),
            lr=float(task.lr),
            optimizer_name=str(_WORKER_ARGS.optimizer_name),
            epochs=int(_WORKER_ARGS.cvx_epochs),
            batch_size=None if int(_WORKER_ARGS.batch_size) == -1 else int(_WORKER_ARGS.batch_size),
        ),
        device=run_device,
    )
    val_loss = float(cvx_out.final_losses["val_loss"])
    score = float(cvx_out.final_losses.get("val_objective", val_loss))
    return SweepResult(
        task_type="cvx",
        beta=float(task.beta),
        lr=float(task.lr),
        bias=float(task.bias),
        score=score,
        val_loss=val_loss,
        test_loss=float(cvx_out.final_losses["test_loss"]),
        test_last_step_acc=float(
            _cvx_last_step_acc(
                cvx_out,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=init_cfg,
            )
        ),
        extra={
            "train_objective": float(cvx_out.final_losses.get("train_objective", float("nan"))),
            "primal_value": float(cvx_out.diagnostics.primal_value),
            "dual_value": float(cvx_out.diagnostics.dual_value),
            "gap": float(cvx_out.diagnostics.gap),
        },
    )


def _gpu_worker_loop(
    worker_id: int,
    num_gpus: int,
    args_dict: Dict[str, Any],
    queue_gpu: mp.Queue,
    queue_result: mp.Queue,
) -> None:
    _worker_init(worker_kind="gpu", worker_id=worker_id, num_gpus=num_gpus, args_dict=args_dict)
    while True:
        try:
            task = queue_gpu.get_nowait()
        except Empty:
            break
        if task is None:
            break
        result = _run_one_task(task)
        queue_result.put(asdict(result))


def _cpu_worker_loop(
    worker_id: int,
    args_dict: Dict[str, Any],
    queue_cpu: mp.Queue,
    queue_result: mp.Queue,
) -> None:
    _worker_init(worker_kind="cpu", worker_id=worker_id, num_gpus=0, args_dict=args_dict)
    while True:
        try:
            task = queue_cpu.get_nowait()
        except Empty:
            break
        if task is None:
            break
        result = _run_one_task(task)
        queue_result.put(asdict(result))


def _build_tasks(args: argparse.Namespace) -> Tuple[List[SweepTask], List[SweepTask]]:
    ste_tasks: List[SweepTask] = []
    cvx_tasks: List[SweepTask] = []

    for lr in args.lr_grid:
        for beta in args.beta_grid:
            ste_tasks.append(SweepTask(task_type="ste", beta=float(beta), lr=float(lr), bias=0.0))
            for bias in args.bias_grid:
                cvx_tasks.append(SweepTask(task_type="cvx", beta=float(beta), lr=float(lr), bias=float(bias)))

    # Heuristic ordering: higher beta and higher lr often costlier/unstable first for quicker pruning visibility.
    ste_tasks.sort(key=lambda t: (t.lr, t.beta), reverse=True)
    cvx_tasks.sort(key=lambda t: (t.lr, t.beta, t.bias), reverse=True)
    return ste_tasks, cvx_tasks


def _collect_results(queue_result: mp.Queue, expected: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _ in range(expected):
        out.append(queue_result.get())
    return out


def _best_of(results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    subset = [r for r in results if r["task_type"] == task_type]
    if len(subset) == 0:
        raise RuntimeError(f"No results for task_type={task_type}.")
    return min(subset, key=lambda r: float(r["score"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue each beta-lr(-bias) combo as an independent run and schedule over GPUs/CPUs."
    )
    parser.add_argument("--dataset", choices=("mnist_seq", "mnist_perm_seq", "cifar_seq", "arithmetic_seq", "dfa", "uci"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--n_train", type=int, default=6000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--P_rec", type=int, default=2048)
    parser.add_argument("--P_last", type=int, default=4096)
    parser.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge")
    parser.add_argument("--optimizer_name", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--cvx_method", choices=("cvx", "sgd"), default="sgd")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--cvx_epochs", type=int, default=150)
    parser.add_argument("--ste_epochs", type=int, default=100)
    parser.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="membrane")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="GPU workers. -1 => auto from detected accelerator slots.",
    )
    parser.add_argument(
        "--num_cpu_workers",
        type=int,
        default=-1,
        help="CPU workers for cvx_method=cvx tasks. -1 => auto from cores/RAM.",
    )
    parser.add_argument("--beta_grid", type=float, nargs="+", default=list(BETA_GRID_DEFAULT))
    parser.add_argument("--lr_grid", type=float, nargs="+", default=list(LR_GRID_DEFAULT))
    parser.add_argument("--bias_grid", type=float, nargs="+", default=list(BIAS_GRID_DEFAULT))
    parser.add_argument("--output_json", type=str, default="")

    # Dataset-specific args mirrored from simple_testing.
    parser.add_argument("--arith_op", choices=("add", "sub", "mul", "div"), default="add")
    parser.add_argument("--arith_base", type=int, default=2)
    parser.add_argument("--n_digits", type=int, default=5)
    parser.add_argument("--dfa_spec", type=str, default="first_last_xor")
    parser.add_argument("--uci_name", type=str, default="pima")
    parser.add_argument("--uci_test_size", type=float, default=0.2)
    parser.add_argument("--uci_val_size", type=float, default=0.2)
    parser.add_argument("--uci_no_standardize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    ste_tasks, cvx_tasks = _build_tasks(args)
    gpu_tasks: List[SweepTask] = []
    cpu_tasks: List[SweepTask] = []
    gpu_tasks.extend(ste_tasks)
    if args.cvx_method == "sgd":
        gpu_tasks.extend(cvx_tasks)
    else:
        cpu_tasks.extend(cvx_tasks)

    queue_gpu: mp.Queue = mp.Queue()
    queue_cpu: mp.Queue = mp.Queue()
    queue_result: mp.Queue = mp.Queue()

    for t in gpu_tasks:
        queue_gpu.put(t)
    for t in cpu_tasks:
        queue_cpu.put(t)

    args_dict = vars(args)
    workers: List[mp.Process] = []

    if int(args.num_gpus) < 0 or int(args.num_cpu_workers) < 0:
        auto_gpu_workers, auto_cpu_workers, hw_stats = _auto_worker_counts(
            gpu_jobs=len(gpu_tasks),
            cpu_jobs=len(cpu_tasks),
            cvx_method=str(args.cvx_method),
        )
    else:
        auto_gpu_workers, auto_cpu_workers, hw_stats = 0, 0, {
            "detected_gpu_slots": _detected_gpu_slots(),
            "detected_cpu_cores": int(os.cpu_count() or 2),
            "detected_mem_gb": _total_system_memory_gb(),
            "mem_budget_per_cpu_worker_gb": 8.0 if str(args.cvx_method) == "cvx" else 4.0,
        }

    gpu_workers = auto_gpu_workers if int(args.num_gpus) < 0 else max(0, int(args.num_gpus))
    # Safety clamp if user asks for too many GPU workers.
    gpu_workers = min(gpu_workers, max(1, len(gpu_tasks))) if len(gpu_tasks) > 0 else 0
    for wid in range(gpu_workers):
        p = mp.Process(target=_gpu_worker_loop, args=(wid, gpu_workers, args_dict, queue_gpu, queue_result))
        p.start()
        workers.append(p)

    cpu_workers = auto_cpu_workers if int(args.num_cpu_workers) < 0 else max(0, int(args.num_cpu_workers))
    cpu_workers = min(cpu_workers, len(cpu_tasks)) if len(cpu_tasks) > 0 else 0
    for wid in range(cpu_workers):
        p = mp.Process(target=_cpu_worker_loop, args=(wid, args_dict, queue_cpu, queue_result))
        p.start()
        workers.append(p)

    expected = len(gpu_tasks) + len(cpu_tasks)
    if expected == 0:
        raise RuntimeError("No tasks were created for sweep.")

    results = _collect_results(queue_result=queue_result, expected=expected)
    for p in workers:
        p.join()

    best_ste = _best_of(results, task_type="ste")
    best_cvx = _best_of(results, task_type="cvx")
    out = {
        "algorithm": {
            "queues": {
                "gpu_queue_jobs": len(gpu_tasks),
                "cpu_queue_jobs": len(cpu_tasks),
            },
            "worker_selection": {
                "num_gpus_arg": int(args.num_gpus),
                "num_cpu_workers_arg": int(args.num_cpu_workers),
                "gpu_workers_used": int(gpu_workers),
                "cpu_workers_used": int(cpu_workers),
                "auto_tuned": bool(int(args.num_gpus) < 0 or int(args.num_cpu_workers) < 0),
                "hardware_stats": hw_stats,
            },
            "routing": {
                "ste": "gpu_queue",
                "cvx_sgd": "gpu_queue",
                "cvx_cvx": "cpu_queue",
            },
            "selection": "min(score) where score=val_loss+beta for STE, score=val_objective (or val_loss) for CVX",
        },
        "sweep": {
            "beta_grid": [float(x) for x in args.beta_grid],
            "lr_grid": [float(x) for x in args.lr_grid],
            "bias_grid": [float(x) for x in args.bias_grid],
            "cvx_method": args.cvx_method,
            "num_gpus": int(args.num_gpus),
            "num_cpu_workers": int(args.num_cpu_workers),
        },
        "best": {"ste": best_ste, "cvx": best_cvx},
        "all_results": results,
    }
    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
