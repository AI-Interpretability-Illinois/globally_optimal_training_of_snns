from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from fine_tune import FineTuneConfig, run_fine_tune_pipeline
    from layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from simple_testing import _cvx_last_step_acc, _load_dataset_from_args, _set_seed, _ste_last_step_acc
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers import ste_parallel_Solve as ste_par
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .fine_tune import FineTuneConfig, run_fine_tune_pipeline
    from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from .simple_testing import _cvx_last_step_acc, _load_dataset_from_args, _set_seed, _ste_last_step_acc
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers import ste_parallel_Solve as ste_par
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


@dataclass
class SweepTask:
    task_type: Literal["ste", "cvx", "fine_tune", "layer_wise"]
    pipeline_mode: Literal["simple", "fine_tune", "layer_wise"]
    run_id: str
    pair_id: str
    seed: int
    beta: float
    lr: float
    bias: float


@dataclass
class SweepResult:
    task_type: Literal["ste", "cvx", "fine_tune", "layer_wise"]
    pipeline_mode: Literal["simple", "fine_tune", "layer_wise"]
    run_id: str
    pair_id: str
    seed: int
    beta: float
    lr: float
    bias: float
    score: float
    train_loss: float
    val_loss: float
    test_loss: float
    train_last_step_acc: float
    val_last_step_acc: float
    test_last_step_acc: float
    extra: Dict[str, Any]


_WORKER_DATA: Dict[str, Any] | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_DEVICE: torch.device | None = None


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v))


def _arithmetic_acc_suffix(extra: Dict[str, Any]) -> str:
    token = extra.get("test_token_acc")
    seq = extra.get("test_seq_acc")
    if _is_finite_number(token) and _is_finite_number(seq):
        return f" test_token_acc={float(token):.6f} test_seq_acc={float(seq):.6f}"
    return ""


def _sweep_context_suffix() -> str:
    """Best-effort context from the worker argparse namespace (set in _worker_init)."""
    if _WORKER_ARGS is None:
        return ""
    ds = getattr(_WORKER_ARGS, "dataset", "")
    kpar = int(getattr(_WORKER_ARGS, "K_parallel", 1))
    t = int(getattr(_WORKER_ARGS, "T", 0))
    l = int(getattr(_WORKER_ARGS, "L", 0))
    p_rec = int(getattr(_WORKER_ARGS, "P_rec", 0))
    p_last = int(getattr(_WORKER_ARGS, "P_last", 0))
    cvx_m = getattr(_WORKER_ARGS, "cvx_method", "")
    simple_side = getattr(_WORKER_ARGS, "simple_side", "")
    return (
        f" dataset={ds} K_parallel={kpar} T={t} L={l} P_rec={p_rec} P_last={p_last}"
        f" cvx_method={cvx_m} simple_side={simple_side}"
    )


def _fmt_metric(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.6f}"


def _log_run_start(task: SweepTask) -> None:
    print(
        "[run-start] "
        f"task_type={task.task_type} pipeline_mode={task.pipeline_mode} "
        f"beta={float(task.beta):.8g} lr={float(task.lr):.8g} bias={float(task.bias):.8g} "
        f"seed={int(task.seed)} run_id={task.run_id} pair_id={task.pair_id}"
        f"{_sweep_context_suffix()}",
        flush=True,
    )


def _log_run_finish(task: SweepTask, result: SweepResult) -> None:
    extra = result.extra if isinstance(result.extra, dict) else {}
    arithmetic_suffix = _arithmetic_acc_suffix(extra)
    arith_val = ""
    if _is_finite_number(extra.get("val_token_acc")) and _is_finite_number(extra.get("val_seq_acc")):
        arith_val = (
            f" val_token_acc={_fmt_metric(extra.get('val_token_acc'))}"
            f" val_seq_acc={_fmt_metric(extra.get('val_seq_acc'))}"
        )
    if arithmetic_suffix:
        arith_val += " " + arithmetic_suffix.strip()
    print(
        "[run-finish] "
        f"task_type={result.task_type} "
        f"beta={float(task.beta):.8g} lr={float(task.lr):.8g} bias={float(task.bias):.8g} "
        f"seed={int(task.seed)} run_id={task.run_id} pair_id={task.pair_id} "
        f"score={_fmt_metric(result.score)} "
        f"train_loss={_fmt_metric(result.train_loss)} val_loss={_fmt_metric(result.val_loss)} "
        f"test_loss={_fmt_metric(result.test_loss)} "
        f"train_acc={_fmt_metric(result.train_last_step_acc)} val_acc={_fmt_metric(result.val_last_step_acc)} "
        f"test_acc={_fmt_metric(result.test_last_step_acc)}"
        f"{arith_val}",
        flush=True,
    )


@contextlib.contextmanager
def _mute_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


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


def _cgroup_memory_limit_gb() -> float | None:
    """Return cgroup RAM cap for this process when finite (Linux jobs/containers)."""
    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.is_file():
        raw = v2.read_text().strip()
        if raw == "max":
            return None
        try:
            limit_b = int(raw)
        except ValueError:
            return None
        if limit_b <= 0 or limit_b >= (1 << 60):
            return None
        return float(limit_b) / float(1024**3)

    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.is_file():
        try:
            limit_b = int(v1.read_text().strip())
        except ValueError:
            return None
        if limit_b <= 0 or limit_b >= (1 << 60):
            return None
        return float(limit_b) / float(1024**3)
    return None


def _slurm_job_memory_gb() -> float | None:
    """Slurm sets total job RAM on the allocation (megabytes)."""
    raw = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if raw.isdigit():
        return float(raw) / 1024.0
    return None


def _effective_job_memory_budget_gb() -> float | None:
    """Prefer cgroup/Slurm limits over host total RAM (avoids overspawn on batch nodes)."""
    cg = _cgroup_memory_limit_gb()
    if cg is not None:
        return cg
    sl = _slurm_job_memory_gb()
    if sl is not None:
        return sl
    return _total_system_memory_gb()


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
    gpu_mem_budget_gb_per_worker: float,
) -> Tuple[int, int, Dict[str, Any]]:
    gpu_slots = _detected_gpu_slots()
    alloc_cores = _allocated_cpu_cores()
    cores = int(alloc_cores if alloc_cores is not None else (os.cpu_count() or 2))
    host_mem_gb = _total_system_memory_gb()
    mem_gb = _effective_job_memory_budget_gb()

    # One heavy job per GPU slot by default; cap by RAM (each worker loads torch + dataset in spawn).
    gpu_workers = min(gpu_slots, gpu_jobs)
    if gpu_workers == 0 and gpu_jobs > 0:
        # CPU fallback for "GPU" queue when no accelerators exist.
        gpu_workers = 1
    if mem_gb is not None and gpu_jobs > 0 and float(gpu_mem_budget_gb_per_worker) > 0:
        max_gpu_by_mem = max(1, int(float(mem_gb) // float(gpu_mem_budget_gb_per_worker)))
        gpu_workers = min(gpu_workers, max_gpu_by_mem)

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
        "host_total_mem_gb": host_mem_gb,
        "cgroup_mem_limit_gb": _cgroup_memory_limit_gb(),
        "slurm_mem_per_node_gb": _slurm_job_memory_gb(),
        "effective_mem_budget_gb": mem_gb,
        "mem_budget_per_gpu_worker_gb": float(gpu_mem_budget_gb_per_worker),
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
    _set_seed(int(task.seed))
    suppress_task_logs = not bool(_WORKER_ARGS.verbose_runs)
    _log_run_start(task)

    if task.task_type == "fine_tune":
        cfg = FineTuneConfig(
            T=int(_WORKER_ARGS.T),
            P_in=d_in,
            P_rec=int(_WORKER_ARGS.P_rec),
            P_last=int(_WORKER_ARGS.P_last),
            L=int(_WORKER_ARGS.L),
            loss_type=str(_WORKER_ARGS.loss_type),
            epochs=int(_WORKER_ARGS.cvx_epochs),
            batch_size=int(_WORKER_ARGS.batch_size),
            cvx_optimizer=str(_WORKER_ARGS.optimizer_name),
            cvx_method=str(_WORKER_ARGS.cvx_method),
            ste_pretrain_epochs=int(_WORKER_ARGS.ste_pretrain_epochs),
            ste_post_epochs=int(_WORKER_ARGS.ste_post_epochs),
            last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
            readout_modes=(str(_WORKER_ARGS.last_layer_readout),),
            K_parallel=int(_WORKER_ARGS.K_parallel),
            beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
            bias_grid=tuple(float(x) for x in _WORKER_ARGS.bias_grid),
            ste_beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            ste_lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
        )
        with _mute_output(suppress_task_logs):
            out = run_fine_tune_pipeline(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                seed=int(task.seed),
                cfg=cfg,
            )
        selected_readout = str(out["selected_readout"])
        selected = out["by_readout"][selected_readout]
        ste_post = selected["ste_post"]
        cvx_by_source = selected["cvx_by_source"]
        cvx_sel = selected["cvx_selected_params"]
        best_source = None
        best_source_val = float("inf")
        for source_name, cvx_out in cvx_by_source.items():
            val_obj = float(cvx_out.final_losses.get("val_objective", cvx_out.final_losses["val_loss"]))
            if val_obj < best_source_val:
                best_source_val = val_obj
                best_source = str(source_name)
        if best_source is None:
            raise RuntimeError("Fine-tune produced no CVX source candidate.")
        best_cvx = cvx_by_source[best_source]
        cvx_init_mode = "gaussian" if best_source == "gaussian" else "pretraining"
        pretrained_weights = None
        if cvx_init_mode == "pretraining":
            ste_pre = selected["ste_pretrain"]
            if not isinstance(ste_pre.model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
                raise TypeError("Expected SNNBaselineSeq (or parallel variant) in fine_tune ste_pretrain result.")
            pretrained_weights = _extract_weights_from_snn(ste_pre.model)
        test_acc = float(
            _cvx_last_step_acc(
                best_cvx,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=InitializationConfig(
                    mode=cvx_init_mode,
                    seed=int(task.seed),
                    L=int(_WORKER_ARGS.L),
                    P_rec=int(_WORKER_ARGS.P_rec),
                    P_last=int(_WORKER_ARGS.P_last),
                    feature_count=int(_WORKER_ARGS.P_last),
                    last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
                    bias=float(cvx_sel[best_source]["bias"]),
                    pretrained_weights=pretrained_weights,
                    K_parallel=int(_WORKER_ARGS.K_parallel),
                ),
            )
        )
        result = SweepResult(
            task_type="fine_tune",
            pipeline_mode=task.pipeline_mode,
            run_id=task.run_id,
            pair_id=task.pair_id,
            seed=int(task.seed),
            beta=0.0,
            lr=0.0,
            bias=0.0,
            score=float(best_source_val),
            train_loss=float(best_cvx.final_losses.get("train_loss", float("nan"))),
            val_loss=float(best_cvx.final_losses["val_loss"]),
            test_loss=float(best_cvx.final_losses["test_loss"]),
            train_last_step_acc=float(
                _cvx_last_step_acc(
                    best_cvx,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_train,
                    y_test=y_train,
                    init_cfg=InitializationConfig(
                        mode=cvx_init_mode,
                        seed=int(task.seed),
                        L=int(_WORKER_ARGS.L),
                        P_rec=int(_WORKER_ARGS.P_rec),
                        P_last=int(_WORKER_ARGS.P_last),
                        feature_count=int(_WORKER_ARGS.P_last),
                        last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
                        bias=float(cvx_sel[best_source]["bias"]),
                        pretrained_weights=pretrained_weights,
                        K_parallel=int(_WORKER_ARGS.K_parallel),
                    ),
                )
            ),
            val_last_step_acc=float(
                _cvx_last_step_acc(
                    best_cvx,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_val,
                    y_test=y_val,
                    init_cfg=InitializationConfig(
                        mode=cvx_init_mode,
                        seed=int(task.seed),
                        L=int(_WORKER_ARGS.L),
                        P_rec=int(_WORKER_ARGS.P_rec),
                        P_last=int(_WORKER_ARGS.P_last),
                        feature_count=int(_WORKER_ARGS.P_last),
                        last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
                        bias=float(cvx_sel[best_source]["bias"]),
                        pretrained_weights=pretrained_weights,
                        K_parallel=int(_WORKER_ARGS.K_parallel),
                    ),
                )
            ),
            test_last_step_acc=test_acc,
            extra={
                "selected_by": "val_objective",
                "selected_readout": selected_readout,
                "best_cvx_source": best_source,
                "best_cvx_selected_params": cvx_sel[best_source],
                "ste_post_selected_params": selected["ste_post_selected_params"],
                "ste_post_val_loss": float(ste_post.best_losses["val_loss"]),
                "ste_post_test_loss": float(ste_post.best_losses["test_loss"]),
                "cvx_train_token_acc": float(best_cvx.final_losses.get("train_token_acc", float("nan"))),
                "cvx_val_token_acc": float(best_cvx.final_losses.get("val_token_acc", float("nan"))),
                "cvx_test_token_acc": float(best_cvx.final_losses.get("test_token_acc", float("nan"))),
                "cvx_train_seq_acc": float(best_cvx.final_losses.get("train_seq_acc", float("nan"))),
                "cvx_val_seq_acc": float(best_cvx.final_losses.get("val_seq_acc", float("nan"))),
                "cvx_test_seq_acc": float(best_cvx.final_losses.get("test_seq_acc", float("nan"))),
                "cvx_train_token_loss": float(best_cvx.final_losses.get("train_token_loss", float("nan"))),
                "cvx_val_token_loss": float(best_cvx.final_losses.get("val_token_loss", float("nan"))),
                "cvx_test_token_loss": float(best_cvx.final_losses.get("test_token_loss", float("nan"))),
                "cvx_train_seq_loss": float(best_cvx.final_losses.get("train_seq_loss", float("nan"))),
                "cvx_val_seq_loss": float(best_cvx.final_losses.get("val_seq_loss", float("nan"))),
                "cvx_test_seq_loss": float(best_cvx.final_losses.get("test_seq_loss", float("nan"))),
                "ste_post_train_token_acc": float(ste_post.best_losses.get("train_token_acc", float("nan"))),
                "ste_post_val_token_acc": float(ste_post.best_losses.get("val_token_acc", float("nan"))),
                "ste_post_test_token_acc": float(ste_post.best_losses.get("test_token_acc", float("nan"))),
                "ste_post_train_seq_acc": float(ste_post.best_losses.get("train_seq_acc", float("nan"))),
                "ste_post_val_seq_acc": float(ste_post.best_losses.get("val_seq_acc", float("nan"))),
                "ste_post_test_seq_acc": float(ste_post.best_losses.get("test_seq_acc", float("nan"))),
                "ste_post_train_token_loss": float(ste_post.best_losses.get("train_token_loss", float("nan"))),
                "ste_post_val_token_loss": float(ste_post.best_losses.get("val_token_loss", float("nan"))),
                "ste_post_test_token_loss": float(ste_post.best_losses.get("test_token_loss", float("nan"))),
                "ste_post_train_seq_loss": float(ste_post.best_losses.get("train_seq_loss", float("nan"))),
                "ste_post_val_seq_loss": float(ste_post.best_losses.get("val_seq_loss", float("nan"))),
                "ste_post_test_seq_loss": float(ste_post.best_losses.get("test_seq_loss", float("nan"))),
            },
        )
        _log_run_finish(task, result)
        return result

    if task.task_type == "layer_wise":
        lw_cfg = LayerWiseConfig(
            num_blocks=int(_WORKER_ARGS.num_blocks),
            L=int(_WORKER_ARGS.L),
            P_rec=int(_WORKER_ARGS.P_rec),
            P_last=int(_WORKER_ARGS.P_last),
            ste_pretrain_epochs=int(_WORKER_ARGS.ste_pretrain_epochs),
            ste_finetune_epochs=int(_WORKER_ARGS.ste_post_epochs),
            loss_name=str(_WORKER_ARGS.loss_type),
            optimizer_name=str(_WORKER_ARGS.optimizer_name),
            cvx_method=str(_WORKER_ARGS.cvx_method),
            cvx_beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            cvx_lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
            cvx_bias_grid=tuple(float(x) for x in _WORKER_ARGS.bias_grid),
            ste_beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            ste_lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
            K_parallel=int(_WORKER_ARGS.K_parallel),
        )
        with _mute_output(suppress_task_logs):
            rows = run_layer_wise_stacking_test_bench(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                cfg=lw_cfg,
            )
        if len(rows) == 0:
            raise RuntimeError("Layer-wise pipeline returned zero blocks.")
        final_block = rows[-1]
        cvx_out = final_block["cvx"]
        ste_ft = final_block["ste_finetune"]
        ste_pre = final_block["ste_pre"]
        if not isinstance(ste_pre.model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
            raise TypeError("Expected SNNBaselineSeq (or parallel variant) in layer-wise ste_pre result.")
        cvx_init = InitializationConfig(
            mode="pretraining",
            seed=int(task.seed),
            L=int(_WORKER_ARGS.L),
            P_rec=int(_WORKER_ARGS.P_rec),
            P_last=int(_WORKER_ARGS.P_last),
            feature_count=int(_WORKER_ARGS.P_last),
            bias=float(final_block["cvx_selected_params"]["bias"]),
            pretrained_weights=_extract_weights_from_snn(ste_pre.model),
            K_parallel=int(_WORKER_ARGS.K_parallel),
        )
        result = SweepResult(
            task_type="layer_wise",
            pipeline_mode=task.pipeline_mode,
            run_id=task.run_id,
            pair_id=task.pair_id,
            seed=int(task.seed),
            beta=0.0,
            lr=0.0,
            bias=0.0,
            score=float(cvx_out.final_losses.get("val_objective", cvx_out.final_losses["val_loss"])),
            train_loss=float(cvx_out.final_losses.get("train_loss", float("nan"))),
            val_loss=float(cvx_out.final_losses["val_loss"]),
            test_loss=float(cvx_out.final_losses["test_loss"]),
            train_last_step_acc=float(
                _cvx_last_step_acc(
                    cvx_out,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_train,
                    y_test=y_train,
                    init_cfg=cvx_init,
                )
            ),
            val_last_step_acc=float(
                _cvx_last_step_acc(
                    cvx_out,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_val,
                    y_test=y_val,
                    init_cfg=cvx_init,
                )
            ),
            test_last_step_acc=float(
                _cvx_last_step_acc(
                    cvx_out,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_test,
                    y_test=y_test,
                    init_cfg=cvx_init,
                )
            ),
            extra={
                "selected_by": "val_objective",
                "num_blocks": int(_WORKER_ARGS.num_blocks),
                "final_block_idx": int(final_block["block_idx"]),
                "cvx_selected_params": final_block["cvx_selected_params"],
                "ste_finetune_selected_params": final_block["ste_finetune_selected_params"],
                "ste_finetune_val_loss": float(ste_ft.best_losses["val_loss"]),
                "ste_finetune_test_loss": float(ste_ft.best_losses["test_loss"]),
                "cvx_train_token_acc": float(cvx_out.final_losses.get("train_token_acc", float("nan"))),
                "cvx_val_token_acc": float(cvx_out.final_losses.get("val_token_acc", float("nan"))),
                "cvx_test_token_acc": float(cvx_out.final_losses.get("test_token_acc", float("nan"))),
                "cvx_train_seq_acc": float(cvx_out.final_losses.get("train_seq_acc", float("nan"))),
                "cvx_val_seq_acc": float(cvx_out.final_losses.get("val_seq_acc", float("nan"))),
                "cvx_test_seq_acc": float(cvx_out.final_losses.get("test_seq_acc", float("nan"))),
                "cvx_train_token_loss": float(cvx_out.final_losses.get("train_token_loss", float("nan"))),
                "cvx_val_token_loss": float(cvx_out.final_losses.get("val_token_loss", float("nan"))),
                "cvx_test_token_loss": float(cvx_out.final_losses.get("test_token_loss", float("nan"))),
                "cvx_train_seq_loss": float(cvx_out.final_losses.get("train_seq_loss", float("nan"))),
                "cvx_val_seq_loss": float(cvx_out.final_losses.get("val_seq_loss", float("nan"))),
                "cvx_test_seq_loss": float(cvx_out.final_losses.get("test_seq_loss", float("nan"))),
                "ste_finetune_train_token_acc": float(ste_ft.best_losses.get("train_token_acc", float("nan"))),
                "ste_finetune_val_token_acc": float(ste_ft.best_losses.get("val_token_acc", float("nan"))),
                "ste_finetune_test_token_acc": float(ste_ft.best_losses.get("test_token_acc", float("nan"))),
                "ste_finetune_train_seq_acc": float(ste_ft.best_losses.get("train_seq_acc", float("nan"))),
                "ste_finetune_val_seq_acc": float(ste_ft.best_losses.get("val_seq_acc", float("nan"))),
                "ste_finetune_test_seq_acc": float(ste_ft.best_losses.get("test_seq_acc", float("nan"))),
                "ste_finetune_train_token_loss": float(ste_ft.best_losses.get("train_token_loss", float("nan"))),
                "ste_finetune_val_token_loss": float(ste_ft.best_losses.get("val_token_loss", float("nan"))),
                "ste_finetune_test_token_loss": float(ste_ft.best_losses.get("test_token_loss", float("nan"))),
                "ste_finetune_train_seq_loss": float(ste_ft.best_losses.get("train_seq_loss", float("nan"))),
                "ste_finetune_val_seq_loss": float(ste_ft.best_losses.get("val_seq_loss", float("nan"))),
                "ste_finetune_test_seq_loss": float(ste_ft.best_losses.get("test_seq_loss", float("nan"))),
            },
        )
        _log_run_finish(task, result)
        return result

    if task.task_type == "ste":
        with _mute_output(suppress_task_logs):
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
                    K_parallel=int(_WORKER_ARGS.K_parallel),
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
        result = SweepResult(
            task_type="ste",
            pipeline_mode=task.pipeline_mode,
            run_id=task.run_id,
            pair_id=task.pair_id,
            seed=int(task.seed),
            beta=float(task.beta),
            lr=float(task.lr),
            bias=0.0,
            score=score,
            train_loss=float(ste_out.best_losses["train_loss"]),
            val_loss=val_loss,
            test_loss=float(ste_out.best_losses["test_loss"]),
            train_last_step_acc=float(_ste_last_step_acc(ste_out.model, x_train, y_train)),
            val_last_step_acc=float(_ste_last_step_acc(ste_out.model, x_val, y_val)),
            test_last_step_acc=float(_ste_last_step_acc(ste_out.model, x_test, y_test)),
            extra={
                "selected_by": "val_objective",
                "train_objective": float(ste_out.final_train_objective),
                "train_token_acc": float(ste_out.best_losses.get("train_token_acc", float("nan"))),
                "val_token_acc": float(ste_out.best_losses.get("val_token_acc", float("nan"))),
                "test_token_acc": float(ste_out.best_losses.get("test_token_acc", float("nan"))),
                "train_seq_acc": float(ste_out.best_losses.get("train_seq_acc", float("nan"))),
                "val_seq_acc": float(ste_out.best_losses.get("val_seq_acc", float("nan"))),
                "test_seq_acc": float(ste_out.best_losses.get("test_seq_acc", float("nan"))),
                "train_token_loss": float(ste_out.best_losses.get("train_token_loss", float("nan"))),
                "val_token_loss": float(ste_out.best_losses.get("val_token_loss", float("nan"))),
                "test_token_loss": float(ste_out.best_losses.get("test_token_loss", float("nan"))),
                "train_seq_loss": float(ste_out.best_losses.get("train_seq_loss", float("nan"))),
                "val_seq_loss": float(ste_out.best_losses.get("val_seq_loss", float("nan"))),
                "test_seq_loss": float(ste_out.best_losses.get("test_seq_loss", float("nan"))),
            },
        )
        _log_run_finish(task, result)
        return result

    init_cfg = InitializationConfig(
        mode="gaussian",
        seed=int(_WORKER_ARGS.seed),
        L=int(_WORKER_ARGS.L),
        P_rec=int(_WORKER_ARGS.P_rec),
        P_last=int(_WORKER_ARGS.P_last),
        feature_count=int(_WORKER_ARGS.P_last),
        last_layer_readout=str(_WORKER_ARGS.last_layer_readout),
        bias=float(task.bias),
        K_parallel=int(_WORKER_ARGS.K_parallel),
    )
    run_device = torch.device("cpu") if str(_WORKER_ARGS.cvx_method) == "cvx" else _WORKER_DEVICE
    with _mute_output(suppress_task_logs):
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
    result = SweepResult(
        task_type="cvx",
        pipeline_mode=task.pipeline_mode,
        run_id=task.run_id,
        pair_id=task.pair_id,
        seed=int(task.seed),
        beta=float(task.beta),
        lr=float(task.lr),
        bias=float(task.bias),
        score=score,
        train_loss=float(cvx_out.final_losses.get("train_loss", float("nan"))),
        val_loss=val_loss,
        test_loss=float(cvx_out.final_losses["test_loss"]),
        train_last_step_acc=float(
            _cvx_last_step_acc(
                cvx_out,
                x_train=x_train,
                x_val=x_val,
                x_test=x_train,
                y_test=y_train,
                init_cfg=init_cfg,
            )
        ),
        val_last_step_acc=float(
            _cvx_last_step_acc(
                cvx_out,
                x_train=x_train,
                x_val=x_val,
                x_test=x_val,
                y_test=y_val,
                init_cfg=init_cfg,
            )
        ),
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
            "selected_by": "val_objective",
            "train_objective": float(cvx_out.final_losses.get("train_objective", float("nan"))),
            "primal_value": float(cvx_out.diagnostics.primal_value),
            "dual_value": float(cvx_out.diagnostics.dual_value),
            "gap": float(cvx_out.diagnostics.gap),
            "train_token_acc": float(cvx_out.final_losses.get("train_token_acc", float("nan"))),
            "val_token_acc": float(cvx_out.final_losses.get("val_token_acc", float("nan"))),
            "test_token_acc": float(cvx_out.final_losses.get("test_token_acc", float("nan"))),
            "train_seq_acc": float(cvx_out.final_losses.get("train_seq_acc", float("nan"))),
            "val_seq_acc": float(cvx_out.final_losses.get("val_seq_acc", float("nan"))),
            "test_seq_acc": float(cvx_out.final_losses.get("test_seq_acc", float("nan"))),
            "train_token_loss": float(cvx_out.final_losses.get("train_token_loss", float("nan"))),
            "val_token_loss": float(cvx_out.final_losses.get("val_token_loss", float("nan"))),
            "test_token_loss": float(cvx_out.final_losses.get("test_token_loss", float("nan"))),
            "train_seq_loss": float(cvx_out.final_losses.get("train_seq_loss", float("nan"))),
            "val_seq_loss": float(cvx_out.final_losses.get("val_seq_loss", float("nan"))),
            "test_seq_loss": float(cvx_out.final_losses.get("test_seq_loss", float("nan"))),
        },
    )
    _log_run_finish(task, result)
    return result


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
    pipeline_tasks: List[SweepTask] = []

    seeds = [int(s) for s in args.seeds]
    if str(args.pipeline_mode) == "simple":
        simple_side = str(args.simple_side)
        run_ste = simple_side in ("both", "ste_only")
        run_cvx = simple_side in ("both", "cvx_only")
        for seed in seeds:
            for lr in args.lr_grid:
                for beta in args.beta_grid:
                    pair_id = f"{args.dataset}|seed={seed}|beta={float(beta):.8g}|lr={float(lr):.8g}"
                    ste_run_id = f"ste|{pair_id}"
                    if run_ste:
                        ste_tasks.append(
                            SweepTask(
                                task_type="ste",
                                pipeline_mode="simple",
                                run_id=ste_run_id,
                                pair_id=pair_id,
                                seed=seed,
                                beta=float(beta),
                                lr=float(lr),
                                bias=0.0,
                            )
                        )
                    if run_cvx:
                        for bias in args.bias_grid:
                            cvx_tasks.append(
                                SweepTask(
                                    task_type="cvx",
                                    pipeline_mode="simple",
                                    run_id=f"cvx|{pair_id}|bias={float(bias):.8g}",
                                    pair_id=pair_id,
                                    seed=seed,
                                    beta=float(beta),
                                    lr=float(lr),
                                    bias=float(bias),
                                )
                            )
    else:
        task_type = "fine_tune" if str(args.pipeline_mode) == "fine_tune" else "layer_wise"
        for seed in seeds:
            pipeline_tasks.append(
                SweepTask(
                    task_type=task_type,
                    pipeline_mode=str(args.pipeline_mode),
                    run_id=f"{task_type}|{args.dataset}|seed={seed}",
                    pair_id=f"{task_type}|{args.dataset}|seed={seed}",
                    seed=seed,
                    beta=0.0,
                    lr=0.0,
                    bias=0.0,
                )
            )

    # Heuristic ordering: higher beta and higher lr often costlier/unstable first for quicker pruning visibility.
    ste_tasks.sort(key=lambda t: (t.lr, t.beta), reverse=True)
    cvx_tasks.sort(key=lambda t: (t.lr, t.beta, t.bias), reverse=True)
    if len(pipeline_tasks) > 0:
        # Keep deterministic seed order.
        pipeline_tasks.sort(key=lambda t: t.seed)
        if str(args.cvx_method) == "cvx":
            return [], pipeline_tasks
        return pipeline_tasks, []
    return ste_tasks, cvx_tasks


def _collect_results(queue_result: mp.Queue, expected: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _ in range(expected):
        out.append(queue_result.get())
    return out


def _build_task_name(args: argparse.Namespace) -> str:
    if str(args.dataset) == "dfa":
        return f"dfa_{args.dfa_spec}"
    if str(args.dataset) == "arithmetic_seq":
        return f"arithmetic_{args.arith_op}_base{args.arith_base}_digits{args.n_digits}"
    if str(args.dataset) == "uci":
        return f"uci_{args.uci_name}"
    return str(args.dataset)


def _extract_weights_from_snn(model: torch.nn.Module) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    if hasattr(model, "fcs") and hasattr(model, "classifier"):
        for fc in model.fcs:
            weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    if hasattr(model, "branches") and hasattr(model, "classifier"):
        for br in model.branches:
            for fc in br.fcs:
                weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    raise TypeError(f"Unsupported SNN layout for weight export: {type(model)}")


def _default_csv_paths(args: argparse.Namespace) -> Dict[str, Path]:
    task_name = _build_task_name(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ste": out_dir / f"{task_name}_ste_{stamp}.csv",
        "cvx": out_dir / f"{task_name}_cvx_{stamp}.csv",
        "fine_tune": out_dir / f"{task_name}_fine_tune_{stamp}.csv",
        "layer_wise": out_dir / f"{task_name}_layer_wise_{stamp}.csv",
    }


def _flatten_result_row(
    *,
    row: Dict[str, Any],
    args: argparse.Namespace,
    gpu_workers: int,
    cpu_workers: int,
) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "task_name": _build_task_name(args),
        "seed": int(args.seed),
        "T": int(args.T),
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "loss_type": str(args.loss_type),
        "optimizer_name": str(args.optimizer_name),
        "cvx_method": str(args.cvx_method),
        "ste_epochs": int(args.ste_epochs),
        "cvx_epochs": int(args.cvx_epochs),
        "last_layer_readout": str(args.last_layer_readout),
        "gpu_workers_used": int(gpu_workers),
        "cpu_workers_used": int(cpu_workers),
        "pipeline_mode": str(row["pipeline_mode"]),
        "run_id": str(row["run_id"]),
        "pair_id": str(row["pair_id"]),
        "pair_ste_run_id": f"ste|{str(row['pair_id'])}",
        "task_seed": int(row["seed"]),
        "task_type": str(row["task_type"]),
        "beta": float(row["beta"]),
        "lr": float(row["lr"]),
        "bias": float(row["bias"]),
        "score": float(row["score"]),
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
        "test_loss": float(row["test_loss"]),
        "train_last_step_acc": float(row["train_last_step_acc"]),
        "val_last_step_acc": float(row["val_last_step_acc"]),
        "test_last_step_acc": float(row["test_last_step_acc"]),
    }
    extra = row.get("extra", {})
    if isinstance(extra, dict):
        for key, value in extra.items():
            flat[f"extra_{key}"] = value
    return flat


def _append_csv_row(path: Path, row: Dict[str, Any], header_cache: Dict[Path, List[str]]) -> None:
    if path not in header_cache:
        header_cache[path] = list(row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_cache[path], extrasaction="ignore")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def _best_of(results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    subset = [r for r in results if r["task_type"] == task_type]
    if len(subset) == 0:
        raise RuntimeError(f"No results for task_type={task_type}.")
    return min(subset, key=lambda r: float(r["score"]))


def _best_of_seed(results: List[Dict[str, Any]], task_type: str, seed: int) -> Dict[str, Any]:
    subset = [r for r in results if str(r["task_type"]) == task_type and int(r["seed"]) == int(seed)]
    if len(subset) == 0:
        raise RuntimeError(f"No results for task_type={task_type} seed={seed}.")
    return min(subset, key=lambda r: float(r["score"]))


def _mean_std_test_accs(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute mean/std of empty test-acc list.")
    mean_v = float(np.mean(arr))
    if arr.size == 1:
        return mean_v, 0.0
    return mean_v, float(np.std(arr, ddof=1))


def _hyperparams_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    """Full sweep / run configuration for JSON manifests (reviewer-friendly)."""
    return {
        "dataset": str(args.dataset),
        "pipeline_mode": str(args.pipeline_mode),
        "simple_side": str(args.simple_side),
        "last_layer_readout": str(args.last_layer_readout),
        "K_parallel": int(args.K_parallel),
        "T": int(args.T),
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "loss_type": str(args.loss_type),
        "cvx_method": str(args.cvx_method),
        "optimizer_name": str(args.optimizer_name),
        "batch_size": int(args.batch_size),
        "cvx_epochs": int(args.cvx_epochs),
        "ste_epochs": int(args.ste_epochs),
        "ste_pretrain_epochs": int(args.ste_pretrain_epochs),
        "ste_post_epochs": int(args.ste_post_epochs),
        "num_blocks": int(args.num_blocks),
        "beta_grid": [float(x) for x in args.beta_grid],
        "lr_grid": [float(x) for x in args.lr_grid],
        "bias_grid": [float(x) for x in args.bias_grid],
        "seeds": [int(s) for s in args.seeds],
        "dfa_spec": str(args.dfa_spec),
        "arith_op": str(args.arith_op),
        "arith_base": int(args.arith_base),
        "n_digits": int(args.n_digits),
        "uci_name": str(args.uci_name),
    }


def _task_types_for_bundle(
    results: List[Dict[str, Any]],
    pipeline_mode: str,
    *,
    simple_side: str | None = None,
) -> List[str]:
    pm = str(pipeline_mode)
    if pm == "simple":
        side = str(simple_side) if simple_side is not None else "both"
        if side == "both":
            return ["ste", "cvx"]
        if side == "ste_only":
            return ["ste"]
        if side == "cvx_only":
            return ["cvx"]
        raise ValueError(f"Unknown simple_side={side}.")
    if pm == "fine_tune":
        return ["fine_tune"]
    if pm == "layer_wise":
        return ["layer_wise"]
    raise ValueError(f"Unknown pipeline_mode={pipeline_mode}.")


def _champion_param_record(task_type: str, seed: int, b: Dict[str, Any]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "seed": int(seed),
        "score": float(b["score"]),
        "test_last_step_acc": float(b["test_last_step_acc"]),
        "val_last_step_acc": float(b["val_last_step_acc"]),
        "train_last_step_acc": float(b["train_last_step_acc"]),
    }
    if task_type in ("ste", "cvx"):
        rec["beta"] = float(b["beta"])
        rec["lr"] = float(b["lr"])
    if task_type == "cvx":
        rec["bias"] = float(b["bias"])
    rec["run_id"] = str(b["run_id"])
    rec["pair_id"] = str(b["pair_id"])
    return rec


def _write_seed_bundle(
    *,
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    bundle_dir: Path,
    csv_paths: Dict[str, Path],
    timestamp: str,
) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds]
    simple_side = str(args.simple_side) if str(args.pipeline_mode) == "simple" else None
    task_types = _task_types_for_bundle(results, str(args.pipeline_mode), simple_side=simple_side)
    if len(task_types) == 0:
        raise RuntimeError("No task types in results for seed bundle.")
    for tt in task_types:
        if not any(str(r["task_type"]) == tt for r in results):
            raise RuntimeError(
                f"Seed bundle expects task_type={tt} (simple_side={simple_side!r}) but no rows were returned."
            )

    per_seed_files: Dict[str, str] = {}
    acc_by_tt: Dict[str, List[float]] = {tt: [] for tt in task_types}
    params_by_tt: Dict[str, List[Dict[str, Any]]] = {tt: [] for tt in task_types}

    for seed in seeds:
        seed_payload: Dict[str, Any] = {
            "bundle_timestamp": timestamp,
            "seed": seed,
            "hyperparams": _hyperparams_bundle(args),
            "best": {},
            "csv_paths": {k: str(v.resolve()) for k, v in csv_paths.items()},
        }
        for tt in task_types:
            b = _best_of_seed(results, tt, seed)
            seed_payload["best"][tt] = b
            acc_by_tt[tt].append(float(b["test_last_step_acc"]))
            params_by_tt[tt].append(_champion_param_record(tt, seed, b))
        seed_path = bundle_dir / f"seed_{seed}_{timestamp}.json"
        seed_path.write_text(json.dumps(seed_payload, indent=2, default=str) + "\n")
        per_seed_files[str(seed)] = str(seed_path.resolve())

    aggregate: Dict[str, Any] = {}
    for tt in task_types:
        m, s = _mean_std_test_accs(acc_by_tt[tt])
        aggregate[tt] = {
            "test_last_step_acc_mean": m,
            "test_last_step_acc_std": s,
            "n_seeds": len(acc_by_tt[tt]),
            "per_seed_test_last_step_acc": acc_by_tt[tt],
            "per_seed_champion_params": params_by_tt[tt],
        }

    summary: Dict[str, Any] = {
        "bundle_timestamp": timestamp,
        "created_at": timestamp,
        "bundle_dir": str(bundle_dir.resolve()),
        "pipeline_mode": str(args.pipeline_mode),
        "simple_side": str(args.simple_side),
        "per_seed_json": per_seed_files,
        "hyperparams_tested": _hyperparams_bundle(args),
        "aggregate": aggregate,
    }
    summary_path = bundle_dir / f"multi_seed_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    summary["multi_seed_summary_path"] = str(summary_path.resolve())
    summary["bundle_timestamp"] = timestamp
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue each beta-lr(-bias) combo as an independent run and schedule over GPUs/CPUs."
    )
    parser.add_argument("--pipeline_mode", choices=("simple", "fine_tune", "layer_wise"), default="simple")
    parser.add_argument(
        "--simple_side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Used only when --pipeline_mode simple. Run both sides or only one side.",
    )
    parser.add_argument("--dataset", choices=("mnist_seq", "mnist_perm_seq", "cifar_seq", "arithmetic_seq", "dfa", "uci"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds to run in parallel.")
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--n_train", type=int, default=6000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--P_rec", type=int, default=2048)
    parser.add_argument("--P_last", type=int, default=4096)
    parser.add_argument(
        "--K_parallel",
        type=int,
        default=1,
        help="Parallel SNN/CVX branch count; 1 uses monolithic solvers.",
    )
    parser.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge")
    parser.add_argument("--optimizer_name", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--cvx_method", choices=("cvx", "sgd"), default="sgd")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--cvx_epochs", type=int, default=100)
    parser.add_argument("--ste_epochs", type=int, default=100)
    parser.add_argument("--ste_pretrain_epochs", type=int, default=80)
    parser.add_argument("--ste_post_epochs", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=2, help="Used in pipeline_mode=layer_wise.")
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
    parser.add_argument(
        "--gpu_mem_budget_gb_per_worker",
        type=float,
        default=8.0,
        help=(
            "Assumed RAM (GiB) per STE GPU worker when --num_gpus=-1. Each worker loads PyTorch + the full "
            "dataset; reduce this or set --num_gpus 1 if workers are SIGKILLed (exit -9, often OOM). "
            "Env PARALLEL_SWEEP_GPU_MEM_PER_WORKER_GB overrides this after parse."
        ),
    )
    parser.add_argument("--beta_grid", type=float, nargs="+", default=list(BETA_GRID_DEFAULT))
    parser.add_argument("--lr_grid", type=float, nargs="+", default=list(LR_GRID_DEFAULT))
    parser.add_argument("--bias_grid", type=float, nargs="+", default=list(BIAS_GRID_DEFAULT))
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument(
        "--verbose_runs",
        action="store_true",
        help="If set, stream every STE/CVX run log. Default prints only final best summary.",
    )
    parser.add_argument(
        "--include_all_results_json",
        action="store_true",
        help="If set, include all per-task rows in printed/final JSON output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="atomic/sweep_results",
        help="Directory for per-run CSV/JSON outputs.",
    )
    parser.add_argument(
        "--seed_bundle_dir",
        type=str,
        default="",
        help=(
            "If set, write per-seed champion JSON (seed_<s>.json) and multi_seed_summary.json here. "
            "If empty and len(--seeds)>1, a subdirectory seed_bundle_<task>_<timestamp> is created under --output_dir."
        ),
    )
    parser.add_argument("--ste_csv_path", type=str, default="", help="Optional override for STE per-task CSV.")
    parser.add_argument("--cvx_csv_path", type=str, default="", help="Optional override for CVX per-task CSV.")
    parser.add_argument("--fine_tune_csv_path", type=str, default="", help="Optional override for fine_tune CSV.")
    parser.add_argument("--layer_wise_csv_path", type=str, default="", help="Optional override for layer_wise CSV.")

    # Dataset-specific args mirrored from simple_testing.
    parser.add_argument("--arith_op", choices=("add", "sub", "mul", "div"), default="add")
    parser.add_argument("--arith_base", type=int, default=2)
    parser.add_argument("--n_digits", type=int, default=5)
    parser.add_argument("--dfa_spec", type=str, default="first_last_xor")
    parser.add_argument("--uci_name", type=str, default="pima")
    parser.add_argument("--uci_test_size", type=float, default=0.2)
    parser.add_argument("--uci_val_size", type=float, default=0.2)
    parser.add_argument("--uci_no_standardize", action="store_true")
    parsed = parser.parse_args()
    env_gpu_mem = os.environ.get("PARALLEL_SWEEP_GPU_MEM_PER_WORKER_GB", "").strip()
    if env_gpu_mem != "":
        parsed.gpu_mem_budget_gb_per_worker = float(env_gpu_mem)
    if parsed.seeds is None:
        parsed.seeds = [int(parsed.seed)]
    if len(parsed.seeds) == 0:
        raise ValueError("--seeds cannot be empty.")
    return parsed


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
            gpu_mem_budget_gb_per_worker=float(args.gpu_mem_budget_gb_per_worker),
        )
    else:
        auto_gpu_workers, auto_cpu_workers, hw_stats = 0, 0, {
            "detected_gpu_slots": _detected_gpu_slots(),
            "detected_cpu_cores": int(os.cpu_count() or 2),
            "host_total_mem_gb": _total_system_memory_gb(),
            "cgroup_mem_limit_gb": _cgroup_memory_limit_gb(),
            "slurm_mem_per_node_gb": _slurm_job_memory_gb(),
            "effective_mem_budget_gb": _effective_job_memory_budget_gb(),
            "mem_budget_per_gpu_worker_gb": float(args.gpu_mem_budget_gb_per_worker),
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

    default_csv_paths = _default_csv_paths(args)
    csv_paths: Dict[str, Path] = {
        "ste": Path(args.ste_csv_path).resolve() if str(args.ste_csv_path) else default_csv_paths["ste"],
        "cvx": Path(args.cvx_csv_path).resolve() if str(args.cvx_csv_path) else default_csv_paths["cvx"],
        "fine_tune": (
            Path(args.fine_tune_csv_path).resolve()
            if str(args.fine_tune_csv_path)
            else default_csv_paths["fine_tune"]
        ),
        "layer_wise": (
            Path(args.layer_wise_csv_path).resolve()
            if str(args.layer_wise_csv_path)
            else default_csv_paths["layer_wise"]
        ),
    }
    for p in csv_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    csv_headers: Dict[Path, List[str]] = {}
    stall_polls = 0
    poll_timeout_s = 5.0
    max_stall_polls = 72  # ~6 minutes with 5s polling.
    while len(results) < expected:
        try:
            row = queue_result.get(timeout=poll_timeout_s)
            stall_polls = 0
            results.append(row)
            csv_row = _flatten_result_row(row=row, args=args, gpu_workers=gpu_workers, cpu_workers=cpu_workers)
            row_type = str(row["task_type"])
            if row_type not in csv_paths:
                raise ValueError(f"Unknown task_type in result row: {row_type}")
            out_path = csv_paths[row_type]
            _append_csv_row(path=out_path, row=csv_row, header_cache=csv_headers)
            continue
        except Empty:
            stall_polls += 1

        failed = [(idx, p.pid, p.exitcode) for idx, p in enumerate(workers) if p.exitcode not in (None, 0)]
        alive_count = sum(1 for p in workers if p.is_alive())
        pending = expected - len(results)
        if len(failed) > 0:
            for p in workers:
                if p.is_alive():
                    p.terminate()
            for p in workers:
                p.join(timeout=2)
            failed_str = ", ".join(
                f"worker[{idx}] pid={pid} exitcode={exitcode}" for idx, pid, exitcode in failed
            )
            oom_hint = ""
            if any(int(ex) == -9 for _, _, ex in failed if ex is not None):
                oom_hint = (
                    " Exitcode -9 is SIGKILL (often Linux OOM killer or cgroup limit). "
                    "Retry with --num_gpus 1, a larger Slurm/cgroup memory request, or "
                    "raise --gpu_mem_budget_gb_per_worker only if each worker truly needs more RAM."
                )
            raise RuntimeError(
                "Parallel sweep aborted: one or more workers crashed before returning all results. "
                f"received={len(results)} expected={expected} pending={pending}. Failed: {failed_str}.{oom_hint}"
            )

        if alive_count == 0 and queue_result.empty():
            raise RuntimeError(
                "Parallel sweep stalled: all workers exited but result queue is incomplete. "
                f"received={len(results)} expected={expected} pending={pending}."
            )

        if stall_polls >= max_stall_polls:
            raise RuntimeError(
                "Parallel sweep timeout while waiting for worker results. "
                f"received={len(results)} expected={expected} pending={pending}. "
                "No progress was observed for ~6 minutes."
            )

    for p in workers:
        p.join()

    best: Dict[str, Any] = {}
    if any(r["task_type"] == "ste" for r in results):
        best["ste"] = _best_of(results, task_type="ste")
    if any(r["task_type"] == "cvx" for r in results):
        best["cvx"] = _best_of(results, task_type="cvx")
    if any(r["task_type"] == "fine_tune" for r in results):
        best["fine_tune"] = _best_of(results, task_type="fine_tune")
    if any(r["task_type"] == "layer_wise" for r in results):
        best["layer_wise"] = _best_of(results, task_type="layer_wise")
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
                "fine_tune": "gpu_queue if cvx_method=sgd else cpu_queue",
                "layer_wise": "gpu_queue if cvx_method=sgd else cpu_queue",
            },
            "selection": "min(score) where score=val_loss+beta for STE, score=val_objective (or val_loss) for CVX",
        },
        "sweep": {
            "pipeline_mode": str(args.pipeline_mode),
            "seeds": [int(s) for s in args.seeds],
            "beta_grid": [float(x) for x in args.beta_grid],
            "lr_grid": [float(x) for x in args.lr_grid],
            "bias_grid": [float(x) for x in args.bias_grid],
            "cvx_method": args.cvx_method,
            "num_gpus": int(args.num_gpus),
            "num_cpu_workers": int(args.num_cpu_workers),
            "csv_paths": {k: str(v) for k, v in csv_paths.items()},
            "hyperparams_tested": _hyperparams_bundle(args),
        },
        "best": best,
    }
    if bool(args.include_all_results_json):
        out["all_results"] = results

    bundle_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_bundle = len(args.seeds) > 1 or str(getattr(args, "seed_bundle_dir", "")).strip() != ""
    if write_bundle:
        task_nm = _build_task_name(args)
        if str(args.seed_bundle_dir).strip():
            bundle_base = Path(str(args.seed_bundle_dir)).expanduser().resolve()
        else:
            bundle_base = Path(args.output_dir).expanduser().resolve() / f"seed_bundle_{task_nm}_{bundle_ts}"
        out["seed_bundle"] = _write_seed_bundle(
            results=results,
            args=args,
            bundle_dir=bundle_base,
            csv_paths=csv_paths,
            timestamp=bundle_ts,
        )
        print(
            f"[seed-bundle] wrote per-seed champions under {out['seed_bundle']['bundle_dir']} "
            f"summary={out['seed_bundle']['multi_seed_summary_path']}",
            flush=True,
        )
        for tt, block in out["seed_bundle"]["aggregate"].items():
            print(
                f"[aggregate-{tt}] test_last_step_acc mean={float(block['test_last_step_acc_mean']):.6f} "
                f"std={float(block['test_last_step_acc_std']):.6f} n_seeds={int(block['n_seeds'])}",
                flush=True,
            )

    # Human-readable summary: only best combos, printed once at end.
    if str(args.pipeline_mode) == "simple":
        if "ste" in best:
            bste = best["ste"]
            ste_extra = bste.get("extra", {}) if isinstance(bste, dict) else {}
            ste_arith_suffix = _arithmetic_acc_suffix(ste_extra if isinstance(ste_extra, dict) else {})
            if ste_arith_suffix:
                ste_acc_part = ste_arith_suffix
            else:
                ste_acc_part = f" test_acc={float(bste['test_last_step_acc']):.6f}"
            print(
                "[best-ste] "
                f"seed={int(bste['seed'])} beta={float(bste['beta']):.8g} lr={float(bste['lr']):.8g} "
                f"val_score={float(bste['score']):.6f}{ste_acc_part}"
            )
        if "cvx" in best:
            bcvx = best["cvx"]
            cvx_extra = bcvx.get("extra", {}) if isinstance(bcvx, dict) else {}
            cvx_arith_suffix = _arithmetic_acc_suffix(cvx_extra if isinstance(cvx_extra, dict) else {})
            if cvx_arith_suffix:
                cvx_acc_part = cvx_arith_suffix
            else:
                cvx_acc_part = f" test_acc={float(bcvx['test_last_step_acc']):.6f}"
            print(
                "[best-cvx] "
                f"seed={int(bcvx['seed'])} beta={float(bcvx['beta']):.8g} lr={float(bcvx['lr']):.8g} bias={float(bcvx['bias']):.8g} "
                f"val_score={float(bcvx['score']):.6f}{cvx_acc_part}"
            )
    elif str(args.pipeline_mode) == "fine_tune" and "fine_tune" in best:
        bft = best["fine_tune"]
        print(
            "[best-fine_tune] "
            f"seed={int(bft['seed'])} val_score={float(bft['score']):.6f} test_acc={float(bft['test_last_step_acc']):.6f}"
        )
    elif str(args.pipeline_mode) == "layer_wise" and "layer_wise" in best:
        blw = best["layer_wise"]
        print(
            "[best-layer_wise] "
            f"seed={int(blw['seed'])} val_score={float(blw['score']):.6f} test_acc={float(blw['test_last_step_acc']):.6f}"
        )

    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
