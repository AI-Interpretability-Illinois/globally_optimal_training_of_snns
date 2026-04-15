from __future__ import annotations

import argparse
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
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .fine_tune import FineTuneConfig, run_fine_tune_pipeline
    from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from .simple_testing import _cvx_last_step_acc, _load_dataset_from_args, _set_seed, _ste_last_step_acc
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
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
    _set_seed(int(task.seed))

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
            beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
            bias_grid=tuple(float(x) for x in _WORKER_ARGS.bias_grid),
            ste_beta_grid=tuple(float(x) for x in _WORKER_ARGS.beta_grid),
            ste_lr_grid=tuple(float(x) for x in _WORKER_ARGS.lr_grid),
        )
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
            if not isinstance(ste_pre.model, SNNBaselineSeq):
                raise TypeError("Expected SNNBaselineSeq model in fine_tune ste_pretrain result.")
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
                ),
            )
        )
        return SweepResult(
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
        )
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
        if not isinstance(ste_pre.model, SNNBaselineSeq):
            raise TypeError("Expected SNNBaselineSeq model in layer-wise ste_pre result.")
        cvx_init = InitializationConfig(
            mode="pretraining",
            seed=int(task.seed),
            L=int(_WORKER_ARGS.L),
            P_rec=int(_WORKER_ARGS.P_rec),
            P_last=int(_WORKER_ARGS.P_last),
            feature_count=int(_WORKER_ARGS.P_last),
            bias=float(final_block["cvx_selected_params"]["bias"]),
            pretrained_weights=_extract_weights_from_snn(ste_pre.model),
        )
        return SweepResult(
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
        for seed in seeds:
            for lr in args.lr_grid:
                for beta in args.beta_grid:
                    pair_id = f"{args.dataset}|seed={seed}|beta={float(beta):.8g}|lr={float(lr):.8g}"
                    ste_run_id = f"ste|{pair_id}"
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


def _extract_weights_from_snn(model: SNNBaselineSeq) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    for fc in model.fcs:
        weights.append(fc.weight.detach().cpu().numpy().copy())
    weights.append(model.classifier.weight.detach().cpu().numpy().copy())
    return weights


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue each beta-lr(-bias) combo as an independent run and schedule over GPUs/CPUs."
    )
    parser.add_argument("--pipeline_mode", choices=("simple", "fine_tune", "layer_wise"), default="simple")
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
    parser.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge")
    parser.add_argument("--optimizer_name", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--cvx_method", choices=("cvx", "sgd"), default="sgd")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--cvx_epochs", type=int, default=150)
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
    parser.add_argument("--beta_grid", type=float, nargs="+", default=list(BETA_GRID_DEFAULT))
    parser.add_argument("--lr_grid", type=float, nargs="+", default=list(LR_GRID_DEFAULT))
    parser.add_argument("--bias_grid", type=float, nargs="+", default=list(BIAS_GRID_DEFAULT))
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="atomic/sweep_results",
        help="Directory for per-run CSV/JSON outputs.",
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
    for _ in range(expected):
        row = queue_result.get()
        results.append(row)
        csv_row = _flatten_result_row(row=row, args=args, gpu_workers=gpu_workers, cpu_workers=cpu_workers)
        row_type = str(row["task_type"])
        if row_type not in csv_paths:
            raise ValueError(f"Unknown task_type in result row: {row_type}")
        out_path = csv_paths[row_type]
        _append_csv_row(path=out_path, row=csv_row, header_cache=csv_headers)
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
        },
        "best": best,
        "all_results": results,
    }
    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
