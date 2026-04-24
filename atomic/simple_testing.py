from __future__ import annotations

import argparse
import io
import json
import re
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import ArithmeticDataset, load_arithmetic_dataset
    from data_loaders.dfa_data_loader import make_dfa_dataset
    from data_loaders.image_data_loader import ImageSequenceDataset, load_cifar_seq_dataset, load_mnist_seq_dataset
    from data_loaders.uci_data_loader import UciDataset, load_uci_dataset
    from finetune_manifest import assert_finetune_manifest_matches_runtime, validate_finetune_weight_shapes_against_manifest
    from fine_tune import (
        FineTuneConfig,
        _extract_weight_list,
        load_finetune_weight_checkpoint,
        run_fine_tune_pipeline,
    )
    from layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from solvers.cvx_solve import (
        InitializationConfig,
        SolveConfig,
        _build_feature_map,
        _prepare_sequence_targets,
        cvx_solve,
    )
    from solvers import ste_parallel_Solve as ste_par
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, SteSolveResult, ste_solve
else:
    from .data_loaders.arithmetic_data_loader import ArithmeticDataset, load_arithmetic_dataset
    from .data_loaders.dfa_data_loader import make_dfa_dataset
    from .data_loaders.image_data_loader import ImageSequenceDataset, load_cifar_seq_dataset, load_mnist_seq_dataset
    from .data_loaders.uci_data_loader import UciDataset, load_uci_dataset
    from .finetune_manifest import assert_finetune_manifest_matches_runtime, validate_finetune_weight_shapes_against_manifest
    from .fine_tune import (
        FineTuneConfig,
        _extract_weight_list,
        load_finetune_weight_checkpoint,
        run_fine_tune_pipeline,
    )
    from .layer_wise_stacking_test_bench import LayerWiseConfig, run_layer_wise_stacking_test_bench
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from .solvers.cvx_solve import (
        InitializationConfig,
        SolveConfig,
        _build_feature_map,
        _prepare_sequence_targets,
        cvx_solve,
    )
    from .solvers import ste_parallel_Solve as ste_par
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, SteSolveResult, ste_solve

# K_parallel>1: ste_solve.ste_solve forwards to ste_parallel_Solve, which returns that module's SteSolveResult.
_STE_SOLVE_RESULT_TYPES = (SteSolveResult, ste_par.SteSolveResult)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_cvx_device(device_name: str) -> torch.device | None:
    if device_name == "auto":
        return None
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --cvx_device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested --cvx_device mps, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown cvx device option: {device_name}")


def _as_seq_from_uci(ds: UciDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    x_train = ds.X_train[:, None, :].astype(np.float32, copy=False)
    x_val = ds.X_val[:, None, :].astype(np.float32, copy=False)
    x_test = ds.X_test[:, None, :].astype(np.float32, copy=False)
    num_classes = int(np.max(ds.y_train) + 1)
    return x_train, ds.y_train, x_val, ds.y_val, x_test, ds.y_test, num_classes, int(x_train.shape[2])


def _load_dataset_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset in ("mnist_seq", "mnist_perm_seq"):
        ds: ImageSequenceDataset = load_mnist_seq_dataset(
            task=args.dataset,
            T=args.T,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": ds.d_in,
        }
    if args.dataset == "cifar_seq":
        ds = load_cifar_seq_dataset(
            task=args.dataset,
            T=args.T,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": ds.d_in,
        }
    if args.dataset == "arithmetic_seq":
        ds: ArithmeticDataset = load_arithmetic_dataset(
            op=args.arith_op,
            base=args.arith_base,
            n_digits=args.n_digits,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": ds.num_classes,
            "d_in": int(ds.X_train.shape[2]),
        }
    if args.dataset == "dfa":
        n_total = args.n_train + args.n_val + args.n_test
        x_all, y_all, num_classes = make_dfa_dataset(
            dfa_spec=args.dfa_spec,
            n=n_total,
            T=args.T,
            seed=args.seed,
            balanced=True,
        )
        x_train = x_all[: args.n_train]
        y_train = y_all[: args.n_train]
        x_val = x_all[args.n_train : args.n_train + args.n_val]
        y_val = y_all[args.n_train : args.n_train + args.n_val]
        x_test = x_all[args.n_train + args.n_val :]
        y_test = y_all[args.n_train + args.n_val :]
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "num_classes": int(num_classes),
            "d_in": int(x_train.shape[2]),
        }
    if args.dataset == "uci":
        ds = load_uci_dataset(
            name=args.uci_name,
            seed=args.seed,
            test_size=args.uci_test_size,
            val_size=args.uci_val_size,
            standardize=not args.uci_no_standardize,
        )
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes, d_in = _as_seq_from_uci(ds)
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "num_classes": num_classes,
            "d_in": d_in,
        }
    raise ValueError(f"Unsupported dataset={args.dataset}.")


def _build_cvx_features_for_eval(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
    all_timesteps: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Must match ``cvx_solve`` routing: K_parallel>1 uses parallel feature map (same as training)."""
    if int(getattr(init_cfg, "K_parallel", 1)) > 1:
        if __package__ in (None, ""):
            from solvers import cvx_parallel_Solve as cvx_par
        else:
            from .solvers import cvx_parallel_Solve as cvx_par

        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        return cvx_par._build_feature_map(
            x_train,
            x_val,
            x_test,
            pic,
            all_timesteps=all_timesteps,
        )
    return _build_feature_map(
        x_train,
        x_val,
        x_test,
        init_cfg,
        all_timesteps=all_timesteps,
    )


def _ste_last_step_acc(model: torch.nn.Module, x_test: np.ndarray, y_test: np.ndarray) -> float:
    device = next(model.parameters()).device
    x = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
        preds = logits[:, -1, :].argmax(dim=1).detach().cpu().numpy()
    if y_test.ndim == 2:
        y_last = y_test[:, -1]
    else:
        y_last = y_test
    return float(np.mean(preds == y_last))


def _cvx_last_step_acc(
    cvx_result: Any,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    init_cfg: InitializationConfig,
) -> float:
    supervise_all = y_test.ndim == 2
    _, _, d_test, _ = _build_cvx_features_for_eval(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        init_cfg=init_cfg,
        all_timesteps=supervise_all,
    )
    y_last = _prepare_sequence_targets(y_test)
    if isinstance(cvx_result.trained_model, dict) and "weights" in cvx_result.trained_model:
        w = cvx_result.trained_model["weights"]
        scores = d_test @ w
        preds = scores.argmax(axis=1)
        if supervise_all:
            n, steps = y_test.shape
            if int(preds.shape[0]) != n * steps:
                raise ValueError(
                    f"All-timestep CVX: expected {n * steps} score rows, got {preds.shape[0]}."
                )
            preds = preds.reshape(n, steps)[:, -1]
        return float(np.mean(preds == y_last))
    model = cvx_result.trained_model
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(d_test, dtype=torch.float32))
        preds = logits.argmax(dim=1).cpu().numpy()
    if supervise_all:
        n, steps = y_test.shape
        if int(preds.shape[0]) != n * steps:
            raise ValueError(
                f"All-timestep CVX: expected {n * steps} score rows, got {preds.shape[0]}."
            )
        preds = preds.reshape(n, steps)[:, -1]
    return float(np.mean(preds == y_last))


def _grid_to_list(grid: Any) -> list[float]:
    if isinstance(grid, (tuple, list)):
        return [float(v) for v in grid]
    return [float(grid)]


def _resolve_simple_bias_grid(args: argparse.Namespace) -> list[float]:
    if args.bias_grid is None:
        # In simple mode we do not sweep bias unless explicitly requested.
        return [0.0]
    return [float(v) for v in args.bias_grid]


def _arithmetic_reported_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "train_token_loss": float(metrics["train_token_loss"]),
        "train_seq_loss": float(metrics["train_seq_loss"]),
        "val_token_acc": float(metrics["val_token_acc"]),
        "val_seq_acc": float(metrics["val_seq_acc"]),
        "test_token_acc": float(metrics["test_token_acc"]),
        "test_seq_acc": float(metrics["test_seq_acc"]),
    }


# Matches ``fine_tune.run_fine_tune_pipeline`` final reruns with ``log_every=10``.
_FINE_TUNE_SOLVER_LOG_EVERY = 10


def _cvx_json_key_for_init_mode(init_mode: str, *, source_tag: str | None = None) -> str:
    """Top-level JSON field for a CVX run, e.g. ``gaussian`` -> ``cvx_gaussian`` (same naming as legacy simple mode)."""
    m = str(init_mode).strip().lower()
    if m == "gaussian":
        base = "cvx_gaussian"
    elif m == "pretraining":
        base = "cvx_pretraining"
    else:
        base = f"cvx_{m}"
    if source_tag is None:
        return base
    st = str(source_tag).strip().lower().replace(" ", "_")
    return f"{base}__{st}"


def _experiment_hyperparameters(
    args: argparse.Namespace,
    data: Dict[str, Any],
    *,
    search_grids: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Serializable run configuration only (no filesystem paths, manifests, or weights)."""
    hp: Dict[str, Any] = {
        "mode": args.mode,
        "dataset": args.dataset_tag,
        "seed": int(args.seed),
        "T": int(args.T),
        "L": int(args.L),
        "d_in": int(data["d_in"]),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "num_classes": int(data["num_classes"]),
        "loss_type": str(args.loss_type),
        "optimizer_name": str(args.optimizer_name),
        "cvx_method": str(args.cvx_method),
        "batch_size": int(args.batch_size),
        "cvx_epochs": int(args.cvx_epochs),
        "last_layer_readout": str(args.last_layer_readout),
    }
    if search_grids is not None:
        hp["search_grids"] = search_grids
    if args.mode == "simple":
        hp["simple_side"] = str(args.simple_side)
        hp["ste_epochs"] = int(args.ste_epochs)
        hp["cvx_device"] = str(args.cvx_device)
        hp["used_pretrained_weight_init"] = bool(str(getattr(args, "init_weights_dir", "") or "").strip())
        if hp["used_pretrained_weight_init"]:
            hp["init_weights_variant"] = str(args.init_weights_variant)
        if args.bias_grid is not None:
            hp["bias_grid_cli"] = [float(x) for x in args.bias_grid]
    elif args.mode == "fine_tune":
        hp["ste_pretrain_epochs"] = int(args.ste_pretrain_epochs)
        hp["ste_post_epochs"] = int(args.ste_post_epochs)
        hp["used_pretrained_weight_init"] = bool(str(getattr(args, "init_weights_dir", "") or "").strip())
        if hp["used_pretrained_weight_init"]:
            hp["init_weights_variant"] = str(args.init_weights_variant)
        hp["weights_export_enabled"] = _resolve_weights_save_dir(args) is not None
    elif args.mode == "layer_wise":
        hp["num_blocks"] = int(args.num_blocks)
        hp["ste_pretrain_epochs"] = int(args.ste_pretrain_epochs)
        hp["ste_finetune_epochs"] = int(args.ste_post_epochs)
    if args.dataset == "arithmetic_seq":
        hp["arith_op"] = str(args.arith_op)
        hp["arith_base"] = int(args.arith_base)
        hp["n_digits"] = int(args.n_digits)
    if args.dataset == "dfa":
        hp["dfa_spec"] = str(args.dfa_spec)
    if args.dataset == "uci":
        hp["uci_name"] = str(args.uci_name)
        hp["uci_test_size"] = float(args.uci_test_size)
        hp["uci_val_size"] = float(args.uci_val_size)
        hp["uci_standardize"] = not bool(args.uci_no_standardize)
    return hp


def _flat_sweep_json_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """Match on-disk ``sweep_results/*.json`` layout: flat ``mode``, ``dataset``, ``fixed_grids``, ``simple_side`` (simple only), then metric blocks (``ste``, ``cvx_*``, ``ste_pretrain``, …)."""
    if "hyperparameters" not in result or "metrics" not in result:
        raise TypeError("Expected result dict with 'hyperparameters' and 'metrics'.")
    hp = result["hyperparameters"]
    m = result["metrics"]
    mode = str(hp["mode"])
    grids = hp.get("search_grids")
    out: Dict[str, Any] = {
        "mode": mode,
        "dataset": hp["dataset"],
        "fixed_grids": dict(grids) if grids is not None else {},
    }
    if mode == "simple":
        out["simple_side"] = hp["simple_side"]
    for k, v in m.items():
        out[k] = v
    return out


def _ste_training_curve_from_loss_history(loss_history: List[float], log_every: int) -> Dict[str, List[float]]:
    if log_every <= 0 or not loss_history:
        return {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    return {
        "epoch": [float((i + 1) * log_every) for i in range(len(loss_history))],
        "train_loss": [float(v) for v in loss_history],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }


def _cvx_training_curve_for_json(
    cvx_result: Any,
    *,
    log_every: int,
    epochs: int,
) -> Dict[str, List[float]]:
    _ = epochs
    lh = list(cvx_result.loss_history)
    empty: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_obj": [],
        "val_obj": [],
        "train_acc": [],
        "val_acc": [],
    }
    if not lh:
        return empty
    if len(lh) == 1:
        return {
            "epoch": [1.0],
            "train_loss": [float(lh[0])],
            "val_loss": [],
            "train_obj": [],
            "val_obj": [],
            "train_acc": [],
            "val_acc": [],
        }
    if log_every <= 0:
        return empty
    return {
        "epoch": [float((i + 1) * log_every) for i in range(len(lh))],
        "train_loss": [float(v) for v in lh],
        "val_loss": [],
        "train_obj": [],
        "val_obj": [],
        "train_acc": [],
        "val_acc": [],
    }


def _serialize_ste_block_simple_style(
    ste_res: SteSolveResult,
    *,
    selected_params: Dict[str, float],
    x_test: np.ndarray,
    y_test: np.ndarray,
    arithmetic_mode: bool,
    log_every_for_curve: int,
) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "selected_params": {k: float(v) for k, v in selected_params.items()},
        "best_losses": dict(ste_res.best_losses),
        "training_curve": _ste_training_curve_from_loss_history(ste_res.loss_history, log_every_for_curve),
    }
    if arithmetic_mode:
        block["reported_metrics"] = _arithmetic_reported_metrics(ste_res.best_losses)
    else:
        block["test_last_step_acc"] = _ste_last_step_acc(ste_res.model, x_test, y_test)
    return block


def _serialize_cvx_block_simple_style(
    cvx_res: Any,
    *,
    selected_params: Dict[str, Any],
    init_cfg_eval: InitializationConfig,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    arithmetic_mode: bool,
    cvx_epochs: int,
    log_every_for_curve: int,
) -> Dict[str, Any]:
    sel = dict(selected_params)
    block: Dict[str, Any] = {
        "selected_params": sel,
        "final_losses": dict(cvx_res.final_losses),
        "diagnostics": asdict(cvx_res.diagnostics),
        "training_curve": _cvx_training_curve_for_json(
            cvx_res, log_every=log_every_for_curve, epochs=cvx_epochs
        ),
    }
    if arithmetic_mode:
        block["reported_metrics"] = _arithmetic_reported_metrics(cvx_res.final_losses)
    else:
        block["test_last_step_acc"] = _cvx_last_step_acc(
            cvx_res,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_test=y_test,
            init_cfg=init_cfg_eval,
        )
    return block


def _finetune_pipeline_to_simple_style_json(
    pipeline_out: Dict[str, Any],
    args: argparse.Namespace,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    bias_grid_default = _grid_to_list(BIAS_GRID_DEFAULT)
    cvx_lr_eff = cvx_lr_sweep_values(args.cvx_method, LR_GRID_DEFAULT)
    fixed_grids: Dict[str, Any] = {
        "beta_grid": list(BETA_GRID_DEFAULT),
        "lr_grid": list(LR_GRID_DEFAULT),
        "bias_grid": bias_grid_default,
        "ste_lr_grid": list(LR_GRID_DEFAULT),
        "ste_beta_grid": list(BETA_GRID_DEFAULT),
        "cvx_lr_grid_effective": list(cvx_lr_eff),
    }
    if args.cvx_method == "cvx":
        fixed_grids["cvx_lr_sweep_note"] = (
            "For method=cvx, lr is not swept (single placeholder 0.0); sweep is beta × bias only."
        )

    selected_readout = str(pipeline_out["selected_readout"])
    br: Dict[str, Any] = pipeline_out["by_readout"][selected_readout]
    pre = br["ste_pretrain"]
    post = br["ste_post"]
    if not isinstance(pre, _STE_SOLVE_RESULT_TYPES) or not isinstance(post, _STE_SOLVE_RESULT_TYPES):
        raise TypeError("Expected SteSolveResult for fine_tune STE stages.")

    arithmetic_mode = args.dataset == "arithmetic_seq"
    x_train = data["x_train"]
    x_val = data["x_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    ste_pre_json = _serialize_ste_block_simple_style(
        pre,
        selected_params=dict(br["ste_pretrain_selected_params"]),
        x_test=x_test,
        y_test=y_test,
        arithmetic_mode=arithmetic_mode,
        log_every_for_curve=0,
    )
    ste_pre_json["pretrain_accuracy"] = dict(br["pretrain_accuracy"])

    hp = _experiment_hyperparameters(args, data, search_grids=fixed_grids)
    metrics: Dict[str, Any] = {
        "ste_pretrain": ste_pre_json,
        "ste_post": _serialize_ste_block_simple_style(
            post,
            selected_params=dict(br["ste_post_selected_params"]),
            x_test=x_test,
            y_test=y_test,
            arithmetic_mode=arithmetic_mode,
            log_every_for_curve=_FINE_TUNE_SOLVER_LOG_EVERY,
        ),
    }
    cvx_by_src: Dict[str, Any] = br["cvx_by_source"]
    init_mode = "pretraining"
    cvx_items = list(cvx_by_src.items())
    if len(cvx_items) == 0:
        raise ValueError("fine_tune json: cvx_by_source is empty.")
    if len(cvx_items) == 1:
        src, cvx_one = cvx_items[0]
        cvx_pk = _cvx_json_key_for_init_mode(init_mode)
        cvx_sel_one = br["cvx_selected_params"][src]
        init_one = InitializationConfig(
            mode=init_mode,
            seed=int(args.seed),
            feature_count=int(args.P_last),
            bias=float(cvx_sel_one["bias"]),
            pretrained_weights=_extract_weight_list(pre.model),
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=selected_readout,
        )
        metrics[cvx_pk] = _serialize_cvx_block_simple_style(
            cvx_one,
            selected_params=dict(cvx_sel_one),
            init_cfg_eval=init_one,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_test=y_test,
            arithmetic_mode=arithmetic_mode,
            cvx_epochs=int(args.cvx_epochs),
            log_every_for_curve=_FINE_TUNE_SOLVER_LOG_EVERY,
        )
    else:
        for src, cvx_one in cvx_items:
            cvx_pk = _cvx_json_key_for_init_mode(init_mode, source_tag=src)
            if cvx_pk in metrics:
                raise ValueError(f"Duplicate CVX JSON key {cvx_pk!r} for fine_tune cvx_by_source.")
            cvx_sel_one = br["cvx_selected_params"][src]
            init_one = InitializationConfig(
                mode=init_mode,
                seed=int(args.seed),
                feature_count=int(args.P_last),
                bias=float(cvx_sel_one["bias"]),
                pretrained_weights=_extract_weight_list(pre.model),
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                last_layer_readout=selected_readout,
            )
            metrics[cvx_pk] = _serialize_cvx_block_simple_style(
                cvx_one,
                selected_params=dict(cvx_sel_one),
                init_cfg_eval=init_one,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                arithmetic_mode=arithmetic_mode,
                cvx_epochs=int(args.cvx_epochs),
                log_every_for_curve=_FINE_TUNE_SOLVER_LOG_EVERY,
            )
    return {"hyperparameters": hp, "metrics": metrics}


def _layer_wise_pipeline_to_simple_style_json(
    block_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    bias_grid_default = _grid_to_list(BIAS_GRID_DEFAULT)
    cvx_lr_eff = cvx_lr_sweep_values(args.cvx_method, LR_GRID_DEFAULT)
    fixed_grids: Dict[str, Any] = {
        "cvx_beta_grid": list(BETA_GRID_DEFAULT),
        "cvx_lr_grid": list(cvx_lr_eff),
        "cvx_bias_grid": bias_grid_default,
        "ste_lr_grid": list(LR_GRID_DEFAULT),
        "ste_beta_grid": list(BETA_GRID_DEFAULT),
    }
    if args.cvx_method == "cvx":
        fixed_grids["cvx_lr_sweep_note"] = (
            "For method=cvx, lr is not swept (single placeholder 0.0); sweep is beta × bias only."
        )

    arithmetic_mode = args.dataset == "arithmetic_seq"
    x_train = data["x_train"]
    x_val = data["x_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    blocks_out: List[Dict[str, Any]] = []
    for block in block_rows:
        ste_pre = block["ste_pre"]
        cvx_b = block["cvx"]
        ste_ft = block["ste_finetune"]
        if not isinstance(ste_pre, _STE_SOLVE_RESULT_TYPES) or not isinstance(ste_ft, _STE_SOLVE_RESULT_TYPES):
            raise TypeError("Expected SteSolveResult in layer_wise block.")
        readout = str(ste_pre.model.last_layer_readout)
        cvx_sel = dict(block["cvx_selected_params"])
        init_cfg_eval = InitializationConfig(
            mode="pretraining",
            seed=int(args.seed),
            feature_count=int(args.P_last),
            bias=float(cvx_sel["bias"]),
            pretrained_weights=_extract_weight_list(ste_pre.model),
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=readout,
        )
        cvx_field = _cvx_json_key_for_init_mode("pretraining")
        blocks_out.append(
            {
                "block_idx": int(block["block_idx"]),
                "ste_pretrain": _serialize_ste_block_simple_style(
                    ste_pre,
                    selected_params=dict(block["ste_pre_selected_params"]),
                    x_test=x_test,
                    y_test=y_test,
                    arithmetic_mode=arithmetic_mode,
                    log_every_for_curve=0,
                ),
                cvx_field: _serialize_cvx_block_simple_style(
                    cvx_b,
                    selected_params=cvx_sel,
                    init_cfg_eval=init_cfg_eval,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_test,
                    y_test=y_test,
                    arithmetic_mode=arithmetic_mode,
                    cvx_epochs=int(args.cvx_epochs),
                    log_every_for_curve=_FINE_TUNE_SOLVER_LOG_EVERY,
                ),
                "ste_post": _serialize_ste_block_simple_style(
                    ste_ft,
                    selected_params=dict(block["ste_finetune_selected_params"]),
                    x_test=x_test,
                    y_test=y_test,
                    arithmetic_mode=arithmetic_mode,
                    log_every_for_curve=0,
                ),
            }
        )

    hp = _experiment_hyperparameters(args, data, search_grids=fixed_grids)
    return {"hyperparameters": hp, "metrics": {"blocks": blocks_out}}


def _capture_solver_stdout(fn: Any, **kwargs: Any) -> Tuple[Any, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(**kwargs)
    return out, buf.getvalue()


def _extract_ste_curve(log_text: str) -> Dict[str, list[float]]:
    pattern = re.compile(
        r"\[ste\] epoch=(\d+)/(\d+) train_loss=([-+eE0-9\.]+) val_loss=([-+eE0-9\.]+) "
        r"train_acc=([-+eE0-9\.]+) val_acc=([-+eE0-9\.]+)"
    )
    curve: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    for match in pattern.finditer(log_text):
        curve["epoch"].append(float(match.group(1)))
        curve["train_loss"].append(float(match.group(3)))
        curve["val_loss"].append(float(match.group(4)))
        curve["train_acc"].append(float(match.group(5)))
        curve["val_acc"].append(float(match.group(6)))
    return curve


def _extract_cvx_curve(log_text: str) -> Dict[str, list[float]]:
    pattern = re.compile(
        r"\[cvx-sgd\] epoch=(\d+)/(\d+) train_loss=([-+eE0-9\.]+) val_loss=([-+eE0-9\.]+) "
        r"train_obj=([-+eE0-9\.]+) val_obj=([-+eE0-9\.]+) train_acc=([-+eE0-9\.]+) val_acc=([-+eE0-9\.]+)"
    )
    curve: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_obj": [],
        "val_obj": [],
        "train_acc": [],
        "val_acc": [],
    }
    for match in pattern.finditer(log_text):
        curve["epoch"].append(float(match.group(1)))
        curve["train_loss"].append(float(match.group(3)))
        curve["val_loss"].append(float(match.group(4)))
        curve["train_obj"].append(float(match.group(5)))
        curve["val_obj"].append(float(match.group(6)))
        curve["train_acc"].append(float(match.group(7)))
        curve["val_acc"].append(float(match.group(8)))
    return curve


_DATASET_CLI_BASE = frozenset(
    {"mnist_seq", "mnist_perm_seq", "cifar_seq", "arithmetic_seq", "dfa", "uci"}
)


def _dataset_tag(args: argparse.Namespace) -> str:
    """Human-readable dataset id for JSON (for DFA always ``dfa:<spec>``)."""
    if str(args.dataset) == "dfa":
        return f"dfa:{args.dfa_spec}"
    return str(args.dataset)


def _apply_dataset_cli(args: argparse.Namespace) -> None:
    """Parse ``--dataset``; supports ``dfa:<dfa_spec>`` to name the exact automaton (plain ``dfa`` uses ``--dfa_spec``)."""
    raw = str(args.dataset).strip()
    if raw.startswith("dfa:"):
        spec = raw.split(":", 1)[1].strip()
        if not spec:
            raise ValueError("Invalid --dataset: empty name after 'dfa:'.")
        args.dataset = "dfa"
        args.dfa_spec = spec
    elif raw in _DATASET_CLI_BASE:
        args.dataset = raw
    else:
        raise ValueError(
            f"Unsupported --dataset {raw!r}. Expected one of {sorted(_DATASET_CLI_BASE)} "
            "or 'dfa:<dfa_spec>' (e.g. 'dfa:first_last_xor')."
        )
    args.dataset_tag = _dataset_tag(args)


def _resolve_task_name(args: argparse.Namespace) -> str:
    if args.dataset == "dfa":
        raw = args.dfa_spec
    elif args.dataset == "arithmetic_seq":
        raw = f"{args.dataset}_{args.arith_op}"
    elif args.dataset == "uci":
        raw = f"{args.dataset}_{args.uci_name}"
    else:
        raw = args.dataset
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")


def _resolve_weights_save_dir(args: argparse.Namespace) -> str | None:
    """Relative paths are resolved under the atomic package directory (e.g. finetune_weights -> atomic/finetune_weights)."""
    raw = (args.weights_save_dir or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p.resolve())


def _artifact_paths(args: argparse.Namespace) -> Dict[str, Path]:
    task_name = _resolve_task_name(args)
    out_dir = Path(__file__).resolve().parent / "sweep_results" / f"{task_name}_{args.L}_{args.T}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_P_last{args.P_last}_P_rec{args.P_rec}_n_train{args.n_train}"
    return {
        "out_dir": out_dir,
        "json": out_dir / f"{stem}.json",
        "md": out_dir / f"{stem}.md",
        "png": out_dir / f"{stem}.png",
    }


def _make_simple_summary_md(args: argparse.Namespace, result: Dict[str, Any]) -> str:
    title_task = _resolve_task_name(args)
    lines = [
        f"# Simple Run Summary ({title_task}, L={args.L}, T={args.T})",
        "",
        "| Hyperparameter | Value |",
        "|---|---:|",
        f"| P_rec | {args.P_rec} |",
        f"| P_last | {args.P_last} |",
        f"| n_train | {args.n_train} |",
        f"| n_val | {args.n_val} |",
        f"| n_test | {args.n_test} |",
        "",
        "| Model | Selected lr | Selected beta | Selected bias | Train loss | Val loss | Test loss | Test acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    m = result["metrics"]
    ste = m.get("ste")
    if ste is not None:
        ste_p = ste["selected_params"]
        ste_l = ste["best_losses"]
        if "test_last_step_acc" in ste:
            ste_acc = float(ste["test_last_step_acc"])
        else:
            ste_acc = float(ste["reported_metrics"]["test_seq_acc"])
        lines.append(
            (
                f"| STE | {ste_p['lr']:.6g} | {ste_p['beta']:.6g} | - | "
                f"{ste_l['train_loss']:.6f} | {ste_l['val_loss']:.6f} | {ste_l['test_loss']:.6f} | "
                f"{ste_acc:.4f} |"
            )
        )
    for cvx_key in sorted(k for k in m if k.startswith("cvx_")):
        cvx = m[cvx_key]
        cvx_p = cvx["selected_params"]
        cvx_l = cvx["final_losses"]
        lr_cell = "-" if cvx_p.get("lr") is None else f"{float(cvx_p['lr']):.6g}"
        init_label = cvx_key[len("cvx_") :].split("__", 1)[0]
        if "test_last_step_acc" in cvx:
            cvx_acc = float(cvx["test_last_step_acc"])
        else:
            cvx_acc = float(cvx["reported_metrics"]["test_seq_acc"])
        lines.append(
            (
                f"| CVX ({init_label}, {args.cvx_method}) | {lr_cell} | {cvx_p['beta']:.6g} | {cvx_p['bias']:.6g} | "
                f"{cvx_l['train_loss']:.6f} | {cvx_l['val_loss']:.6f} | {cvx_l['test_loss']:.6f} | "
                f"{cvx_acc:.4f} |"
            )
        )
    return "\n".join(lines) + "\n"


def _save_simple_plot(png_path: Path, ste_curve: Dict[str, list[float]], cvx_curve: Dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    if ste_curve["epoch"]:
        ax_acc.plot(ste_curve["epoch"], ste_curve["train_acc"], marker="o", linewidth=2.2, label="SNN train_acc")
        ax_acc.plot(ste_curve["epoch"], ste_curve["val_acc"], marker="s", linewidth=2.2, label="SNN val_acc")
        ax_loss.plot(ste_curve["epoch"], ste_curve["train_loss"], marker="o", linewidth=2.2, label="SNN train_loss")
        ax_loss.plot(ste_curve["epoch"], ste_curve["val_loss"], marker="s", linewidth=2.2, label="SNN val_loss")
    if cvx_curve["epoch"]:
        ax_acc.plot(cvx_curve["epoch"], cvx_curve["train_acc"], marker="^", linewidth=2.2, label="CVX train_acc")
        ax_acc.plot(cvx_curve["epoch"], cvx_curve["val_acc"], marker="v", linewidth=2.2, label="CVX val_acc")
        ax_loss.plot(cvx_curve["epoch"], cvx_curve["train_loss"], marker="^", linewidth=2.2, label="CVX train_loss")
        ax_loss.plot(cvx_curve["epoch"], cvx_curve["val_loss"], marker="v", linewidth=2.2, label="CVX val_loss")
    ax_acc.set_title("Selected best training curves (accuracy)")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax_loss.set_title("Selected best training curves (loss)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _run_simple_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, list[float]]]]:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    num_classes = data["num_classes"]
    d_in = data["d_in"]
    cvx_device = _resolve_cvx_device(args.cvx_device)

    run_ste = args.simple_side in ("both", "ste_only")
    run_cvx = args.simple_side in ("both", "cvx_only")
    bias_grid = _resolve_simple_bias_grid(args)

    init_weights: List[np.ndarray] | None = None
    init_dir = args.init_weights_dir.strip() if getattr(args, "init_weights_dir", "") else ""
    if init_dir:
        man, w_list = load_finetune_weight_checkpoint(init_dir, args.init_weights_variant)
        assert_finetune_manifest_matches_runtime(
            man,
            readout_mode=args.last_layer_readout,
            P_in=d_in,
            P_rec=args.P_rec,
            P_last=args.P_last,
            L=args.L,
            num_classes=num_classes,
            K_parallel=int(args.K_parallel),
        )
        validate_finetune_weight_shapes_against_manifest(man, w_list)
        init_weights = [np.asarray(w, dtype=np.float32) for w in w_list]

    best_ste = None
    best_ste_params = None
    best_ste_curve: Dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if run_ste:
        best_ste_score = float("inf")
        for ste_lr in LR_GRID_DEFAULT:
            for ste_beta in BETA_GRID_DEFAULT:
                _set_seed(args.seed)
                out, _ = _capture_solver_stdout(
                    ste_solve,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=d_in,
                        num_classes=num_classes,
                        L=args.L,
                        P_rec=args.P_rec,
                        P_last=args.P_last,
                        K_parallel=int(args.K_parallel),
                        last_layer_readout=args.last_layer_readout,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name=args.loss_type,
                        optimizer_name=args.optimizer_name,
                        lr=float(ste_lr),
                        epochs=args.ste_epochs,
                        batch_size=None if args.batch_size == -1 else int(args.batch_size),
                        log_every=0,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                    pretrained_weights=init_weights,
                )
                score = float(out.best_losses["val_loss"]) + float(ste_beta)
                if score < best_ste_score:
                    best_ste_score = score
                    best_ste_params = {"lr": float(ste_lr), "beta": float(ste_beta)}
        if best_ste_params is None:
            raise RuntimeError("Simple mode failed to find STE candidate.")
        _set_seed(args.seed)
        best_ste, ste_log_text = _capture_solver_stdout(
            ste_solve,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            model_cfg=SteModelConfig(
                d_in=d_in,
                num_classes=num_classes,
                L=args.L,
                P_rec=args.P_rec,
                P_last=args.P_last,
                K_parallel=int(args.K_parallel),
                last_layer_readout=args.last_layer_readout,
            ),
            solve_cfg=SteSolveConfig(
                loss_name=args.loss_type,
                optimizer_name=args.optimizer_name,
                lr=float(best_ste_params["lr"]),
                epochs=args.ste_epochs,
                batch_size=None if args.batch_size == -1 else int(args.batch_size),
                weight_decay=0.0,
                beta_path_reg=float(best_ste_params["beta"]),
            ),
            pretrained_weights=init_weights,
        )
        print(ste_log_text, end="")
        best_ste_curve = _extract_ste_curve(ste_log_text)
        if best_ste is None:
            raise RuntimeError("Simple mode failed to rerun best STE candidate.")
        if not isinstance(best_ste.model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
            raise TypeError("Expected SNNBaselineSeq from ste_solve.")

    best_cvx = None
    best_cvx_params = None
    best_init = None
    best_cvx_curve: Dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_obj": [],
        "val_obj": [],
        "train_acc": [],
        "val_acc": [],
    }
    if run_cvx:
        best_cvx_score = float("inf")
        cvx_lr_eff = cvx_lr_sweep_values(args.cvx_method, LR_GRID_DEFAULT)
        for cvx_beta in BETA_GRID_DEFAULT:
            for cvx_lr in cvx_lr_eff:
                for cvx_bias in bias_grid:
                    _set_seed(args.seed)
                    if init_weights is not None:
                        init_cfg = InitializationConfig(
                            mode="pretraining",
                            seed=args.seed,
                            L=args.L,
                            P_rec=args.P_rec,
                            P_last=args.P_last,
                            K_parallel=int(args.K_parallel),
                            feature_count=args.P_last,
                            last_layer_readout=args.last_layer_readout,
                            bias=float(cvx_bias),
                            pretrained_weights=init_weights,
                        )
                    else:
                        init_cfg = InitializationConfig(
                            mode="gaussian",
                            seed=args.seed,
                            L=args.L,
                            P_rec=args.P_rec,
                            P_last=args.P_last,
                            K_parallel=int(args.K_parallel),
                            feature_count=args.P_last,
                            last_layer_readout=args.last_layer_readout,
                            bias=float(cvx_bias),
                        )
                    out, _ = _capture_solver_stdout(
                        cvx_solve,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        x_test=x_test,
                        y_test=y_test,
                        init_cfg=init_cfg,
                        solve_cfg=SolveConfig(
                            method=args.cvx_method,
                            loss_name=args.loss_type,
                            beta=float(cvx_beta),
                            lr=float(cvx_lr),
                            optimizer_name=args.optimizer_name,
                            epochs=args.cvx_epochs,
                            batch_size=None if args.batch_size == -1 else int(args.batch_size),
                            log_every=0,
                        ),
                        device=cvx_device,
                    )
                    score = float(out.final_losses.get("val_objective", out.final_losses["val_loss"]))
                    if score < best_cvx_score:
                        best_cvx_score = score
                        best_cvx_params = {
                            "lr": None if args.cvx_method == "cvx" else float(cvx_lr),
                            "beta": float(cvx_beta),
                            "bias": float(cvx_bias),
                        }
        if best_cvx_params is not None:
            _set_seed(args.seed)
            if init_weights is not None:
                best_init = InitializationConfig(
                    mode="pretraining",
                    seed=args.seed,
                    L=args.L,
                    P_rec=args.P_rec,
                    P_last=args.P_last,
                    K_parallel=int(args.K_parallel),
                    feature_count=args.P_last,
                    last_layer_readout=args.last_layer_readout,
                    bias=float(best_cvx_params["bias"]),
                    pretrained_weights=init_weights,
                )
            else:
                best_init = InitializationConfig(
                    mode="gaussian",
                    seed=args.seed,
                    L=args.L,
                    P_rec=args.P_rec,
                    P_last=args.P_last,
                    K_parallel=int(args.K_parallel),
                    feature_count=args.P_last,
                    last_layer_readout=args.last_layer_readout,
                    bias=float(best_cvx_params["bias"]),
                )
            best_cvx, cvx_log_text = _capture_solver_stdout(
                cvx_solve,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=best_init,
                solve_cfg=SolveConfig(
                    method=args.cvx_method,
                    loss_name=args.loss_type,
                    beta=float(best_cvx_params["beta"]),
                    lr=float(best_cvx_params["lr"]) if best_cvx_params["lr"] is not None else 0.0,
                    optimizer_name=args.optimizer_name,
                    epochs=args.cvx_epochs,
                    batch_size=None if args.batch_size == -1 else int(args.batch_size),
                ),
                device=cvx_device,
            )
            print(cvx_log_text, end="")
            best_cvx_curve = _extract_cvx_curve(cvx_log_text)
        if best_cvx is None or best_cvx_params is None or best_init is None:
            raise RuntimeError("Simple mode failed to find CVX candidate.")

    fixed_grids: Dict[str, Any] = {}
    if run_ste:
        fixed_grids["ste_lr_grid"] = list(LR_GRID_DEFAULT)
        fixed_grids["ste_beta_grid"] = list(BETA_GRID_DEFAULT)
    if run_cvx:
        fixed_grids["cvx_beta_grid"] = list(BETA_GRID_DEFAULT)
        fixed_grids["cvx_lr_grid"] = list(cvx_lr_sweep_values(args.cvx_method, LR_GRID_DEFAULT))
        fixed_grids["bias_grid"] = list(bias_grid)
        fixed_grids["cvx_lr_sweep_note"] = (
            "For method=cvx, lr is not swept (single placeholder 0.0); sweep is beta × bias only."
        )

    hp = _experiment_hyperparameters(args, data, search_grids=fixed_grids)
    metrics: Dict[str, Any] = {}
    arithmetic_mode = args.dataset == "arithmetic_seq"
    if run_ste and best_ste is not None and best_ste_params is not None:
        ste_losses = dict(best_ste.best_losses)
        metrics["ste"] = {
            "selected_params": best_ste_params,
            "best_losses": ste_losses,
        }
        if not arithmetic_mode:
            metrics["ste"]["test_last_step_acc"] = _ste_last_step_acc(best_ste.model, x_test, y_test)
        else:
            metrics["ste"]["reported_metrics"] = _arithmetic_reported_metrics(ste_losses)
        if args.simple_side in ("ste_only", "both"):
            metrics["ste"]["training_curve"] = best_ste_curve
    if run_cvx and best_cvx is not None and best_cvx_params is not None and best_init is not None:
        cvx_losses = dict(best_cvx.final_losses)
        cvx_json_key = _cvx_json_key_for_init_mode(best_init.mode)
        metrics[cvx_json_key] = {
            "selected_params": best_cvx_params,
            "final_losses": cvx_losses,
            "diagnostics": asdict(best_cvx.diagnostics),
        }
        if not arithmetic_mode:
            metrics[cvx_json_key]["test_last_step_acc"] = _cvx_last_step_acc(
                best_cvx,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=best_init,
            )
        else:
            metrics[cvx_json_key]["reported_metrics"] = _arithmetic_reported_metrics(cvx_losses)
        if args.simple_side in ("cvx_only", "both"):
            metrics[cvx_json_key]["training_curve"] = best_cvx_curve
    result = {"hyperparameters": hp, "metrics": metrics}
    return result, {"ste": best_ste_curve, "cvx": best_cvx_curve}


def _save_run_artifacts(
    args: argparse.Namespace,
    result: Dict[str, Any],
    curves: Dict[str, Dict[str, list[float]]],
) -> Dict[str, str]:
    """Persist run JSON under ``sweep_results/`` exactly like simple mode; plot/md only for ``--mode simple`` with ``--simple_side both``."""
    paths = _artifact_paths(args)
    flat = _flat_sweep_json_record(result)
    json_text = json.dumps(flat, indent=2, default=str)
    paths["json"].write_text(json_text + "\n")
    out_paths: Dict[str, str] = {"json": str(paths["json"])}
    if args.mode == "simple" and args.simple_side == "both":
        _save_simple_plot(paths["png"], curves["ste"], curves["cvx"])
        paths["md"].write_text(_make_simple_summary_md(args, result))
        out_paths["plot"] = str(paths["png"])
        out_paths["summary_md"] = str(paths["md"])
    return out_paths


def _run_fine_tune_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    bias_grid_default = _grid_to_list(BIAS_GRID_DEFAULT)
    cfg = FineTuneConfig(
        T=args.T,
        P_in=data["d_in"],
        P_rec=args.P_rec,
        P_last=args.P_last,
        L=args.L,
        loss_type=args.loss_type,
        epochs=args.cvx_epochs,
        batch_size=args.batch_size,
        cvx_optimizer=args.optimizer_name,
        cvx_method=args.cvx_method,
        ste_pretrain_epochs=args.ste_pretrain_epochs,
        ste_post_epochs=args.ste_post_epochs,
        last_layer_readout=args.last_layer_readout,
        readout_modes=(args.last_layer_readout,),
        # Fixed internal grids (not exposed as CLI args)
        beta_grid=BETA_GRID_DEFAULT,
        lr_grid=LR_GRID_DEFAULT,
        bias_grid=bias_grid_default,
        ste_beta_grid=BETA_GRID_DEFAULT,
        ste_lr_grid=LR_GRID_DEFAULT,
        weights_save_dir=_resolve_weights_save_dir(args),
        init_weights_dir=(args.init_weights_dir.strip() or None),
        init_weights_variant=args.init_weights_variant,
        K_parallel=int(args.K_parallel),
    )
    pipeline_out = run_fine_tune_pipeline(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_classes=data["num_classes"],
        seed=args.seed,
        cfg=cfg,
    )
    return _finetune_pipeline_to_simple_style_json(pipeline_out, args, data)


def _run_layer_wise_mode(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    bias_grid_default = _grid_to_list(BIAS_GRID_DEFAULT)
    cfg = LayerWiseConfig(
        num_blocks=args.num_blocks,
        L=args.L,
        P_rec=args.P_rec,
        P_last=args.P_last,
        ste_pretrain_epochs=args.ste_pretrain_epochs,
        ste_finetune_epochs=args.ste_post_epochs,
        loss_name=args.loss_type,
        optimizer_name=args.optimizer_name,
        cvx_method=args.cvx_method,
        # Fixed internal grids (not exposed as CLI args)
        cvx_beta_grid=BETA_GRID_DEFAULT,
        cvx_lr_grid=LR_GRID_DEFAULT,
        cvx_bias_grid=bias_grid_default,
        ste_beta_grid=BETA_GRID_DEFAULT,
        ste_lr_grid=LR_GRID_DEFAULT,
        K_parallel=int(args.K_parallel),
    )
    block_rows = run_layer_wise_stacking_test_bench(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_classes=data["num_classes"],
        cfg=cfg,
    )
    return _layer_wise_pipeline_to_simple_style_json(block_rows, args, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple atomic test runner: fine_tune, layer_wise, or gaussian-cvx-vs-ste baseline.",
    )
    parser.add_argument("--mode", choices=("fine_tune", "layer_wise", "simple"), default="simple")
    parser.add_argument(
        "--simple_side",
        choices=("both", "cvx_only", "ste_only"),
        default="both",
        help="Used only with --mode simple. Choose whether to run both baselines, only CVX, or only STE.",
    )
    parser.add_argument(
        "--snn_only",
        action="store_true",
        help="Shortcut for --mode simple --simple_side ste_only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist_seq",
        help=(
            "Sequence dataset id. Use ``dfa:<spec>`` for a specific DFA (e.g. ``dfa:first_last_xor``); "
            "plain ``dfa`` uses ``--dfa_spec`` (default tomita_3)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--P_rec", type=int, default=128)
    parser.add_argument("--P_last", type=int, default=128)
    parser.add_argument(
        "--K_parallel",
        type=int,
        default=1,
        help="Number of parallel SNN branches; 1 uses default solvers, >1 uses parallel CVX/STE backends.",
    )
    parser.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge_ovr")
    parser.add_argument("--optimizer_name", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--cvx_method", choices=("cvx", "sgd"), default="cvx")
    parser.add_argument(
        "--cvx_device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for CVX-SGD optimizer path in simple mode. Use cpu to avoid GPU memory pressure.",
    )
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--cvx_epochs", type=int, default=200)
    parser.add_argument("--ste_pretrain_epochs", type=int, default=80)
    parser.add_argument("--ste_post_epochs", type=int, default=100)
    parser.add_argument("--ste_epochs", type=int, default=200, help="Used only in --mode simple.")
    parser.add_argument("--num_blocks", type=int, default=2, help="Used only in --mode layer_wise.")
    parser.add_argument("--last_layer_readout", choices=("membrane", "spike"), default="membrane")
    parser.add_argument(
        "--bias_grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional CVX bias sweep values for --mode simple. If omitted, simple mode uses bias=0.0 only.",
    )
    parser.add_argument(
        "--weights_save_dir",
        type=str,
        default="finetune_weights",
        help=(
            "With --mode fine_tune, save STE pretrain SNN weights under this directory (per readout subfolder) "
            "immediately after pretrain (before CVX and STE post). "
            "Relative paths are resolved under atomic/ (default: finetune_weights). Pass empty to disable."
        ),
    )
    parser.add_argument(
        "--init_weights_dir",
        type=str,
        default="",
        help=(
            "Directory with finetune_manifest.json + pretrain_snn_weights.npz (or ste_post per --init_weights_variant). "
            "When set, STE and CVX use these SNN weights (simple + fine_tune); architecture must match CLI L, P_rec, P_last, readout."
        ),
    )
    parser.add_argument(
        "--init_weights_variant",
        type=str,
        choices=("pretrain", "ste_post"),
        default="pretrain",
        help="Which SNN npz inside init_weights_dir to load for initialization.",
    )

    # Arithmetic dataset options
    parser.add_argument("--arith_op", choices=("add", "sub", "mul", "div"), default="add")
    parser.add_argument("--arith_base", type=int, default=2)
    parser.add_argument("--n_digits", type=int, default=5)

    # DFA dataset options
    parser.add_argument("--dfa_spec", type=str, default="tomita_3")

    # UCI dataset options
    parser.add_argument("--uci_name", type=str, default="pima")
    parser.add_argument("--uci_test_size", type=float, default=0.2)
    parser.add_argument("--uci_val_size", type=float, default=0.2)
    parser.add_argument("--uci_no_standardize", action="store_true")

    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    if args.snn_only:
        if args.mode != "simple":
            raise ValueError("--snn_only is only valid with --mode simple.")
        args.simple_side = "ste_only"
    _apply_dataset_cli(args)
    return args


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    data = _load_dataset_from_args(args)

    curves: Dict[str, Dict[str, list[float]]] = {"ste": {}, "cvx": {}}
    if args.mode == "simple":
        result, curves = _run_simple_mode(args, data)
    elif args.mode == "fine_tune":
        result = _run_fine_tune_mode(args, data)
    elif args.mode == "layer_wise":
        result = _run_layer_wise_mode(args, data)
    else:
        raise ValueError(f"Unsupported mode={args.mode}")

    _save_run_artifacts(args, result, curves)

    flat = _flat_sweep_json_record(result)
    text = json.dumps(flat, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
