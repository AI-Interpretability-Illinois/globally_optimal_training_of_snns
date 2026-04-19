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
    from fine_tune import (
        FineTuneConfig,
        assert_finetune_manifest_matches_runtime,
        load_finetune_weight_checkpoint,
        run_fine_tune_pipeline,
        validate_finetune_weight_shapes_against_manifest,
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
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .data_loaders.arithmetic_data_loader import ArithmeticDataset, load_arithmetic_dataset
    from .data_loaders.dfa_data_loader import make_dfa_dataset
    from .data_loaders.image_data_loader import ImageSequenceDataset, load_cifar_seq_dataset, load_mnist_seq_dataset
    from .data_loaders.uci_data_loader import UciDataset, load_uci_dataset
    from .fine_tune import (
        FineTuneConfig,
        assert_finetune_manifest_matches_runtime,
        load_finetune_weight_checkpoint,
        run_fine_tune_pipeline,
        validate_finetune_weight_shapes_against_manifest,
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
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


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


def _ste_last_step_acc(model: SNNBaselineSeq, x_test: np.ndarray, y_test: np.ndarray) -> float:
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
    _, _, d_test, _ = _build_feature_map(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        init_cfg=init_cfg,
    )
    y_last = _prepare_sequence_targets(y_test)
    if isinstance(cvx_result.trained_model, dict) and "weights" in cvx_result.trained_model:
        w = cvx_result.trained_model["weights"]
        scores = d_test @ w
        preds = scores.argmax(axis=1)
        return float(np.mean(preds == y_last))
    model = cvx_result.trained_model
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(d_test, dtype=torch.float32))
        preds = logits.argmax(dim=1).cpu().numpy()
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
    ste = result.get("ste")
    if ste is not None:
        ste_p = ste["selected_params"]
        ste_l = ste["best_losses"]
        lines.append(
            (
                f"| STE | {ste_p['lr']:.6g} | {ste_p['beta']:.6g} | - | "
                f"{ste_l['train_loss']:.6f} | {ste_l['val_loss']:.6f} | {ste_l['test_loss']:.6f} | "
                f"{float(ste['test_last_step_acc']):.4f} |"
            )
        )
    cvx = result.get("cvx_gaussian")
    if cvx is not None:
        cvx_p = cvx["selected_params"]
        cvx_l = cvx["final_losses"]
        lr_cell = "-" if cvx_p.get("lr") is None else f"{float(cvx_p['lr']):.6g}"
        lines.append(
            (
                f"| CVX (gaussian, {args.cvx_method}) | {lr_cell} | {cvx_p['beta']:.6g} | {cvx_p['bias']:.6g} | "
                f"{cvx_l['train_loss']:.6f} | {cvx_l['val_loss']:.6f} | {cvx_l['test_loss']:.6f} | "
                f"{float(cvx['test_last_step_acc']):.4f} |"
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
    init_ckpt_meta: Dict[str, Any] | None = None
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
        )
        validate_finetune_weight_shapes_against_manifest(man, w_list)
        init_weights = [np.asarray(w, dtype=np.float32) for w in w_list]
        init_ckpt_meta = {"dir": str(Path(init_dir).resolve()), "variant": args.init_weights_variant, "manifest": man}

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
        if not isinstance(best_ste.model, SNNBaselineSeq):
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

    result: Dict[str, Any] = {
        "mode": "simple",
        "simple_side": args.simple_side,
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(cvx_lr_sweep_values(args.cvx_method, LR_GRID_DEFAULT)),
            "bias_grid": list(bias_grid),
            "cvx_lr_sweep_note": "For method=cvx, lr is not swept (single placeholder 0.0); sweep is beta × bias only.",
        },
    }
    if init_ckpt_meta is not None:
        result["init_checkpoint"] = init_ckpt_meta
    arithmetic_mode = args.dataset == "arithmetic_seq"
    if run_ste and best_ste is not None and best_ste_params is not None:
        ste_losses = dict(best_ste.best_losses)
        result["ste"] = {
            "selected_params": best_ste_params,
            "best_losses": ste_losses,
        }
        if not arithmetic_mode:
            result["ste"]["test_last_step_acc"] = _ste_last_step_acc(best_ste.model, x_test, y_test)
        else:
            result["ste"]["reported_metrics"] = _arithmetic_reported_metrics(ste_losses)
        if args.simple_side == "ste_only":
            result["ste"]["training_curve"] = best_ste_curve
    if run_cvx and best_cvx is not None and best_cvx_params is not None and best_init is not None:
        cvx_losses = dict(best_cvx.final_losses)
        result["cvx_gaussian"] = {
            "selected_params": best_cvx_params,
            "final_losses": cvx_losses,
            "diagnostics": asdict(best_cvx.diagnostics),
        }
        if not arithmetic_mode:
            result["cvx_gaussian"]["test_last_step_acc"] = _cvx_last_step_acc(
                best_cvx,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=best_init,
            )
        else:
            result["cvx_gaussian"]["reported_metrics"] = _arithmetic_reported_metrics(cvx_losses)
        if args.simple_side == "cvx_only":
            result["cvx_gaussian"]["training_curve"] = best_cvx_curve
    return result, {"ste": best_ste_curve, "cvx": best_cvx_curve}


def _save_simple_artifacts(args: argparse.Namespace, result: Dict[str, Any], curves: Dict[str, Dict[str, list[float]]]) -> Dict[str, str]:
    paths = _artifact_paths(args)
    json_text = json.dumps(result, indent=2, default=str)
    paths["json"].write_text(json_text + "\n")
    out_paths: Dict[str, str] = {"json": str(paths["json"])}
    if args.simple_side == "both":
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
    )
    out = run_fine_tune_pipeline(
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
    return {
        "mode": "fine_tune",
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(LR_GRID_DEFAULT),
            "bias_grid": bias_grid_default,
        },
        "result": out,
    }


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
    )
    out = run_layer_wise_stacking_test_bench(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        num_classes=data["num_classes"],
        cfg=cfg,
    )
    return {
        "mode": "layer_wise",
        "dataset": args.dataset,
        "fixed_grids": {
            "beta_grid": list(BETA_GRID_DEFAULT),
            "lr_grid": list(LR_GRID_DEFAULT),
            "bias_grid": bias_grid_default,
        },
        "result": out,
    }


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
        choices=("mnist_seq", "mnist_perm_seq", "cifar_seq", "arithmetic_seq", "dfa", "uci"),
        default="mnist_seq",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--P_rec", type=int, default=128)
    parser.add_argument("--P_last", type=int, default=128)
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
            "With --mode fine_tune, save weights under this directory (per readout subfolder). "
            "Relative paths are resolved under atomic/ (default: finetune_weights -> atomic/finetune_weights). "
            "Pass empty to disable saving."
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

    if args.mode == "simple":
        result["artifact_paths"] = _save_simple_artifacts(args, result, curves)

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
