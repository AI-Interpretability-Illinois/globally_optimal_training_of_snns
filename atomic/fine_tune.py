from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve


@dataclass
class FineTuneConfig:
    # architecture
    T: int
    P_in: int
    P_rec: int
    P_last: int
    L: int

    # names/shape aligned to snn_generalized_pt2/snn_convex_fine_tune.py arguments
    loss_type: str = "hinge_ovr"
    epochs: int = 100
    batch_size: int = -1
    cvx_optimizer: str = "adam"
    cvx_method: str = "cvx"  # cvx | sgd
    cvx_step_size: int = 30
    cvx_gamma: float = 0.5
    beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    lr_grid: Sequence[float] = LR_GRID_DEFAULT
    bias_grid: Sequence[float] = BIAS_GRID_DEFAULT
    pretrain_lr: float = 1e-3
    pretrain_beta_path_reg: float = 0.0
    ste_lr_grid: Sequence[float] = LR_GRID_DEFAULT
    ste_beta_grid: Sequence[float] = BETA_GRID_DEFAULT
    ste_step_size: int = 30
    ste_gamma: float = 0.5
    learn_beta: bool = False
    learn_threshold: bool = False
    init_method: str = "pretrain"
    target_act_rate: float | None = None
    last_layer_readout: str = "membrane"
    readout_modes: Sequence[str] = ("membrane", "spike")
    beta_dist: str = "fixed"
    cvx_feature_source: str = "threshold_dict"
    compare_all_cvx_sources: bool = True
    cvx_sources: Sequence[str] = ("pretraining",)
    ste_pretrain_epochs: int = 80
    ste_post_epochs: int = 100


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_weight_list(model: SNNBaselineSeq) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    for fc in model.fcs:
        weights.append(fc.weight.detach().cpu().numpy().copy())
    weights.append(model.classifier.weight.detach().cpu().numpy().copy())
    return weights


def _last_step_acc(model: SNNBaselineSeq, x: np.ndarray, y: np.ndarray) -> float:
    device = next(model.parameters()).device
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t)
        preds = logits[:, -1, :].argmax(dim=1).cpu().numpy()
    if y.ndim == 2:
        y_last = y[:, -1]
    else:
        y_last = y
    return float(np.mean(preds == y_last))


def run_fine_tune_pipeline(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    seed: int,
    cfg: FineTuneConfig,
) -> Dict[str, object]:
    if x_train.ndim != 3 or x_val.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Expected x_train/x_val/x_test to be rank-3 arrays shaped (N, T, d_in).")
    if x_train.shape[1] != cfg.T:
        raise ValueError(f"T mismatch: cfg.T={cfg.T}, but x_train has T={x_train.shape[1]}.")
    if x_train.shape[2] != cfg.P_in:
        raise ValueError(f"P_in mismatch: cfg.P_in={cfg.P_in}, but x_train has d_in={x_train.shape[2]}.")

    readout_modes = tuple(dict.fromkeys(cfg.readout_modes))
    if len(readout_modes) == 0:
        raise ValueError("FineTuneConfig.readout_modes cannot be empty.")
    for mode in readout_modes:
        if mode not in ("membrane", "spike"):
            raise ValueError(f"Unsupported readout mode: {mode}. Expected membrane|spike.")

    by_readout: Dict[str, Dict[str, object]] = {}
    for readout_mode in readout_modes:
        _set_global_seed(seed)
        best_pre = None
        best_pre_val = float("inf")
        for ste_lr in cfg.ste_lr_grid:
            for ste_beta in cfg.ste_beta_grid:
                _set_global_seed(seed)
                out = ste_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=cfg.P_in,
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                        last_layer_readout=readout_mode,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name=cfg.loss_type,
                        optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                        lr=float(ste_lr),
                        epochs=cfg.ste_pretrain_epochs,
                        batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                        log_every=0,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                )
                val_loss = out.best_losses["val_loss"] + float(ste_beta)
                if val_loss < best_pre_val:
                    best_pre_val = val_loss
                    best_pre = out
        if best_pre is None:
            raise RuntimeError("Fine-tune pretraining failed to produce any candidate.")
        pretrained_model = best_pre.model
        if not isinstance(pretrained_model, SNNBaselineSeq):
            raise TypeError("Expected SNNBaselineSeq from ste_solve.")
        transferred_weights = _extract_weight_list(pretrained_model)
        pre_acc = {
            "train_last_step_acc": _last_step_acc(pretrained_model, x_train, y_train),
            "val_last_step_acc": _last_step_acc(pretrained_model, x_val, y_val),
            "test_last_step_acc": _last_step_acc(pretrained_model, x_test, y_test),
        }
        print(
            (
                f"[fine-tune][{readout_mode}] pretrain_acc "
                f"train={pre_acc['train_last_step_acc']:.4f} "
                f"val={pre_acc['val_last_step_acc']:.4f} "
                f"test={pre_acc['test_last_step_acc']:.4f}"
            ),
            flush=True,
        )

        cvx_sources = tuple(dict.fromkeys(cfg.cvx_sources))
        if len(cvx_sources) == 0:
            raise ValueError("FineTuneConfig.cvx_sources cannot be empty.")
        if any(source != "pretraining" for source in cvx_sources):
            raise ValueError(
                "Fine-tune mode requires CVX initialization from pretraining weights only. "
                "Set cvx_sources=('pretraining',)."
            )

        cvx_results: Dict[str, object] = {}
        cvx_selected_params: Dict[str, Dict[str, float]] = {}
        for source in cvx_sources:
            init_cfg = InitializationConfig(
                mode="pretraining",
                seed=seed,
                feature_count=int(cfg.P_last),
                pretrained_weights=transferred_weights,
                L=int(cfg.L),
                P_rec=int(cfg.P_rec),
                P_last=int(cfg.P_last),
                last_layer_readout=readout_mode,
            )
            best_cvx = None
            best_cvx_val = float("inf")
            best_beta = None
            best_lr = None
            best_bias = None
            for cvx_beta in cfg.beta_grid:
                for cvx_lr in cfg.lr_grid:
                    for cvx_bias in cfg.bias_grid:
                        init_cfg_trial = InitializationConfig(
                            mode=init_cfg.mode,
                            variant=init_cfg.variant,
                            seed=init_cfg.seed,
                            feature_count=init_cfg.feature_count,
                            bias=float(cvx_bias),
                            pretrained_weights=init_cfg.pretrained_weights,
                            L=init_cfg.L,
                            P_rec=init_cfg.P_rec,
                            P_last=init_cfg.P_last,
                            beta_leak=init_cfg.beta_leak,
                            threshold=init_cfg.threshold,
                            last_layer_readout=init_cfg.last_layer_readout,
                        )
                        out = cvx_solve(
                            x_train=x_train,
                            y_train=y_train,
                            x_val=x_val,
                            y_val=y_val,
                            x_test=x_test,
                            y_test=y_test,
                            init_cfg=init_cfg_trial,
                            solve_cfg=SolveConfig(
                                method=cfg.cvx_method,
                                loss_name=cfg.loss_type,
                                beta=float(cvx_beta),
                                lr=float(cvx_lr),
                                optimizer_name=cfg.cvx_optimizer,
                                epochs=cfg.epochs,
                                batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                                log_every=0,
                            ),
                        )
                        val_obj = out.final_losses.get("val_objective", out.final_losses["val_loss"])
                        if float(val_obj) < best_cvx_val:
                            best_cvx_val = float(val_obj)
                            best_cvx = out
                            best_beta = float(cvx_beta)
                            best_lr = float(cvx_lr)
                            best_bias = float(cvx_bias)
            if best_cvx is None or best_beta is None or best_lr is None or best_bias is None:
                raise RuntimeError(f"CVX sweep failed for source={source}.")
            # Rerun selected best CVX config with logging enabled so the selected curve is captured.
            init_cfg_best = InitializationConfig(
                mode=init_cfg.mode,
                variant=init_cfg.variant,
                seed=init_cfg.seed,
                feature_count=init_cfg.feature_count,
                bias=float(best_bias),
                pretrained_weights=init_cfg.pretrained_weights,
                L=init_cfg.L,
                P_rec=init_cfg.P_rec,
                P_last=init_cfg.P_last,
                beta_leak=init_cfg.beta_leak,
                threshold=init_cfg.threshold,
                last_layer_readout=init_cfg.last_layer_readout,
            )
            best_cvx = cvx_solve(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                init_cfg=init_cfg_best,
                solve_cfg=SolveConfig(
                    method=cfg.cvx_method,
                    loss_name=cfg.loss_type,
                    beta=float(best_beta),
                    lr=float(best_lr),
                    optimizer_name=cfg.cvx_optimizer,
                    epochs=cfg.epochs,
                    batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                    log_every=10,
                ),
            )
            cvx_results[source] = best_cvx
            cvx_selected_params[source] = {"beta": best_beta, "lr": best_lr, "bias": best_bias}

        best_post = None
        best_post_val = float("inf")
        best_post_lr = None
        best_post_beta = None
        for ste_lr in cfg.ste_lr_grid:
            for ste_beta in cfg.ste_beta_grid:
                _set_global_seed(seed)
                out = ste_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    model_cfg=SteModelConfig(
                        d_in=cfg.P_in,
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                        last_layer_readout=readout_mode,
                    ),
                    solve_cfg=SteSolveConfig(
                        loss_name=cfg.loss_type,
                        optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                        lr=float(ste_lr),
                        epochs=cfg.ste_post_epochs,
                        batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                        log_every=0,
                        weight_decay=0.0,
                        beta_path_reg=float(ste_beta),
                    ),
                    pretrained_weights=transferred_weights,
                )
                val_loss = out.best_losses["val_loss"] + float(ste_beta)
                if val_loss < best_post_val:
                    best_post_val = val_loss
                    best_post = out
                    best_post_lr = float(ste_lr)
                    best_post_beta = float(ste_beta)
        if best_post is None or best_post_lr is None or best_post_beta is None:
            raise RuntimeError("Fine-tune STE post-training sweep failed.")
        # Rerun selected best STE post config with logging enabled so selected curve is captured.
        _set_global_seed(seed)
        best_post = ste_solve(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            model_cfg=SteModelConfig(
                d_in=cfg.P_in,
                num_classes=num_classes,
                L=cfg.L,
                P_rec=cfg.P_rec,
                P_last=cfg.P_last,
                last_layer_readout=readout_mode,
            ),
            solve_cfg=SteSolveConfig(
                loss_name=cfg.loss_type,
                optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                lr=float(best_post_lr),
                epochs=cfg.ste_post_epochs,
                batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                log_every=10,
                weight_decay=0.0,
                beta_path_reg=float(best_post_beta),
            ),
            pretrained_weights=transferred_weights,
        )
        by_readout[readout_mode] = {
            "ste_pretrain": best_pre,
            "pretrain_accuracy": pre_acc,
            "cvx_by_source": cvx_results,
            "cvx_selected_params": cvx_selected_params,
            "ste_post": best_post,
            "ste_post_selected_params": {"lr": best_post_lr, "beta": best_post_beta},
            "ste_post_loss_curve": [float(v) for v in best_post.loss_history],
            "cvx_loss_curves": {name: [float(v) for v in cvx_out.loss_history] for name, cvx_out in cvx_results.items()},
        }

    selected_mode = cfg.last_layer_readout if cfg.last_layer_readout in by_readout else readout_modes[0]
    selected = by_readout[selected_mode]
    json_report: Dict[str, object] = {
        "selected_readout": selected_mode,
        "by_readout": {
            mode: {
                "pretrain_accuracy": readout_data["pretrain_accuracy"],
                "ste_post_selected_params": readout_data["ste_post_selected_params"],
                "ste_post_best_losses": dict(readout_data["ste_post"].best_losses),
                "ste_post_loss_curve": list(readout_data["ste_post_loss_curve"]),
                "cvx_selected_params": dict(readout_data["cvx_selected_params"]),
                "cvx_final_losses": {
                    src: dict(cvx_out.final_losses) for src, cvx_out in readout_data["cvx_by_source"].items()
                },
                "cvx_loss_curves": {
                    src: list(curve) for src, curve in readout_data["cvx_loss_curves"].items()
                },
            }
            for mode, readout_data in by_readout.items()
        },
    }

    return {
        "seed": seed,
        "cfg": cfg,
        "selected_readout": selected_mode,
        "by_readout": by_readout,
        "json_report": json_report,
        # Backward-compatible aliases (selected readout)
        "ste_pretrain": selected["ste_pretrain"],
        "cvx_by_source": selected["cvx_by_source"],
        "ste_post": selected["ste_post"],
    }
