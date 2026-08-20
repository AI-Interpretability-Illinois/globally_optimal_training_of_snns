"""
LSM baseline sweeps on XOR (DFA ``first_last_xor``), MNIST-as-sequence, and
base-b addition (``add_b2`` / ``add_b3`` / ``add_b5`` / ``add_b7`` / ``add_b10``).

Task presets mirror the (dataset, T, L, P_rec, P_last, K_parallel, n_train, n_val, n_test)
conventions of :mod:`run_dfa_xor_tl_sweep_simple_and_ste_parallel` and
:mod:`run_mnist_t_sweep_simple_and_ste_parallel`, with the surrogate-gradient STE path
replaced by the criticality-tuned LSM baseline in
:mod:`atomic.solvers.LSM` + :mod:`atomic.solvers.lsm_criticality`.

Pipeline (the "R" condition)
----------------------------
1. **Criticality tuning** — sweep ``(beta_leak, threshold, input_scale)`` and pick the
   combination whose measured branching ratio sigma is closest to 1
   (Bertschinger-Natschläger / Legenstein-Maass / Wilting-Priesemann). This gives a
   principled operating-point selection for the spiking reservoir, matching the SNN-
   reservoir literature's criticality criterion.
2. **Closed-form ridge readout** — sweep ``ridge_lambdas`` on the tuned reservoir's
   features; select by validation MSE. This is the canonical RC readout
   (Lukoševičius & Jaeger 2009), replacing Adam training.

R-CVX
-----
Same criticality-tuned reservoir, different readout objective: extract the
frozen ``W_in`` list from the tuned model (the input-scale multiplier is folded
into the exported weights by :func:`solvers.lsm_criticality._scaled_model`) and
feed it into ``cvx_solve(InitializationConfig(mode='pretraining', ...))``. CVX
recomputes its own thresholded feature map from the same weights and solves the
convex L1-hinge / CE readout; sweeps beta and bias.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from cvx_side_ckpt import ckpt_path, load_weight_list, require_ckpt_task, save_weight_list
    from data_loaders.arithmetic_data_loader import SUPPORTED_BASES, load_arithmetic_dataset
    from data_loaders.dfa_data_loader import make_dfa_dataset
    from data_loaders.image_data_loader import ImageSequenceDataset, load_mnist_seq_dataset
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.LSM import LSMBaselineSeq, LSMModelConfig, _apply_pretrained_weights, extract_lsm_weight_list
    from solvers.lsm_criticality import (
        CriticalityGrid,
        _reservoir_features,
        build_tuned_model,
        lsm_ridge_solve,
        tune_reservoir_criticality,
    )
else:
    from .cvx_side_ckpt import ckpt_path, load_weight_list, require_ckpt_task, save_weight_list
    from .data_loaders.arithmetic_data_loader import SUPPORTED_BASES, load_arithmetic_dataset
    from .data_loaders.dfa_data_loader import make_dfa_dataset
    from .data_loaders.image_data_loader import ImageSequenceDataset, load_mnist_seq_dataset
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.LSM import LSMBaselineSeq, LSMModelConfig, _apply_pretrained_weights, extract_lsm_weight_list
    from .solvers.lsm_criticality import (
        CriticalityGrid,
        _reservoir_features,
        build_tuned_model,
        lsm_ridge_solve,
        tune_reservoir_criticality,
    )


# ---------------------------------------------------------------------------#
# Task presets — chosen to match the existing STE/CVX sweep scripts on the same tasks.
# ---------------------------------------------------------------------------#


@dataclass
class TaskPreset:
    name: str
    dataset: str
    T_list: Tuple[int, ...]
    L_list: Tuple[int, ...]
    K_parallel_list: Tuple[int, ...]
    P_rec: int
    P_last: int
    n_train: int
    n_val: int
    n_test: int
    loss_name: str
    last_layer_readout: str
    dfa_spec: str | None = None
    arith_op: str | None = None
    arith_base: int | None = None
    n_digits: int | None = None
    reservoir_variant: str = "normalized"


XOR_PRESET = TaskPreset(
    name="xor",
    dataset="dfa",
    dfa_spec="first_last_xor",
    T_list=(6, 8, 11, 14),
    L_list=(5, 15),
    K_parallel_list=(2,),
    P_rec=500,
    P_last=1000,
    n_train=2000,
    n_val=2000,
    n_test=4000,
    loss_name="hinge",
    last_layer_readout="membrane",
    reservoir_variant="normalized",
)


MNIST_PRESET = TaskPreset(
    name="mnist",
    dataset="mnist_seq",
    T_list=(4,),
    L_list=(3, 10),
    K_parallel_list=(2,),
    P_rec=512,
    P_last=1024,
    n_train=6000,
    n_val=2000,
    n_test=2000,
    loss_name="ce",
    last_layer_readout="membrane",
    reservoir_variant="normalized",
)


def _add_timesteps(n_digits: int) -> int:
    return int(n_digits) + 1


def _make_add_preset(base: int, *, n_digits: int = 5) -> TaskPreset:
    if int(base) not in SUPPORTED_BASES:
        raise ValueError(f"Unsupported addition base={base}. Supported: {SUPPORTED_BASES}.")
    T = _add_timesteps(int(n_digits))
    return TaskPreset(
        name=f"add_b{int(base)}",
        dataset="arithmetic_seq",
        T_list=(T,),
        L_list=(3, 5),
        K_parallel_list=(2,),
        P_rec=500,
        P_last=1000,
        n_train=2000,
        n_val=2000,
        n_test=4000,
        loss_name="ce",
        last_layer_readout="spike",
        arith_op="add",
        arith_base=int(base),
        n_digits=int(n_digits),
        reservoir_variant="normalized",
    )


ADD_PRESETS: Dict[str, TaskPreset] = {f"add_b{b}": _make_add_preset(b) for b in SUPPORTED_BASES}

TASK_PRESETS: Dict[str, TaskPreset] = {"xor": XOR_PRESET, "mnist": MNIST_PRESET, **ADD_PRESETS}
TASK_NAMES: Tuple[str, ...] = tuple(TASK_PRESETS.keys())


# ---------------------------------------------------------------------------#
# Data loading
# ---------------------------------------------------------------------------#


def _load_task_data(preset: TaskPreset, *, T: int, seed: int) -> Dict[str, Any]:
    if preset.dataset == "dfa":
        if preset.dfa_spec is None:
            raise ValueError(f"Task {preset.name}: dfa_spec must be set for dataset='dfa'.")
        n_total = preset.n_train + preset.n_val + preset.n_test
        x_all, y_all, num_classes = make_dfa_dataset(
            dfa_spec=preset.dfa_spec, n=n_total, T=T, seed=seed, balanced=True
        )
        x_tr = x_all[: preset.n_train]
        y_tr = y_all[: preset.n_train]
        x_va = x_all[preset.n_train : preset.n_train + preset.n_val]
        y_va = y_all[preset.n_train : preset.n_train + preset.n_val]
        x_te = x_all[preset.n_train + preset.n_val :]
        y_te = y_all[preset.n_train + preset.n_val :]
        return {
            "x_train": x_tr,
            "y_train": y_tr,
            "x_val": x_va,
            "y_val": y_va,
            "x_test": x_te,
            "y_test": y_te,
            "num_classes": int(num_classes),
            "d_in": int(x_tr.shape[2]),
        }
    if preset.dataset in ("mnist_seq", "mnist_perm_seq"):
        ds: ImageSequenceDataset = load_mnist_seq_dataset(
            task=preset.dataset,
            T=T,
            n_train=preset.n_train,
            n_val=preset.n_val,
            n_test=preset.n_test,
            seed=seed,
        )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": int(ds.num_classes),
            "d_in": int(ds.d_in),
        }
    if preset.dataset == "arithmetic_seq":
        if preset.arith_op is None or preset.arith_base is None or preset.n_digits is None:
            raise ValueError(
                f"Task {preset.name}: arith_op/arith_base/n_digits must be set for dataset='arithmetic_seq'."
            )
        ds = load_arithmetic_dataset(
            op=str(preset.arith_op),
            base=int(preset.arith_base),
            n_digits=int(preset.n_digits),
            n_train=int(preset.n_train),
            n_val=int(preset.n_val),
            n_test=int(preset.n_test),
            seed=int(seed),
        )
        data_T = int(ds.X_train.shape[1])
        if data_T != int(T):
            raise ValueError(
                f"Task {preset.name}: arithmetic sequence length {data_T} != requested T={T} "
                f"(op={preset.arith_op} n_digits={preset.n_digits})."
            )
        return {
            "x_train": ds.X_train,
            "y_train": ds.y_train,
            "x_val": ds.X_val,
            "y_val": ds.y_val,
            "x_test": ds.X_test,
            "y_test": ds.y_test,
            "num_classes": int(ds.num_classes),
            "d_in": int(ds.X_train.shape[2]),
        }
    raise ValueError(f"Unsupported dataset={preset.dataset!r} for task={preset.name}.")


# ---------------------------------------------------------------------------#
# Sweep / driver
# ---------------------------------------------------------------------------#


@dataclass
class SweepGrids:
    """Grids for criticality tuning, ridge readout, and R-CVX."""

    beta_leak_grid: Tuple[float, ...] = (0.90, 0.95, 0.99, 0.995)
    threshold_grid: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    input_scale_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    ridge_lambdas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    cvx_beta_grid: Tuple[float, ...] = (1e-2, 1e-1, 1.0)
    cvx_bias_grid: Tuple[float, ...] = (0.0,)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cvx_split_accs(
    tuned_model: LSMBaselineSeq,
    cvx_weights: np.ndarray,
    *,
    data: Dict[str, Any],
    bias: float,
) -> Dict[str, float]:
    """Rebuild CVX's prediction path from the tuned LSM to report train/val/test accuracy.

    CVX operates on the *thresholded* reservoir features (``1[mem - bias >= 0]``
    for membrane readout, or raw spikes for spike readout) and returns
    ``(P_last, num_classes)`` weights. To turn its output back into accuracy we:

    1. Run the tuned LSM on the split to get raw last-layer readouts (N, T, P_last).
    2. Apply the same thresholding CVX applied internally (uses ``bias``, the
       CVX-selected bias grid value).
    3. For last-step supervision (rank-1 labels) take the terminal timestep,
       multiply by ``cvx_weights``, argmax, compare to y.
    4. For all-timesteps supervision (rank-2 labels) flatten (N, T, P_last) to
       (N*T, P_last) and compute token-level accuracy the same way. Sequence
       accuracy = fraction of samples where every token matches.
    """
    device = next(tuned_model.parameters()).device
    W = np.asarray(cvx_weights, dtype=np.float64)

    def _acc_for_split(x_np: np.ndarray, y_np: np.ndarray) -> float:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            feats = _reservoir_features(tuned_model, x_t)  # (N, T, P_last)
        if tuned_model.last_layer_readout == "membrane":
            d = (feats - float(bias) >= 0).float().cpu().numpy()
        elif tuned_model.last_layer_readout == "spike":
            d = feats.cpu().numpy()
        else:
            raise ValueError(
                f"Unsupported last_layer_readout={tuned_model.last_layer_readout!r} for CVX-style accuracy."
            )
        if y_np.ndim == 1:
            d_last = d[:, -1, :]  # (N, P_last)
            preds = np.argmax(d_last @ W, axis=1)
            return float(np.mean(preds == y_np))
        if y_np.ndim == 2:
            n, t = int(d.shape[0]), int(d.shape[1])
            d_flat = d.reshape(n * t, d.shape[2])
            preds_flat = np.argmax(d_flat @ W, axis=1).reshape(n, t)
            return float(np.mean(preds_flat == y_np))
        raise ValueError(f"Expected rank-1 or rank-2 labels, got rank {y_np.ndim}.")

    return {
        "train_acc": _acc_for_split(data["x_train"], data["y_train"]),
        "val_acc": _acc_for_split(data["x_val"], data["y_val"]),
        "test_acc": _acc_for_split(data["x_test"], data["y_test"]),
    }


def _r_cvx_from_tuned_lsm(
    tuned_model: LSMBaselineSeq,
    *,
    data: Dict[str, Any],
    grids: SweepGrids,
    L: int,
    K_parallel: int,
    P_rec: int,
    P_last: int,
    reservoir_seed: int,
    loss_name: str,
    cvx_method: str,
    cvx_ovr_workers: int,
    compute_ce_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
) -> Dict[str, Any]:
    """R-CVX: sweep CVX (beta, bias) on the same criticality-tuned reservoir.

    The tuned model already has the input-scale multiplier folded into
    ``fc.weight`` (courtesy of :func:`_scaled_model`), so exporting the weight
    list gives CVX the exact same W_in the ridge readout saw. CVX rebuilds its
    own thresholded feature map from these weights and solves the convex readout.
    Model-level LIF knobs (``beta_leak``, ``threshold``) are threaded through
    :class:`InitializationConfig` so CVX's LIF stack matches the tuned reservoir.
    """
    pretrained = extract_lsm_weight_list(tuned_model)
    tuned_cfg = tuned_model.cfg  # tuned beta_leak and threshold

    best_r_cvx = None
    best_val_acc = -1.0
    best_params: Dict[str, float] = {}
    best_split_accs: Dict[str, float] = {}
    for bias in grids.cvx_bias_grid:
        for beta in grids.cvx_beta_grid:
            init_cfg = InitializationConfig(
                mode="pretraining",
                seed=int(reservoir_seed),
                feature_count=int(P_last),
                bias=float(bias),
                pretrained_weights=pretrained,
                L=int(L),
                P_rec=int(P_rec),
                P_last=int(P_last),
                K_parallel=int(K_parallel),
                beta_leak=float(tuned_cfg.beta_leak),
                threshold=float(tuned_cfg.threshold),
                last_layer_readout=str(tuned_cfg.last_layer_readout),
            )
            out = cvx_solve(
                x_train=data["x_train"], y_train=data["y_train"],
                x_val=data["x_val"], y_val=data["y_val"],
                x_test=data["x_test"], y_test=data["y_test"],
                init_cfg=init_cfg,
                solve_cfg=SolveConfig(
                    method=cvx_method,
                    loss_name=loss_name,
                    beta=float(beta),
                    lr=0.0,
                    optimizer_name="adam",
                    epochs=1,
                    batch_size=None,
                    log_every=0,
                    compute_ce_dual=bool(compute_ce_dual),
                    cvx_ovr_workers=int(cvx_ovr_workers),
                    lite_max_iter=int(lite_max_iter),
                    lite_tol=float(lite_tol),
                ),
            )
            cvx_weights = out.trained_model["weights"]  # (P_last, num_classes)
            split_accs = _cvx_split_accs(tuned_model, cvx_weights, data=data, bias=float(bias))
            if split_accs["val_acc"] > best_val_acc:
                best_val_acc = float(split_accs["val_acc"])
                best_r_cvx = out
                best_params = {"beta": float(beta), "bias": float(bias)}
                best_split_accs = split_accs
    if best_r_cvx is None:
        raise RuntimeError("R-CVX sweep produced no candidates.")
    return {
        "selected_params": best_params,
        "split_accs": {k: float(v) for k, v in best_split_accs.items()},
        "final_losses": {k: float(v) for k, v in best_r_cvx.final_losses.items()},
        "diagnostics": {
            "primal_value": float(best_r_cvx.diagnostics.primal_value),
            "dual_value": float(best_r_cvx.diagnostics.dual_value),
            "gap": float(best_r_cvx.diagnostics.gap),
        },
    }


def _run_R_and_R_CVX_single(
    *,
    preset: TaskPreset,
    T: int,
    L: int,
    K_parallel: int,
    seed: int,
    data: Dict[str, Any],
    grids: SweepGrids,
    reservoir_seed_base: int,
    run_ridge: bool,
    run_r_cvx: bool,
    probe_subsample: int,
    cvx_method: str,
    cvx_ovr_workers: int,
    ckpt_dir: Path,
    compute_ce_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
) -> Dict[str, Any]:
    """R pipeline: criticality tune -> ridge readout, then optional R-CVX.

    ``run_ridge=False`` loads the criticality-tuned reservoir from ``ckpt_dir``
    (CVX-side machine). ``run_ridge=True`` writes that checkpoint so a later
    ``--side cvx`` job can skip the criticality sweep.
    """
    d_in = int(data["d_in"])
    num_classes = int(data["num_classes"])
    device = None
    lsm_ckpt = ckpt_path(ckpt_dir, seed=seed, T=T, L=L, K=K_parallel, tag="lsm", task=preset.name)

    if run_ridge:
        base_cfg = LSMModelConfig(
            d_in=d_in,
            num_classes=num_classes,
            L=L,
            P_rec=preset.P_rec,
            P_last=preset.P_last,
            K_parallel=K_parallel,
            last_layer_readout=preset.last_layer_readout,
            reservoir_seed=int(reservoir_seed_base),
            reservoir_variant=str(preset.reservoir_variant),
        )
        probe = data["x_train"]
        if probe_subsample > 0 and probe.shape[0] > probe_subsample:
            idx = np.random.default_rng(int(seed)).choice(
                probe.shape[0], size=int(probe_subsample), replace=False
            )
            probe = probe[idx]
        crit = tune_reservoir_criticality(
            base_cfg,
            probe,
            grid=CriticalityGrid(
                beta_leak=grids.beta_leak_grid,
                threshold=grids.threshold_grid,
                input_scale=grids.input_scale_grid,
            ),
            device=device,
            verbose=False,
        )
        tuned_model = build_tuned_model(crit, device=device)
        ridge_result = lsm_ridge_solve(
            x_train=data["x_train"], y_train=data["y_train"],
            x_val=data["x_val"], y_val=data["y_val"],
            x_test=data["x_test"], y_test=data["y_test"],
            model=tuned_model,
            num_classes=num_classes,
            ridge_lambdas=grids.ridge_lambdas,
            device=device,
        )
        print(
            (
                f"[lsm-xor-mnist] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
                f"CRIT beta={crit.tuned_config.beta_leak:.3f} thr={crit.tuned_config.threshold:.2f} "
                f"in_scale={crit.input_scale:.2f} sigma={crit.branching_ratio:.4f} | "
                f"RIDGE lambda={ridge_result.best_losses['ridge_lambda']:.4g} "
                f"train_acc={ridge_result.best_losses['train_acc']:.4f} "
                f"val_acc={ridge_result.best_losses['val_acc']:.4f} "
                f"test_acc={ridge_result.best_losses['test_acc']:.4f}"
            ),
            flush=True,
        )
        crit_block = {
            "beta_leak": float(crit.tuned_config.beta_leak),
            "threshold": float(crit.tuned_config.threshold),
            "input_scale": float(crit.input_scale),
            "branching_ratio": float(crit.branching_ratio),
            "table": crit.table,
        }
        ridge_block = {"best_losses": {k: float(v) for k, v in ridge_result.best_losses.items()}}
        save_weight_list(
            lsm_ckpt,
            extract_lsm_weight_list(tuned_model),
            meta={
                "tag": "lsm",
                "task": preset.name,
                "dataset": preset.dataset,
                "arith_op": preset.arith_op,
                "arith_base": preset.arith_base,
                "n_digits": preset.n_digits,
                "d_in": int(d_in),
                "num_classes": int(num_classes),
                "T": int(T),
                "L": int(L),
                "K_parallel": int(K_parallel),
                "P_rec": int(preset.P_rec),
                "P_last": int(preset.P_last),
                "last_layer_readout": str(preset.last_layer_readout),
                "reservoir_seed": int(reservoir_seed_base),
                "reservoir_variant": str(preset.reservoir_variant),
                "beta_leak": float(crit.tuned_config.beta_leak),
                "threshold": float(crit.tuned_config.threshold),
                "input_scale": float(crit.input_scale),
                "branching_ratio": float(crit.branching_ratio),
                "ridge": ridge_block,
            },
        )
    else:
        weights, meta = load_weight_list(lsm_ckpt)
        require_ckpt_task(meta, expected_task=preset.name, path=lsm_ckpt)
        tuned_cfg = LSMModelConfig(
            d_in=int(meta["d_in"]),
            num_classes=int(meta["num_classes"]),
            L=int(meta["L"]),
            P_rec=int(meta["P_rec"]),
            P_last=int(meta["P_last"]),
            K_parallel=int(meta["K_parallel"]),
            beta_leak=float(meta["beta_leak"]),
            threshold=float(meta["threshold"]),
            last_layer_readout=str(meta["last_layer_readout"]),
            reservoir_seed=int(meta["reservoir_seed"]),
            reservoir_variant=str(meta["reservoir_variant"]),
        )
        tuned_model = LSMBaselineSeq(tuned_cfg)
        _apply_pretrained_weights(tuned_model, weights)
        crit_block = {
            "beta_leak": float(meta["beta_leak"]),
            "threshold": float(meta["threshold"]),
            "input_scale": float(meta["input_scale"]),
            "branching_ratio": float(meta["branching_ratio"]),
            "table": None,
            "loaded_from": str(lsm_ckpt),
        }
        ridge_block = meta["ridge"]
        print(
            (
                f"[lsm-xor-mnist] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
                f"loaded LSM ckpt {lsm_ckpt} "
                f"CRIT beta={crit_block['beta_leak']:.3f} thr={crit_block['threshold']:.2f} "
                f"in_scale={crit_block['input_scale']:.2f} sigma={crit_block['branching_ratio']:.4f}"
            ),
            flush=True,
        )

    r_cvx_block: Dict[str, Any] | None = None
    if run_r_cvx:
        r_cvx_block = _r_cvx_from_tuned_lsm(
            tuned_model,
            data=data,
            grids=grids,
            L=int(L),
            K_parallel=int(K_parallel),
            P_rec=int(preset.P_rec),
            P_last=int(preset.P_last),
            reservoir_seed=int(reservoir_seed_base),
            loss_name=preset.loss_name,
            cvx_method=cvx_method,
            cvx_ovr_workers=cvx_ovr_workers,
            compute_ce_dual=compute_ce_dual,
            lite_max_iter=lite_max_iter,
            lite_tol=lite_tol,
        )
        print(
            (
                f"[lsm-xor-mnist] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
                f"R-CVX best_params={r_cvx_block['selected_params']} "
                f"train_acc={r_cvx_block['split_accs']['train_acc']:.4f} "
                f"val_acc={r_cvx_block['split_accs']['val_acc']:.4f} "
                f"test_acc={r_cvx_block['split_accs']['test_acc']:.4f}"
            ),
            flush=True,
        )

    return {
        "task": preset.name,
        "dataset": preset.dataset,
        "dfa_spec": preset.dfa_spec,
        "arith_op": preset.arith_op,
        "arith_base": preset.arith_base,
        "n_digits": preset.n_digits,
        "T": int(T),
        "L": int(L),
        "K_parallel": int(K_parallel),
        "P_rec": int(preset.P_rec),
        "P_last": int(preset.P_last),
        "n_train": int(preset.n_train),
        "n_val": int(preset.n_val),
        "n_test": int(preset.n_test),
        "loss_name": preset.loss_name,
        "last_layer_readout": preset.last_layer_readout,
        "reservoir_variant": preset.reservoir_variant,
        "seed": int(seed),
        "reservoir_seed": int(reservoir_seed_base),
        "criticality": crit_block,
        "ridge": ridge_block,
        "r_cvx": r_cvx_block,
    }


# ---------------------------------------------------------------------------#
# CLI
# ---------------------------------------------------------------------------#


def _apply_debug(preset: TaskPreset, args: argparse.Namespace) -> Tuple[TaskPreset, SweepGrids]:
    if args.debug:
        preset = TaskPreset(
            **{
                **asdict(preset),
                "T_list": (preset.T_list[0],),
                "L_list": (preset.L_list[0],),
                "K_parallel_list": (preset.K_parallel_list[0],),
                "P_rec": min(32, preset.P_rec),
                "P_last": min(32, preset.P_last),
                "n_train": 64,
                "n_val": 32,
                "n_test": 64,
            }
        )
        return preset, SweepGrids(
            beta_leak_grid=(0.9, 0.99),
            threshold_grid=(1.0,),
            input_scale_grid=(1.0, 2.0),
            ridge_lambdas=(1e-2, 1.0),
            cvx_beta_grid=(1e-1,),
            cvx_bias_grid=(0.0,),
        )

    grids = SweepGrids(
        beta_leak_grid=tuple(args.beta_leak_grid),
        threshold_grid=tuple(args.threshold_grid),
        input_scale_grid=tuple(args.input_scale_grid),
        ridge_lambdas=tuple(args.ridge_lambdas),
        cvx_beta_grid=tuple(args.cvx_beta_grid),
        cvx_bias_grid=tuple(args.cvx_bias_grid),
    )
    return preset, grids


def _make_out_dir(root: Path, *, task: str, stamp: str) -> Path:
    out = root / f"lsm_{task}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _run_task(
    *,
    preset: TaskPreset,
    args: argparse.Namespace,
    seeds: Sequence[int],
    out_root: Path,
    ckpt_dir: Path,
    run_ridge: bool,
    run_r_cvx: bool,
) -> Dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _make_out_dir(out_root, task=preset.name, stamp=stamp)
    eff_preset, grids = _apply_debug(preset, args)

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        _set_seed(int(seed))
        for T in eff_preset.T_list:
            data = _load_task_data(eff_preset, T=int(T), seed=int(seed))
            for L in eff_preset.L_list:
                for K in eff_preset.K_parallel_list:
                    if eff_preset.P_rec % int(K) != 0 or eff_preset.P_last % int(K) != 0:
                        raise ValueError(
                            f"P_rec={eff_preset.P_rec} / P_last={eff_preset.P_last} must be "
                            f"divisible by K_parallel={K}."
                        )
                    entry = _run_R_and_R_CVX_single(
                        preset=eff_preset,
                        T=int(T),
                        L=int(L),
                        K_parallel=int(K),
                        seed=int(seed),
                        data=data,
                        grids=grids,
                        reservoir_seed_base=int(seed) * 100003 + 1,
                        run_ridge=bool(run_ridge),
                        run_r_cvx=bool(run_r_cvx),
                        probe_subsample=int(args.probe_subsample),
                        cvx_method=str(args.cvx_method),
                        cvx_ovr_workers=int(args.cvx_ovr_workers),
                        ckpt_dir=ckpt_dir,
                        compute_ce_dual=bool(args.cvx_ce_dual),
                        lite_max_iter=int(args.lite_max_iter),
                        lite_tol=float(args.lite_tol),
                    )
                    results.append(entry)
                    _write_json(
                        out_dir / f"seed{seed}_T{T}_L{L}_K{K}.json",
                        {"config": asdict(eff_preset), "grids": asdict(grids), **entry},
                    )

    payload = {
        "task": preset.name,
        "config": asdict(eff_preset),
        "grids": asdict(grids),
        "seeds": list(int(s) for s in seeds),
        "results": results,
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "metrics.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Criticality-tuned LSM baseline + optional R-CVX on XOR, MNIST-as-sequence, "
            "and base-b addition (add_b2/3/5/7/10). "
            "Default --tasks runs all of them. "
            "Split across machines with --side: non_cvx runs criticality+ridge and writes "
            "the tuned reservoir to --ckpt_dir; cvx loads that checkpoint and runs R-CVX "
            "(use --cvx_method cvx_lite on the big CPU)."
        )
    )
    ap.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--side",
        choices=("all", "cvx", "non_cvx"),
        default="all",
        help=(
            "all: ridge, plus R-CVX if --r_cvx. "
            "cvx: R-CVX only (load tuned reservoir from --ckpt_dir). "
            "non_cvx: criticality + ridge only; writes LSM weights to --ckpt_dir."
        ),
    )
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="LSM reservoir checkpoints. Default: <out_root>/ckpts. Rsync this dir onto the CVX machine.",
    )

    # Criticality-tuning grids
    ap.add_argument("--beta_leak_grid", type=float, nargs="+", default=[0.90, 0.95, 0.99, 0.995])
    ap.add_argument("--threshold_grid", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--input_scale_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument(
        "--probe_subsample", type=int, default=256,
        help="Number of training inputs used to estimate the branching ratio per grid point.",
    )

    # Ridge readout grid
    ap.add_argument("--ridge_lambdas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])

    # R-CVX
    ap.add_argument("--r_cvx", action="store_true", help="With --side all, also run R-CVX after ridge.")
    ap.add_argument(
        "--cvx_method",
        choices=["cvx", "cvx_lite", "sgd"],
        default="cvx",
        help="cvx = CLARABEL/SCS cone program (slow). cvx_lite = primal-only LASSO CD/FISTA (fast).",
    )
    ap.add_argument("--cvx_ovr_workers", type=int, default=1)
    ap.add_argument("--cvx_beta_grid", type=float, nargs="+", default=[1e-2, 1e-1, 1.0])
    ap.add_argument("--cvx_bias_grid", type=float, nargs="+", default=[0.0])
    ap.add_argument(
        "--cvx_ce_dual",
        action="store_true",
        help="Also solve the CVXPY dual after the primal. Off by default; ignored by cvx_lite.",
    )
    ap.add_argument("--lite_max_iter", type=int, default=5000)
    ap.add_argument("--lite_tol", type=float, default=1e-6)

    # Preset overrides
    ap.add_argument(
        "--reservoir_variant",
        choices=["standard", "normalized", "orthogonal"],
        default=None,
        help="Override preset reservoir_variant (default 'normalized').",
    )
    ap.add_argument(
        "--last_layer_readout",
        choices=["membrane", "spike"],
        default=None,
        help="Override preset readout mode.",
    )
    ap.add_argument("--debug", action="store_true", help="Tiny caps + shrunk grids for smoke tests.")
    ap.add_argument("--out_root", type=str, default="sweep_results/lsm_xor_mnist")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else (out_root / "ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.side == "cvx":
        run_ridge, run_r_cvx = False, True
    elif args.side == "non_cvx":
        run_ridge, run_r_cvx = True, False
    else:
        run_ridge, run_r_cvx = True, bool(args.r_cvx)
    print(
        (
            f"[lsm-xor-mnist] side={args.side} run_ridge={run_ridge} run_r_cvx={run_r_cvx} "
            f"cvx_method={args.cvx_method} ckpt_dir={ckpt_dir}"
        ),
        flush=True,
    )

    all_payloads: Dict[str, Any] = {
        "tasks": {},
        "side": args.side,
        "run_ridge": run_ridge,
        "run_r_cvx": run_r_cvx,
        "ckpt_dir": str(ckpt_dir),
    }
    for task in args.tasks:
        preset = TASK_PRESETS[task]
        if args.reservoir_variant is not None:
            preset = TaskPreset(**{**asdict(preset), "reservoir_variant": args.reservoir_variant})
        if args.last_layer_readout is not None:
            preset = TaskPreset(**{**asdict(preset), "last_layer_readout": args.last_layer_readout})
        payload = _run_task(
            preset=preset,
            args=args,
            seeds=args.seeds,
            out_root=out_root,
            ckpt_dir=ckpt_dir,
            run_ridge=run_ridge,
            run_r_cvx=run_r_cvx,
        )
        all_payloads["tasks"][task] = payload

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(all_payloads, indent=2, default=str) + "\n")
    print(f"[lsm-xor-mnist] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
