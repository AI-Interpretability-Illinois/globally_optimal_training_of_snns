"""
LSM baseline + R-CVX fine-tune pipeline.

Ditto structural copy of :func:`atomic.fine_tune.run_fine_tune_pipeline` with the STE
surrogate-gradient backbone (``ste_solve``) swapped out for the reservoir-computing
:mod:`atomic.solvers.LSM`. Reproduces exactly the same five-stage schedule; the CVX
initialization stage is what we call **R-CVX** — CVX with the LSM's frozen reservoir
weights transferred in as ``pretrained_weights``.

Stages
------
1. **LSM pretrain** (``lsm_solve`` on the pretrain split): sweep learning rate and readout
   L2 shrinkage; the L reservoirs are frozen at init (seeded deterministically by
   ``LSMModelConfig.reservoir_seed``), so this only trains the shared linear ``classifier``.
2. **R-CVX** (``cvx_solve`` with ``mode="pretraining"``): CVX L1 readout solve on the LSM's
   frozen reservoir features. The reservoir weights extracted from the trained LSM feed
   the ``pretrained_weights`` slot of :class:`solvers.cvx_solve.InitializationConfig`;
   this replaces the Gaussian-init path in the STE pipeline.
3. **LSM post-train** (``lsm_solve`` on the same pretrained weights again): re-fit the
   readout starting from the identical frozen reservoir. Structurally parallel to the
   STE post-train stage; semantically it just re-runs the classifier fit under a
   potentially different sweep (equivalent to a warm-start baseline for the readout).

Design notes
------------
* API surface mirrors ``fine_tune.py``: same :class:`FineTuneConfig` fields are honored
  (``L``, ``P_rec``, ``P_last``, ``K_parallel``, ``ste_pretrain_epochs``, ``ste_lr_grid``,
  ``ste_beta_grid``, ``beta_grid``, ``lr_grid``, ``bias_grid`` etc.). LSM-specific knobs
  (``reservoir_seed``, ``reservoir_variant``) live on :class:`LSMFineTuneConfig`.
* Weight I/O uses the same npz layout (``w0, w1, ...``) as ``fine_tune._extract_weight_list``
  so LSM-pretrain checkpoints and STE-pretrain checkpoints are drop-in interchangeable at
  the R-CVX / CVX-from-pretraining step.
* Weight-list ordering is branch-major identical to
  ``ste_parallel_Solve.SNNBaselineSeq`` — see :func:`solvers.LSM.extract_lsm_weight_list`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from fine_tune import (
        FineTuneConfig,
        load_finetune_weight_checkpoint,
        load_snn_weight_arrays_from_npz,
    )
    from finetune_manifest import (
        assert_finetune_manifest_matches_runtime,
        validate_finetune_weight_shapes_against_manifest,
    )
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.LSM import (
        LSMBaselineSeq,
        LSMModelConfig,
        LSMSolveConfig,
        extract_lsm_weight_list,
        lsm_solve,
    )
else:
    from .fine_tune import (
        FineTuneConfig,
        load_finetune_weight_checkpoint,
        load_snn_weight_arrays_from_npz,
    )
    from .finetune_manifest import (
        assert_finetune_manifest_matches_runtime,
        validate_finetune_weight_shapes_against_manifest,
    )
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.LSM import (
        LSMBaselineSeq,
        LSMModelConfig,
        LSMSolveConfig,
        extract_lsm_weight_list,
        lsm_solve,
    )


@dataclass
class LSMFineTuneConfig(FineTuneConfig):
    """Same schema as :class:`FineTuneConfig` plus LSM-specific reservoir knobs.

    ``reservoir_seed`` seeds each branch reservoir deterministically (branch ``k`` uses
    ``reservoir_seed + k``); pass a distinct integer per outer ``seed`` if you want the
    LSM reservoir to co-vary with the training seed. ``reservoir_variant`` selects the
    entry distribution of each frozen ``W_in`` (``standard`` / ``normalized`` /
    ``orthogonal``); ``standard`` matches the CVX Gaussian-init distribution so R-CVX
    on a fresh reservoir is identical in law to CVX with Gaussian init.
    """

    reservoir_seed: int = 0
    reservoir_variant: str = "standard"


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _last_step_acc(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float:
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


def _extract_weight_list_lsm(model: torch.nn.Module) -> List[np.ndarray]:
    """Dispatch to :func:`solvers.LSM.extract_lsm_weight_list` for ``LSMBaselineSeq``.

    Also accepts a bare ``SNNBaselineSeq``-shaped model (attributes ``branches`` /
    ``classifier`` or ``fcs`` / ``classifier``) so a reload path can round-trip either.
    """
    if isinstance(model, LSMBaselineSeq):
        return extract_lsm_weight_list(model)
    weights: List[np.ndarray] = []
    if hasattr(model, "branches") and hasattr(model, "classifier"):
        for br in model.branches:
            for fc in br.fcs:
                weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    if hasattr(model, "fcs") and hasattr(model, "classifier"):
        for fc in model.fcs:
            weights.append(fc.weight.detach().cpu().numpy().copy())
        weights.append(model.classifier.weight.detach().cpu().numpy().copy())
        return weights
    raise TypeError(f"Unsupported LSM/SNN layout for weight export: {type(model)}")


def _save_lsm_pretrain_artifacts(
    save_root: Path,
    *,
    readout_mode: str,
    seed: int,
    cfg: LSMFineTuneConfig,
    num_classes: int,
    transferred_weights: List[np.ndarray],
    pretrain_selected: Mapping[str, float],
) -> Dict[str, str]:
    """Same npz + manifest layout as the STE pretrain export, tagged for LSM lineage."""
    root = save_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sub = root / readout_mode
    sub.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sub / "pretrain_lsm_weights.npz", **{f"w{i}": w for i, w in enumerate(transferred_weights)}
    )

    snn_weight_shapes = [[int(x) for x in w.shape] for w in transferred_weights]

    manifest: Dict[str, Any] = {
        "readout_mode": readout_mode,
        "last_layer_readout": readout_mode,
        "checkpoint_stage": "lsm_pretrain_only",
        "backbone": "lsm",
        "seed": seed,
        "T": cfg.T,
        "L": cfg.L,
        "P_in": cfg.P_in,
        "P_rec": cfg.P_rec,
        "P_last": cfg.P_last,
        "K_parallel": int(cfg.K_parallel),
        "num_classes": num_classes,
        "reservoir_seed": int(cfg.reservoir_seed),
        "reservoir_variant": str(cfg.reservoir_variant),
        "snn_weight_shapes": snn_weight_shapes,
        "loss_type": cfg.loss_type,
        "cvx_method": cfg.cvx_method,
        "lsm_selected": dict(pretrain_selected),
        "reload": {
            "pretrain_for_r_cvx_init": (
                "pretrain_lsm_weights.npz keys w0,w1,... in branch-major order: for each branch "
                "k=0..K-1, each hidden fc.weight in order; the final key is the LSM's shared "
                "classifier.weight (shape num_classes x P_last). Total tensors = "
                "K_parallel * n_hidden_layers + 1. All fc.weight tensors are the frozen reservoir "
                "matrices — pass the branch-major hidden weights (i.e. drop the classifier tail) "
                "to InitializationConfig.pretrained_weights for R-CVX."
            ),
            "note": (
                "Structurally identical layout to STE's pretrain_snn_weights.npz — R-CVX can be "
                "seeded from either an LSM export or an STE export, though R-CVX-from-STE is "
                "semantically the standard CVX-from-STE-pretrain path."
            ),
        },
    }
    (sub / "finetune_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {
        "dir": str(sub),
        "manifest": str(sub / "finetune_manifest.json"),
        "pretrain_lsm_weights": str(sub / "pretrain_lsm_weights.npz"),
    }


def _load_lsm_pretrain_weight_checkpoint(
    checkpoint_dir: Path | str,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Load an LSM pretrain checkpoint directory (contains ``pretrain_lsm_weights.npz``).

    Falls back to the STE pretrain npz name so a mixed lineage (e.g. reloading STE weights
    into R-CVX for a direct STE-vs-LSM head-to-head) works out of the box.
    """
    root = Path(checkpoint_dir).expanduser().resolve()
    man_path = root / "finetune_manifest.json"
    if not man_path.is_file():
        raise ValueError(f"Missing finetune_manifest.json under {root}.")
    manifest: Dict[str, Any] = json.loads(man_path.read_text())
    lsm_npz = root / "pretrain_lsm_weights.npz"
    ste_npz = root / "pretrain_snn_weights.npz"
    if lsm_npz.is_file():
        weights = load_snn_weight_arrays_from_npz(lsm_npz)
    elif ste_npz.is_file():
        weights = load_snn_weight_arrays_from_npz(ste_npz)
    else:
        raise ValueError(f"Neither pretrain_lsm_weights.npz nor pretrain_snn_weights.npz under {root}.")
    return manifest, weights


def run_lsm_r_cvx_pipeline(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    seed: int,
    cfg: LSMFineTuneConfig,
) -> Dict[str, object]:
    """Ditto-copy of :func:`fine_tune.run_fine_tune_pipeline` with LSM in place of STE.

    Runs, for each ``readout_mode`` in ``cfg.readout_modes``:

    1. LSM pretrain: sweep ``ste_lr_grid`` × ``ste_beta_grid`` (interpreted as LSM readout
       learning rate × L2 shrinkage on the classifier); select by validation loss +
       shrinkage penalty (mirrors STE's model-selection criterion).
    2. R-CVX: CVX solve on the LSM's frozen reservoir feature map (mode ``pretraining``,
       ``pretrained_weights`` = the LSM's branch-major hidden weights). Sweep
       ``beta_grid`` × ``lr_grid`` × ``bias_grid``, select by ``val_objective``, then
       re-run the winner with logging enabled.
    3. LSM post-train: sweep again with the same grids; the LSM reservoir is re-seeded
       identically so this is a fresh readout fit under the (possibly different) selected
       hyperparameters. Included for structural parity with the STE post-train stage.

    Returns the same nested structure as :func:`fine_tune.run_fine_tune_pipeline`, with
    the stage labels renamed to ``lsm_pretrain`` / ``r_cvx_by_source`` / ``lsm_post``.
    """
    if x_train.ndim != 3 or x_val.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Expected x_train/x_val/x_test to be rank-3 arrays shaped (N, T, d_in).")
    if x_train.shape[1] != cfg.T:
        raise ValueError(f"T mismatch: cfg.T={cfg.T}, but x_train has T={x_train.shape[1]}.")
    if x_train.shape[2] != cfg.P_in:
        raise ValueError(f"P_in mismatch: cfg.P_in={cfg.P_in}, but x_train has d_in={x_train.shape[2]}.")

    readout_modes = tuple(dict.fromkeys(cfg.readout_modes))
    if len(readout_modes) == 0:
        raise ValueError("LSMFineTuneConfig.readout_modes cannot be empty.")
    for mode in readout_modes:
        if mode not in ("membrane", "spike"):
            raise ValueError(f"Unsupported readout mode: {mode}. Expected membrane|spike.")

    by_readout: Dict[str, Dict[str, object]] = {}
    weight_paths_root: Dict[str, Dict[str, str]] = {}
    for readout_mode in readout_modes:
        _set_global_seed(seed)
        if cfg.init_weights_dir:
            ckpt_manifest, loaded_w = _load_lsm_pretrain_weight_checkpoint(Path(cfg.init_weights_dir))
            assert_finetune_manifest_matches_runtime(
                ckpt_manifest,
                readout_mode=readout_mode,
                P_in=cfg.P_in,
                P_rec=cfg.P_rec,
                P_last=cfg.P_last,
                L=cfg.L,
                num_classes=num_classes,
                K_parallel=int(cfg.K_parallel),
            )
            validate_finetune_weight_shapes_against_manifest(ckpt_manifest, loaded_w)
            transferred_weights = [np.asarray(w, dtype=np.float32) for w in loaded_w]
            pre_sel = ckpt_manifest.get("lsm_selected") or ckpt_manifest.get("pretrain_ste_selected")
            if pre_sel is None:
                raise ValueError(
                    f"Checkpoint manifest under {cfg.init_weights_dir!r} has no 'lsm_selected' "
                    "or 'pretrain_ste_selected' block."
                )
            best_pre_lr = float(pre_sel["lr"])
            best_pre_beta = float(pre_sel["beta"])
            _set_global_seed(seed)
            best_pre = lsm_solve(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                model_cfg=LSMModelConfig(
                    d_in=cfg.P_in,
                    num_classes=num_classes,
                    L=cfg.L,
                    P_rec=cfg.P_rec,
                    P_last=cfg.P_last,
                    K_parallel=cfg.K_parallel,
                    last_layer_readout=readout_mode,
                    reservoir_seed=int(cfg.reservoir_seed),
                    reservoir_variant=str(cfg.reservoir_variant),
                ),
                solve_cfg=LSMSolveConfig(
                    loss_name=cfg.loss_type,
                    optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                    lr=float(best_pre_lr),
                    epochs=0,
                    batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                    log_every=0,
                    weight_decay=0.0,
                    beta_path_reg=float(best_pre_beta),
                ),
                pretrained_weights=transferred_weights,
            )
            pretrained_model = best_pre.model
            if not isinstance(pretrained_model, LSMBaselineSeq):
                raise TypeError("Expected LSMBaselineSeq from lsm_solve.")
            pre_acc = {
                "train_last_step_acc": _last_step_acc(pretrained_model, x_train, y_train),
                "val_last_step_acc": _last_step_acc(pretrained_model, x_val, y_val),
                "test_last_step_acc": _last_step_acc(pretrained_model, x_test, y_test),
            }
            print(
                (
                    f"[lsm-r-cvx][{readout_mode}] init from checkpoint {cfg.init_weights_dir!r} "
                    f"lsm_pretrain_acc "
                    f"train={pre_acc['train_last_step_acc']:.4f} "
                    f"val={pre_acc['val_last_step_acc']:.4f} "
                    f"test={pre_acc['test_last_step_acc']:.4f}"
                ),
                flush=True,
            )
        else:
            best_pre = None
            best_pre_val = float("inf")
            best_pre_lr = None
            best_pre_beta = None
            for lsm_lr in cfg.ste_lr_grid:
                for lsm_beta in cfg.ste_beta_grid:
                    _set_global_seed(seed)
                    out = lsm_solve(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        x_test=x_test,
                        y_test=y_test,
                        model_cfg=LSMModelConfig(
                            d_in=cfg.P_in,
                            num_classes=num_classes,
                            L=cfg.L,
                            P_rec=cfg.P_rec,
                            P_last=cfg.P_last,
                            K_parallel=cfg.K_parallel,
                            last_layer_readout=readout_mode,
                            reservoir_seed=int(cfg.reservoir_seed),
                            reservoir_variant=str(cfg.reservoir_variant),
                        ),
                        solve_cfg=LSMSolveConfig(
                            loss_name=cfg.loss_type,
                            optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                            lr=float(lsm_lr),
                            epochs=cfg.ste_pretrain_epochs,
                            batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                            log_every=0,
                            weight_decay=0.0,
                            beta_path_reg=float(lsm_beta),
                        ),
                    )
                    val_loss = out.best_losses["val_loss"] + float(lsm_beta)
                    if val_loss < best_pre_val:
                        best_pre_val = val_loss
                        best_pre = out
                        best_pre_lr = float(lsm_lr)
                        best_pre_beta = float(lsm_beta)
            if best_pre is None or best_pre_lr is None or best_pre_beta is None:
                raise RuntimeError("LSM pretraining failed to produce any candidate.")
            pretrained_model = best_pre.model
            if not isinstance(pretrained_model, LSMBaselineSeq):
                raise TypeError("Expected LSMBaselineSeq from lsm_solve.")
            transferred_weights = _extract_weight_list_lsm(pretrained_model)
            pre_acc = {
                "train_last_step_acc": _last_step_acc(pretrained_model, x_train, y_train),
                "val_last_step_acc": _last_step_acc(pretrained_model, x_val, y_val),
                "test_last_step_acc": _last_step_acc(pretrained_model, x_test, y_test),
            }
            print(
                (
                    f"[lsm-r-cvx][{readout_mode}] lsm_pretrain_acc "
                    f"train={pre_acc['train_last_step_acc']:.4f} "
                    f"val={pre_acc['val_last_step_acc']:.4f} "
                    f"test={pre_acc['test_last_step_acc']:.4f}"
                ),
                flush=True,
            )

        if cfg.weights_save_dir:
            tw_save = _extract_weight_list_lsm(best_pre.model)
            weight_paths_root[readout_mode] = _save_lsm_pretrain_artifacts(
                Path(cfg.weights_save_dir),
                readout_mode=readout_mode,
                seed=seed,
                cfg=cfg,
                num_classes=num_classes,
                transferred_weights=tw_save,
                pretrain_selected={"lr": float(best_pre_lr), "beta": float(best_pre_beta)},
            )
            print(
                f"[lsm-r-cvx][{readout_mode}] saved lsm pretrain weights to "
                f"{weight_paths_root[readout_mode]['dir']}",
                flush=True,
            )

        cvx_sources = tuple(dict.fromkeys(cfg.cvx_sources))
        if len(cvx_sources) == 0:
            raise ValueError("LSMFineTuneConfig.cvx_sources cannot be empty.")
        if any(source != "pretraining" for source in cvx_sources):
            raise ValueError(
                "R-CVX mode requires CVX initialization from LSM-pretraining weights only. "
                "Set cvx_sources=('pretraining',)."
            )

        r_cvx_results: Dict[str, object] = {}
        r_cvx_selected_params: Dict[str, Dict[str, float | None]] = {}
        cvx_lr_grid_eff = cvx_lr_sweep_values(cfg.cvx_method, cfg.lr_grid)
        for source in cvx_sources:
            init_cfg = InitializationConfig(
                mode="pretraining",
                seed=seed,
                feature_count=int(cfg.P_last),
                pretrained_weights=transferred_weights,
                L=int(cfg.L),
                P_rec=int(cfg.P_rec),
                P_last=int(cfg.P_last),
                K_parallel=int(cfg.K_parallel),
                last_layer_readout=readout_mode,
            )
            best_cvx = None
            best_cvx_val = float("inf")
            best_beta = None
            best_lr: float | None = None
            best_bias = None
            for cvx_beta in cfg.beta_grid:
                for cvx_lr in cvx_lr_grid_eff:
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
                            K_parallel=init_cfg.K_parallel,
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
                            best_lr = None if cfg.cvx_method == "cvx" else float(cvx_lr)
                            best_bias = float(cvx_bias)
            if best_cvx is None or best_beta is None or best_bias is None:
                raise RuntimeError(f"R-CVX sweep failed for source={source}.")
            if cfg.cvx_method == "sgd" and best_lr is None:
                raise RuntimeError(f"R-CVX(SGD) sweep failed to record lr for source={source}.")
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
                K_parallel=init_cfg.K_parallel,
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
                    lr=float(best_lr) if best_lr is not None else 0.0,
                    optimizer_name=cfg.cvx_optimizer,
                    epochs=cfg.epochs,
                    batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                    log_every=10,
                ),
            )
            r_cvx_results[source] = best_cvx
            r_cvx_selected_params[source] = {"beta": best_beta, "lr": best_lr, "bias": best_bias}

        best_post = None
        best_post_val = float("inf")
        best_post_lr = None
        best_post_beta = None
        for lsm_lr in cfg.ste_lr_grid:
            for lsm_beta in cfg.ste_beta_grid:
                _set_global_seed(seed)
                out = lsm_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    model_cfg=LSMModelConfig(
                        d_in=cfg.P_in,
                        num_classes=num_classes,
                        L=cfg.L,
                        P_rec=cfg.P_rec,
                        P_last=cfg.P_last,
                        K_parallel=cfg.K_parallel,
                        last_layer_readout=readout_mode,
                        reservoir_seed=int(cfg.reservoir_seed),
                        reservoir_variant=str(cfg.reservoir_variant),
                    ),
                    solve_cfg=LSMSolveConfig(
                        loss_name=cfg.loss_type,
                        optimizer_name=cfg.cvx_optimizer if cfg.cvx_optimizer in ("adam", "sgd") else "adam",
                        lr=float(lsm_lr),
                        epochs=cfg.ste_post_epochs,
                        batch_size=None if cfg.batch_size == -1 else int(cfg.batch_size),
                        log_every=0,
                        weight_decay=0.0,
                        beta_path_reg=float(lsm_beta),
                    ),
                    pretrained_weights=transferred_weights,
                )
                val_loss = out.best_losses["val_loss"] + float(lsm_beta)
                if val_loss < best_post_val:
                    best_post_val = val_loss
                    best_post = out
                    best_post_lr = float(lsm_lr)
                    best_post_beta = float(lsm_beta)
        if best_post is None or best_post_lr is None or best_post_beta is None:
            raise RuntimeError("LSM post-training sweep failed.")
        _set_global_seed(seed)
        best_post = lsm_solve(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            model_cfg=LSMModelConfig(
                d_in=cfg.P_in,
                num_classes=num_classes,
                L=cfg.L,
                P_rec=cfg.P_rec,
                P_last=cfg.P_last,
                K_parallel=cfg.K_parallel,
                last_layer_readout=readout_mode,
                reservoir_seed=int(cfg.reservoir_seed),
                reservoir_variant=str(cfg.reservoir_variant),
            ),
            solve_cfg=LSMSolveConfig(
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
            "lsm_pretrain": best_pre,
            "lsm_pretrain_selected_params": {"lr": best_pre_lr, "beta": best_pre_beta},
            "pretrain_accuracy": pre_acc,
            "r_cvx_by_source": r_cvx_results,
            "r_cvx_selected_params": r_cvx_selected_params,
            "lsm_post": best_post,
            "lsm_post_selected_params": {"lr": best_post_lr, "beta": best_post_beta},
            "lsm_post_loss_curve": [float(v) for v in best_post.loss_history],
            "r_cvx_loss_curves": {
                name: [float(v) for v in r_cvx_out.loss_history]
                for name, r_cvx_out in r_cvx_results.items()
            },
        }

    selected_mode = cfg.last_layer_readout if cfg.last_layer_readout in by_readout else readout_modes[0]
    selected = by_readout[selected_mode]
    json_report: Dict[str, object] = {
        "backbone": "lsm",
        "selected_readout": selected_mode,
        "by_readout": {
            mode: {
                "pretrain_accuracy": readout_data["pretrain_accuracy"],
                "lsm_pretrain_selected_params": dict(readout_data["lsm_pretrain_selected_params"]),
                "lsm_post_selected_params": readout_data["lsm_post_selected_params"],
                "lsm_post_best_losses": dict(readout_data["lsm_post"].best_losses),
                "lsm_post_loss_curve": list(readout_data["lsm_post_loss_curve"]),
                "r_cvx_selected_params": dict(readout_data["r_cvx_selected_params"]),
                "r_cvx_final_losses": {
                    src: dict(r_cvx_out.final_losses)
                    for src, r_cvx_out in readout_data["r_cvx_by_source"].items()
                },
                "r_cvx_loss_curves": {
                    src: list(curve) for src, curve in readout_data["r_cvx_loss_curves"].items()
                },
            }
            for mode, readout_data in by_readout.items()
        },
    }

    out_main: Dict[str, Any] = {
        "seed": seed,
        "cfg": cfg,
        "backbone": "lsm",
        "selected_readout": selected_mode,
        "by_readout": by_readout,
        "json_report": json_report,
        "lsm_pretrain": selected["lsm_pretrain"],
        "r_cvx_by_source": selected["r_cvx_by_source"],
        "lsm_post": selected["lsm_post"],
    }
    if weight_paths_root:
        out_main["weight_artifact_paths"] = weight_paths_root
    return out_main
