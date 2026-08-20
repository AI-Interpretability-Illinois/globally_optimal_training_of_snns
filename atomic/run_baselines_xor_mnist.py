"""
SG / CVX / SG-CVX baseline sweeps on XOR (DFA ``first_last_xor``), MNIST-as-sequence,
and base-b addition (``add_b2`` / ``add_b3`` / ``add_b5`` / ``add_b7`` / ``add_b10``).

Task presets are imported from :mod:`run_lsm_xor_mnist` (single source of truth for
T, L, K_parallel, P_rec, P_last, n_train/val/test, loss_name, last_layer_readout) so the
LSM + R-CVX numbers and these three baselines are directly comparable cell-for-cell.

Default ``--tasks`` is the full preset list (XOR, MNIST, and every addition base).

Baselines
---------
* **SG** — surrogate-gradient trained SNN (``ste_solve`` / dispatches to
  ``ste_parallel_Solve`` when K_parallel > 1). Sweeps ``lr``; selects on the LSM
  runner's shared ``_last_step_acc`` metric so the accuracy definition matches the
  LSM ridge and R-CVX reports exactly. Rank-2 arithmetic labels use token accuracy.
* **CVX** — convex L1-hinge / CE readout on a Gaussian-random-init LIF stack (
  ``cvx_solve(mode='gaussian')`` — no pretraining). Sweeps ``(beta, bias)``.
* **SG-CVX** — SG pretrain → CVX finetune with ``pretrained_weights=<STE-extracted>``,
  ``mode='pretraining'``. Sweeps CVX ``(beta, bias)`` for each SG lr.

Accuracy reporting
------------------
For SG we use :func:`fine_tune._last_step_acc` on the trained model for rank-1 labels
(identical to what the LSM ridge and R-CVX pipelines report). Rank-2 arithmetic
labels use mean token accuracy over all timesteps. For CVX / SG-CVX the solver
returns ``trained_model["weights"]`` shape ``(P_last, num_classes)``, which operates
on the CVX-internal thresholded features (``1[mem - bias >= 0]`` for membrane, raw
spikes for spike). We reproduce that path via :func:`_cvx_split_accs_from_config`.
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
    from fine_tune import _extract_weight_list, _last_step_acc
    from run_lsm_xor_mnist import TASK_NAMES, TASK_PRESETS, TaskPreset, _load_task_data
    from solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from solvers.cvx_solve import _build_feature_map as _build_feature_map_k1
    from solvers.cvx_parallel_Solve import _build_feature_map as _build_feature_map_kp
    from solvers.ste_solve import SteModelConfig, SteSolveConfig, ste_solve
else:
    from .cvx_side_ckpt import ckpt_path, load_weight_list, require_ckpt_task, save_weight_list
    from .fine_tune import _extract_weight_list, _last_step_acc
    from .run_lsm_xor_mnist import TASK_NAMES, TASK_PRESETS, TaskPreset, _load_task_data
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, cvx_solve
    from .solvers.cvx_solve import _build_feature_map as _build_feature_map_k1
    from .solvers.cvx_parallel_Solve import _build_feature_map as _build_feature_map_kp
    from .solvers.ste_solve import SteModelConfig, SteSolveConfig, ste_solve


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_acc_from_model(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    if y.ndim != 2:
        raise ValueError(f"Token accuracy requires rank-2 labels, got rank {y.ndim}.")
    device = next(model.parameters()).device
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t)
        preds = logits.argmax(dim=-1).cpu().numpy()
    if preds.shape != y.shape:
        raise ValueError(f"Pred shape {preds.shape} != label shape {y.shape}.")
    return float(np.mean(preds == y))


def _split_accs_from_model(model: torch.nn.Module, data: Dict[str, Any]) -> Dict[str, float]:
    if data["y_train"].ndim == 2:
        return {
            "train_acc": _token_acc_from_model(model, data["x_train"], data["y_train"]),
            "val_acc": _token_acc_from_model(model, data["x_val"], data["y_val"]),
            "test_acc": _token_acc_from_model(model, data["x_test"], data["y_test"]),
        }
    return {
        "train_acc": float(_last_step_acc(model, data["x_train"], data["y_train"])),
        "val_acc": float(_last_step_acc(model, data["x_val"], data["y_val"])),
        "test_acc": float(_last_step_acc(model, data["x_test"], data["y_test"])),
    }


def _cvx_split_accs_from_config(
    *,
    init_cfg: InitializationConfig,
    cvx_weights: np.ndarray,
    data: Dict[str, Any],
) -> Dict[str, float]:
    """Reproduce CVX's own feature map (bit-exact) and compute train/val/test accuracy.

    We call CVX's internal ``_build_feature_map`` (imported from
    :mod:`solvers.cvx_solve` for ``K_parallel == 1`` and
    :mod:`solvers.cvx_parallel_Solve` for ``K_parallel > 1``). That function is
    the *same* one ``cvx_solve`` uses to build the design matrix ``d_train`` /
    ``d_val`` / ``d_test`` internally, so the features here match CVX's exactly:

    * For ``mode='gaussian'``: the single shared ``np.random.default_rng(init_cfg.seed)``
      is drawn sequentially across branches — this is what makes previous attempts
      that used per-(branch, layer) seeds mismatch CVX and yield near-chance
      accuracy.
    * For ``mode='pretraining'``: consumes ``init_cfg.pretrained_weights`` in
      branch-major order (``[branch0_fc0, ..., branch0_fcL-1, branch1_fc0, ...]``),
      matching :func:`fine_tune._extract_weight_list` and
      :func:`solvers.LSM.extract_lsm_weight_list`.

    Accuracy is then ``argmax(d @ W_cvx)`` vs ``y`` — same as CVX's internal
    ``pred = np.argmax(d_test @ w, axis=1)``.
    """
    all_timesteps = bool(data["y_train"].ndim == 2)
    fmap = _build_feature_map_kp if int(init_cfg.K_parallel) > 1 else _build_feature_map_k1
    d_tr, d_va, d_te, _meta = fmap(
        data["x_train"], data["x_val"], data["x_test"],
        init_cfg, all_timesteps=all_timesteps,
    )
    W = np.asarray(cvx_weights, dtype=np.float64)

    def _acc_from_features(d: np.ndarray, y_np: np.ndarray) -> float:
        preds_flat = np.argmax(d @ W, axis=1)
        if y_np.ndim == 1:
            # d is (N, P_last); labels are per-sequence -> compare directly
            return float(np.mean(preds_flat == y_np))
        if y_np.ndim == 2:
            n, t = int(y_np.shape[0]), int(y_np.shape[1])
            if preds_flat.shape[0] != n * t:
                raise ValueError(
                    f"Feature/label rowcount mismatch: preds {preds_flat.shape[0]} vs {n * t} (N*T)."
                )
            return float(np.mean(preds_flat.reshape(n, t) == y_np))
        raise ValueError(f"Expected rank-1 or rank-2 labels, got rank {y_np.ndim}.")

    return {
        "train_acc": _acc_from_features(d_tr, data["y_train"]),
        "val_acc": _acc_from_features(d_va, data["y_val"]),
        "test_acc": _acc_from_features(d_te, data["y_test"]),
    }


# ---------------------------------------------------------------------------#
# Baseline runners (one per method)
# ---------------------------------------------------------------------------#


@dataclass
class BaselineGrids:
    sg_lr_grid: Tuple[float, ...] = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
    sg_epochs: int = 200
    cvx_beta_grid: Tuple[float, ...] = (1e-2, 1e-1, 1.0)
    cvx_bias_grid: Tuple[float, ...] = (0.0,)


def run_sg(
    *,
    preset: TaskPreset,
    T: int,
    L: int,
    K_parallel: int,
    seed: int,
    data: Dict[str, Any],
    grids: BaselineGrids,
    reservoir_seed_base: int,
) -> Dict[str, Any]:
    """SG (STE) sweep over lr; select on val_acc; report train/val/test accs.

    Returns both the accuracy block and the *best model* so SG-CVX can reuse it
    without paying the training cost twice.
    """
    d_in = int(data["d_in"])
    num_classes = int(data["num_classes"])
    best_model = None
    best_val_acc = -1.0
    best_params: Dict[str, float] = {}
    best_accs: Dict[str, float] = {}
    for lr in grids.sg_lr_grid:
        _set_seed(seed)
        out = ste_solve(
            x_train=data["x_train"], y_train=data["y_train"],
            x_val=data["x_val"], y_val=data["y_val"],
            x_test=data["x_test"], y_test=data["y_test"],
            model_cfg=SteModelConfig(
                d_in=d_in,
                num_classes=num_classes,
                L=int(L),
                P_rec=int(preset.P_rec),
                P_last=int(preset.P_last),
                K_parallel=int(K_parallel),
                last_layer_readout=str(preset.last_layer_readout),
            ),
            solve_cfg=SteSolveConfig(
                loss_name=str(preset.loss_name),
                optimizer_name="adam",
                lr=float(lr),
                epochs=int(grids.sg_epochs),
                batch_size=None,
                log_every=0,
                weight_decay=0.0,
                beta_path_reg=0.0,
            ),
        )
        accs = _split_accs_from_model(out.model, data)
        if accs["val_acc"] > best_val_acc:
            best_val_acc = float(accs["val_acc"])
            best_model = out.model
            best_params = {"lr": float(lr)}
            best_accs = accs
    if best_model is None:
        raise RuntimeError("SG sweep produced no candidates.")
    print(
        (
            f"[baselines] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
            f"SG lr={best_params['lr']} "
            f"train_acc={best_accs['train_acc']:.4f} val_acc={best_accs['val_acc']:.4f} test_acc={best_accs['test_acc']:.4f}"
        ),
        flush=True,
    )
    return {
        "selected_params": best_params,
        "split_accs": best_accs,
        "model": best_model,
    }


def run_cvx(
    *,
    preset: TaskPreset,
    T: int,
    L: int,
    K_parallel: int,
    seed: int,
    data: Dict[str, Any],
    grids: BaselineGrids,
    reservoir_seed_base: int,
    cvx_method: str,
    cvx_ovr_workers: int,
    compute_ce_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
    pretrained_weights: Optional[List[np.ndarray]] = None,
    method_tag: str = "CVX",
) -> Dict[str, Any]:
    """CVX sweep (beta × bias); ``pretrained_weights=None`` -> Gaussian-init CVX;
    otherwise -> SG-CVX (mode='pretraining'). Select on val_acc from
    :func:`_cvx_split_accs_from_config`; report train/val/test accs.
    """
    mode = "gaussian" if pretrained_weights is None else "pretraining"

    best_val_acc = -1.0
    best_params: Dict[str, float] = {}
    best_accs: Dict[str, float] = {}
    best_final_losses: Dict[str, float] = {}
    for bias in grids.cvx_bias_grid:
        for beta in grids.cvx_beta_grid:
            init_cfg = InitializationConfig(
                mode=mode,
                variant="standard",
                seed=int(reservoir_seed_base),
                feature_count=int(preset.P_last),
                bias=float(bias),
                pretrained_weights=pretrained_weights,
                L=int(L),
                P_rec=int(preset.P_rec),
                P_last=int(preset.P_last),
                K_parallel=int(K_parallel),
                beta_leak=0.99,
                threshold=1.0,
                last_layer_readout=str(preset.last_layer_readout),
            )
            out = cvx_solve(
                x_train=data["x_train"], y_train=data["y_train"],
                x_val=data["x_val"], y_val=data["y_val"],
                x_test=data["x_test"], y_test=data["y_test"],
                init_cfg=init_cfg,
                solve_cfg=SolveConfig(
                    method=str(cvx_method),
                    loss_name=str(preset.loss_name),
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
            W = out.trained_model["weights"]  # (P_last, num_classes)
            accs = _cvx_split_accs_from_config(
                init_cfg=init_cfg,
                cvx_weights=W,
                data=data,
            )
            if accs["val_acc"] > best_val_acc:
                best_val_acc = float(accs["val_acc"])
                best_params = {"beta": float(beta), "bias": float(bias)}
                best_accs = accs
                best_final_losses = {k: float(v) for k, v in out.final_losses.items()}
    if not best_accs:
        raise RuntimeError(f"{method_tag} sweep produced no candidates.")
    print(
        (
            f"[baselines] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
            f"{method_tag} best_params={best_params} "
            f"train_acc={best_accs['train_acc']:.4f} val_acc={best_accs['val_acc']:.4f} test_acc={best_accs['test_acc']:.4f}"
        ),
        flush=True,
    )
    return {
        "selected_params": best_params,
        "split_accs": best_accs,
        "final_losses": best_final_losses,
    }


# ---------------------------------------------------------------------------#
# Driver
# ---------------------------------------------------------------------------#


def _run_cell(
    *,
    preset: TaskPreset,
    T: int,
    L: int,
    K_parallel: int,
    seed: int,
    data: Dict[str, Any],
    grids: BaselineGrids,
    reservoir_seed_base: int,
    cvx_method: str,
    cvx_ovr_workers: int,
    which: Sequence[str],
    side: str,
    ckpt_dir: Path,
    compute_ce_dual: bool,
    lite_max_iter: int,
    lite_tol: float,
) -> Dict[str, Any]:
    """Run the requested subset of {sg, cvx, sg_cvx} on one (T, L, K, seed) cell.

    ``sg_cvx`` reuses the trained ``sg`` model — so if you request ``sg_cvx`` we
    either compute ``sg`` first, or load it from ``ckpt_dir`` (CVX-side machine).
    """
    entry: Dict[str, Any] = {
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
        "seed": int(seed),
        "side": str(side),
        "cvx_method": str(cvx_method),
    }

    sg_ckpt = ckpt_path(ckpt_dir, seed=seed, T=T, L=L, K=K_parallel, tag="sg", task=preset.name)
    need_sg_train = "sg" in which or ("sg_cvx" in which and side != "cvx")
    sg_block: Optional[Dict[str, Any]] = None
    pretrained_for_sg_cvx: Optional[List[np.ndarray]] = None

    if need_sg_train:
        sg_block = run_sg(
            preset=preset, T=T, L=L, K_parallel=K_parallel, seed=seed,
            data=data, grids=grids, reservoir_seed_base=reservoir_seed_base,
        )
        pretrained_for_sg_cvx = _extract_weight_list(sg_block["model"])
        save_weight_list(
            sg_ckpt,
            pretrained_for_sg_cvx,
            meta={
                "tag": "sg",
                "task": preset.name,
                "dataset": preset.dataset,
                "arith_op": preset.arith_op,
                "arith_base": preset.arith_base,
                "n_digits": preset.n_digits,
                "T": int(T),
                "L": int(L),
                "K_parallel": int(K_parallel),
                "P_rec": int(preset.P_rec),
                "P_last": int(preset.P_last),
                "seed": int(seed),
                "selected_params": sg_block["selected_params"],
                "split_accs": sg_block["split_accs"],
            },
        )
    elif "sg_cvx" in which:
        pretrained_for_sg_cvx, sg_meta = load_weight_list(sg_ckpt)
        require_ckpt_task(sg_meta, expected_task=preset.name, path=sg_ckpt)
        sg_block = {
            "selected_params": sg_meta["selected_params"],
            "split_accs": sg_meta["split_accs"],
            "model": None,
        }
        print(
            (
                f"[baselines] task={preset.name} T={T} L={L} K={K_parallel} seed={seed} "
                f"SG loaded from {sg_ckpt} "
                f"train_acc={sg_block['split_accs']['train_acc']:.4f} "
                f"val_acc={sg_block['split_accs']['val_acc']:.4f} "
                f"test_acc={sg_block['split_accs']['test_acc']:.4f}"
            ),
            flush=True,
        )

    entry["sg"] = (
        None
        if sg_block is None or "sg" not in which
        else {"selected_params": sg_block["selected_params"], "split_accs": sg_block["split_accs"]}
    )

    cvx_kw = dict(
        preset=preset, T=T, L=L, K_parallel=K_parallel, seed=seed, data=data,
        grids=grids, reservoir_seed_base=reservoir_seed_base,
        cvx_method=cvx_method, cvx_ovr_workers=cvx_ovr_workers,
        compute_ce_dual=compute_ce_dual,
        lite_max_iter=lite_max_iter, lite_tol=lite_tol,
    )
    if "cvx" in which:
        entry["cvx"] = run_cvx(**cvx_kw, pretrained_weights=None, method_tag="CVX")

    if "sg_cvx" in which:
        if pretrained_for_sg_cvx is None:
            raise RuntimeError(
                "sg_cvx requires SG weights: train with --side non_cvx (or --which sg) "
                f"or pass --ckpt_dir pointing at {sg_ckpt}."
            )
        entry["sg_cvx"] = run_cvx(
            **cvx_kw, pretrained_weights=pretrained_for_sg_cvx, method_tag="SG-CVX",
        )

    return entry


def _make_out_dir(root: Path, *, task: str, stamp: str) -> Path:
    out = root / f"baselines_{task}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _apply_debug(preset: TaskPreset, args: argparse.Namespace) -> Tuple[TaskPreset, BaselineGrids]:
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
        return preset, BaselineGrids(sg_lr_grid=(1e-2,), sg_epochs=5, cvx_beta_grid=(1e-1,), cvx_bias_grid=(0.0,))
    grids = BaselineGrids(
        sg_lr_grid=tuple(args.sg_lr_grid),
        sg_epochs=int(args.sg_epochs),
        cvx_beta_grid=tuple(args.cvx_beta_grid),
        cvx_bias_grid=tuple(args.cvx_bias_grid),
    )
    return preset, grids


def _run_task(
    *,
    preset: TaskPreset,
    args: argparse.Namespace,
    seeds: Sequence[int],
    out_root: Path,
    which: Sequence[str],
    ckpt_dir: Path,
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
                    entry = _run_cell(
                        preset=eff_preset, T=int(T), L=int(L), K_parallel=int(K), seed=int(seed),
                        data=data, grids=grids, reservoir_seed_base=int(seed) * 100003 + 1,
                        cvx_method=str(args.cvx_method), cvx_ovr_workers=int(args.cvx_ovr_workers),
                        which=which,
                        side=str(args.side),
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
            "SG / CVX / SG-CVX baselines on the same TaskPreset as run_lsm_xor_mnist "
            "(XOR, MNIST, add_b2/3/5/7/10). Default --tasks runs all of them. "
            "Split across machines with --side: non_cvx trains SG and writes checkpoints; "
            "cvx loads those checkpoints and runs CVX + SG-CVX (use --cvx_method cvx_lite on the big CPU)."
        )
    )
    ap.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--which", nargs="+", choices=("sg", "cvx", "sg_cvx"), default=["sg", "cvx", "sg_cvx"])
    ap.add_argument(
        "--side",
        choices=("all", "cvx", "non_cvx"),
        default="all",
        help=(
            "all: honor --which. "
            "cvx: CVX-family only. Default which → cvx+sg_cvx; pass --which cvx to skip SG-CVX "
            "(no checkpoint needed). sg_cvx still loads SG weights from --ckpt_dir. "
            "non_cvx: only SG; writes SG weights to --ckpt_dir for a later --side cvx run."
        ),
    )
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="SG weight checkpoints. Default: <out_root>/ckpts. Rsync this dir onto the CVX machine.",
    )

    # SG
    ap.add_argument("--sg_lr_grid", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    ap.add_argument("--sg_epochs", type=int, default=200)

    # CVX / SG-CVX
    ap.add_argument(
        "--cvx_method",
        choices=["cvx", "cvx_lite", "sgd"],
        default="cvx",
        help="cvx = CLARABEL/SCS cone program (slow). cvx_lite = primal-only LASSO CD/FISTA (fast). sgd = CVX-SGD.",
    )
    ap.add_argument("--cvx_ovr_workers", type=int, default=1)
    ap.add_argument("--cvx_beta_grid", type=float, nargs="+", default=[1e-2, 1e-1, 1.0])
    ap.add_argument("--cvx_bias_grid", type=float, nargs="+", default=[0.0])
    ap.add_argument(
        "--cvx_ce_dual",
        action="store_true",
        help="Also solve the CVXPY dual after the primal (gap diagnostics). Off by default; never used by cvx_lite.",
    )
    ap.add_argument("--lite_max_iter", type=int, default=5000)
    ap.add_argument("--lite_tol", type=float, default=1e-6)

    ap.add_argument("--debug", action="store_true", help="Tiny caps + shrunk grids for smoke tests.")
    ap.add_argument("--out_root", type=str, default="sweep_results/baselines_xor_mnist")
    return ap.parse_args()


def _resolve_which(args: argparse.Namespace) -> List[str]:
    if args.side == "non_cvx":
        return ["sg"]
    if args.side == "cvx":
        cvx_methods = [m for m in list(args.which) if m in ("cvx", "sg_cvx")]
        if len(cvx_methods) == 0:
            return ["cvx", "sg_cvx"]
        return cvx_methods
    return list(args.which)


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve() if args.ckpt_dir else (out_root / "ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    which = _resolve_which(args)
    print(
        f"[baselines] side={args.side} which={which} cvx_method={args.cvx_method} ckpt_dir={ckpt_dir}",
        flush=True,
    )

    all_payloads: Dict[str, Any] = {"tasks": {}, "side": args.side, "which": which, "ckpt_dir": str(ckpt_dir)}
    for task in args.tasks:
        preset = TASK_PRESETS[task]
        payload = _run_task(
            preset=preset, args=args, seeds=args.seeds, out_root=out_root, which=which, ckpt_dir=ckpt_dir,
        )
        all_payloads["tasks"][task] = payload

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(all_payloads, indent=2, default=str) + "\n")
    print(f"[baselines] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
