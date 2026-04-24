#!/usr/bin/env python3
"""
**General** last-timestep binary benchmark with the same **5-stage** hybrid pipeline as
``run_dfa_laststep_finetune_bench.py`` (STE pretrain, CVX from STE, STE finetune, CVX
Gaussian, STE finetune from CVX).

Supported **tasks** (``--task``):

* ``dfa`` — built-in DFA or ``random_N_M``; uses ``--dfa_spec`` (XOR/parity: e.g.
  ``first_last_xor``, ``parity_2``). **Length OOD** (T_ood = k * T_train) and per-block
  metrics are enabled (same as the DFA-only script).
* ``mnist_seq`` / ``mnist_perm_seq`` — image flattened to length ``T`` (see
  ``data_loaders.image_data_loader``). Binary target from class labels:
  ``--binary_target`` ``even_odd`` or ``high_low`` (class < 5).
* ``cifar_seq`` — CIFAR-10 sequences; binary via ``--binary_target``, or set
  ``--num_head_classes 10`` for full 10-class CE (STE + CVX use softmax CE on the
  last-step readout; hinge only when ``num_head_classes==1``).

**Length OOD** (``--ood_T_multipliers``) is applied **only** for ``task=dfa``:
for image/cifar, timestep length changes ``d_in`` under the current image sequence
code, so the same SNN head cannot be evaluated on a longer T. Those runs still report
in-distribution test metrics for all 5 stages; OOD length blocks are omitted and
``run_config.ood_mode`` records ``"image: id only"``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Reuse the full SNN + CVX + DFA-helpers implementation.
_LS_NAME = "run_dfa_laststep_finetune_bench"
_LS_PATH = Path(__file__).resolve().parent / f"{_LS_NAME}.py"
_spec = importlib.util.spec_from_file_location(_LS_NAME, _LS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load {_LS_PATH}")
LS = importlib.util.module_from_spec(_spec)
sys.modules[_LS_NAME] = LS
_spec.loader.exec_module(LS)

if __package__ in (None, ""):
    from data_loaders.dfa_data_loader import get_dfa, make_dfa_dataset
    from data_loaders import image_data_loader as img_ld
else:
    from .data_loaders.dfa_data_loader import get_dfa, make_dfa_dataset
    from .data_loaders import image_data_loader as img_ld

from datetime import datetime

if __package__ in (None, ""):
    from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT
else:
    from .solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT


def _import_binary_labels(y: np.ndarray, target: str) -> np.ndarray:
    y = y.reshape(-1).astype(np.int64, copy=False)
    if target == "even_odd":
        return (y % 2).astype(np.int64)
    if target == "high_low":
        return (y < 5).astype(np.int64)
    raise ValueError(f"binary_target must be even_odd|high_low, got {target!r}.")


def _load_dfa_laststep_numpy(
    dfa_spec: str,
    T: int,
    n_tr: int,
    n_vap: int,
    n_trf: int,
    n_vaf: int,
    n_te: int,
    pre_seed: int,
    ft_seed: int,
    eval_seed: int,
) -> Tuple[np.ndarray, ...]:
    Xtr, ytr, _ = make_dfa_dataset(str(dfa_spec), n_tr, T, seed=pre_seed + 11, balanced=True)
    Xva, yva, _ = make_dfa_dataset(str(dfa_spec), n_vap, T, seed=pre_seed + 29, balanced=True)
    Xte, yte, _ = make_dfa_dataset(str(dfa_spec), n_te, T, seed=eval_seed + 47, balanced=True)
    Xtrf, ytrf, _ = make_dfa_dataset(str(dfa_spec), n_trf, T, seed=ft_seed + 11, balanced=True)
    Xvaf, yvaf, _ = make_dfa_dataset(str(dfa_spec), n_vaf, T, seed=ft_seed + 29, balanced=True)
    d_in = int(Xtr.shape[2])
    for name, a in {
        "pre_va": (Xva, d_in),
        "pre_te": (Xte, d_in),
        "ft_tr": (Xtrf, d_in),
        "ft_va": (Xvaf, d_in),
    }.items():
        if int(a[0].shape[2]) != a[1]:
            raise ValueError(f"d_in mismatch in {name}")
    return (Xtr, ytr, Xva, yva, Xte, yte, Xtrf, ytrf, Xvaf, yvaf, d_in)


def _load_image_laststep_numpy(
    family: str,
    im_task: str,
    T: int,
    binary_target: str,
    n_tr: int,
    n_vap: int,
    n_trf: int,
    n_vaf: int,
    n_te: int,
    pre_seed: int,
    ft_seed: int,
    eval_seed: int,
    num_head_classes: int = 1,
) -> Tuple[np.ndarray, ...]:
    if im_task not in ("mnist_seq", "mnist_perm_seq", "cifar_seq"):
        raise ValueError(im_task)
    if family == "mnist":
        x_train, y_train, x_test, y_test = img_ld._mnist_arrays(0)  # type: ignore[attr-defined]
    elif family == "cifar":
        x_train, y_train, x_test, y_test = img_ld._cifar_arrays(0)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"family: {family}")

    nhc = int(num_head_classes)

    def pack(x, y, n, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        x_s, y_s = img_ld._subsample(x, y, n, seed)  # type: ignore[attr-defined]
        xq = img_ld._flatten_to_sequence(x_s, T=T)  # type: ignore[attr-defined]
        if nhc <= 1:
            yb = _import_binary_labels(y_s, binary_target)
        else:
            yb = y_s.reshape(-1).astype(np.int64, copy=False)
            if int(yb.max()) >= nhc or int(yb.min()) < 0:
                raise ValueError(f"Labels out of range for num_head_classes={nhc}.")
        return xq, yb

    Xtr, ytr = pack(x_train, y_train, n_tr, pre_seed + 11)
    Xva, yva = pack(x_train, y_train, n_vap, pre_seed + 29)
    Xte, yte = pack(x_test, y_test, n_te, eval_seed + 47)
    Xtrf, ytrf = pack(x_train, y_train, n_trf, ft_seed + 11)
    Xvaf, yvaf = pack(x_train, y_train, n_vaf, ft_seed + 29)

    if im_task == "mnist_perm_seq":
        rng = np.random.default_rng(pre_seed + 3)
        perm = rng.permutation(Xtr.shape[2])
        for a in (Xtr, Xva, Xte, Xtrf, Xvaf):
            a[:, :, :] = a[:, :, perm]

    d_in = int(Xtr.shape[2])
    return (Xtr, ytr, Xva, yva, Xte, yte, Xtrf, ytrf, Xvaf, yvaf, d_in)


def _id_only_metrics(
    model: Any,
    va: Any,
    te: Any,
    best_beta: float,
) -> Dict[str, Any]:
    y_te = te.y
    pr_te = LS.last_step_class_pred(model, te.X)
    return {
        "id_test": {
            "last_step_acc": float((pr_te == y_te).mean()),
            "val_total_objective": LS.last_step_val_objective(model, va.X, va.y, best_beta),
            "test_total_objective": LS.last_step_val_objective(model, te.X, te.y, best_beta),
        },
        "ood_eval": {},
        "ood_eval_note": "No length OOD: image/cifar d_in changes with T under fixed flatten. Use --task dfa for OOD T.",
    }


def _cvx_stage_no_ood(bundle: Dict[str, Any]) -> Dict[str, Any]:
    b = {
        "id_val": dict(bundle["id_val"]),
        "id_test": dict(bundle["id_test"]),
        "ood_eval": {},
        "ood_eval_note": "image/cifar: CVX OOD length not applicable (same as SNN ood note).",
    }
    return b


def _run_one_seed_dfa(
    args: Any,
    t_dim: int,
    n_tr: int,
    n_vap: int,
    n_trf: int,
    n_vaf: int,
    n_te: int,
    n_ood: int,
    ood_m: List[int],
    s_pre: int,
    s_ft: int,
    lro: str,
    base_seed: int,
) -> Dict[str, Any]:
    dfa = get_dfa(str(args.dfa_spec))
    pre_seed = int(base_seed)
    ft_seed = int(base_seed) + int(args.finetune_seed_offset)
    eval_seed = int(base_seed) + int(args.eval_seed_offset)
    ood_data_base = int(eval_seed)

    pack = _load_dfa_laststep_numpy(
        str(args.dfa_spec), t_dim, n_tr, n_vap, n_trf, n_vaf, n_te, pre_seed, ft_seed, eval_seed
    )
    Xtr, ytr, Xva, yva, Xte, yte, Xtrf, ytrf, Xvaf, yvaf, d_in = pack

    tr_p = LS.DFALastDS(Xtr, ytr, d_in)
    va_p = LS.DFALastDS(Xva, yva, d_in)
    te_ds = LS.DFALastDS(Xte, yte, d_in)
    tr_f = LS.DFALastDS(Xtrf, ytrf, d_in)
    va_f = LS.DFALastDS(Xvaf, yvaf, d_in)

    m_ste, ste_pre_sel, _ = LS.ste_sweep_laststep(
        tr_p, va_p, te_ds,
        L=int(args.L), P_rec=int(args.P_rec), P_last=int(args.P_last), K_parallel=int(args.K_parallel),
        last_layer_readout=lro, ste_epochs=s_pre, batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name), beta_leak=float(args.beta_leak), threshold=float(args.threshold),
        seed=pre_seed, ste_lr_grid=args.ste_lr_grid, ste_beta_grid=args.ste_beta_grid,
    )
    b_ste = float(ste_pre_sel["beta"])
    ev_ste = LS._metrics_nn(
        dfa, m_ste, va_p, te_ds, b_ste, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
    )
    br_w = LS._extract_laststep_branch_weights(m_ste)

    cvx_fs = LS.cvx_binary_lif_laststep_sweep(
        x_tr=Xtr, y_tr=ytr, x_va=Xva, y_va=yva, x_te=Xte, y_te=yte,
        L=int(args.L), P_rec=int(args.P_rec), P_last=int(args.P_last), K_parallel=int(args.K_parallel),
        last_layer_readout=lro, beta_leak=float(args.beta_leak), threshold=float(args.threshold),
        beta_grid=args.cvx_beta_grid, bias_grid=args.cvx_bias_grid, init_mode="pretraining",
        pretrained_weights=br_w, seed=pre_seed,
    )
    ood_cvx_fs = LS._stage_cvx_ood(cvx_fs, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base)
    m_sf, ste_ft_ste_sel, _ = LS.ste_sweep_laststep(
        tr_f, va_f, te_ds,
        L=int(args.L), P_rec=int(args.P_rec), P_last=int(args.P_last), K_parallel=int(args.K_parallel),
        last_layer_readout=lro, ste_epochs=s_ft, batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name), beta_leak=float(args.beta_leak), threshold=float(args.threshold),
        seed=ft_seed, ste_lr_grid=args.ste_lr_grid, ste_beta_grid=args.ste_beta_grid,
        init_state_dict={k: v.detach().cpu().clone() for k, v in m_ste.state_dict().items()},
    )
    b_sf = float(ste_ft_ste_sel["beta"])
    ev_sf = LS._metrics_nn(
        dfa, m_sf, va_f, te_ds, b_sf, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
    )

    cvx_g = LS.cvx_binary_lif_laststep_sweep(
        x_tr=Xtr, y_tr=ytr, x_va=Xva, y_va=yva, x_te=Xte, y_te=yte,
        L=int(args.L), P_rec=int(args.P_rec), P_last=int(args.P_last), K_parallel=int(args.K_parallel),
        last_layer_readout=lro, beta_leak=float(args.beta_leak), threshold=float(args.threshold),
        beta_grid=args.cvx_beta_grid, bias_grid=args.cvx_bias_grid, init_mode="gaussian",
        pretrained_weights=None, seed=pre_seed,
    )
    ood_cvx_g = LS._stage_cvx_ood(cvx_g, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base)
    w_cvxg = np.asarray(cvx_g["w"], dtype=np.float64)
    m_cx, ste_ft_cvx_sel, _ = LS.ste_sweep_laststep(
        tr_f, va_f, te_ds,
        L=int(args.L), P_rec=int(args.P_rec), P_last=int(args.P_last), K_parallel=int(args.K_parallel),
        last_layer_readout=lro, ste_epochs=s_ft, batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name), beta_leak=float(args.beta_leak), threshold=float(args.threshold),
        seed=ft_seed, ste_lr_grid=args.ste_lr_grid, ste_beta_grid=args.ste_beta_grid,
        init_state_dict=None, head_weight=w_cvxg,
    )
    b_cx = float(ste_ft_cvx_sel["beta"])
    ev_cx = LS._metrics_nn(
        dfa, m_cx, va_f, te_ds, b_cx, str(args.dfa_spec), n_ood, t_dim, ood_m, ood_data_base
    )

    return {
        "seed": int(base_seed),
        "task": "dfa",
        "dfa_spec": str(args.dfa_spec),
        "T": t_dim,
        "d_in": d_in,
        "split_seeds": {
            "pretrain_train": pre_seed + 11,
            "pretrain_val": pre_seed + 29,
            "finetune_train": ft_seed + 11,
            "finetune_val": ft_seed + 29,
            "eval_test": eval_seed + 47,
        },
        "stages": {
            "ste_pretrain": {"selected_params": ste_pre_sel, **ev_ste},
            "cvx_from_ste_pretrain": LS._public_cvx_record(cvx_fs, ood_cvx_fs),
            "ste_finetune_from_ste_pretrain_new_train": {"selected_params": ste_ft_ste_sel, **ev_sf},
            "cvx_pretrain_gaussian": LS._public_cvx_record(cvx_g, ood_cvx_g),
            "ste_finetune_from_cvx_pretrain_new_train": {"selected_params": ste_ft_cvx_sel, **ev_cx},
        },
    }


def _run_one_seed_image(
    args: Any,
    t_dim: int,
    n_tr: int,
    n_vap: int,
    n_trf: int,
    n_vaf: int,
    n_te: int,
    s_pre: int,
    s_ft: int,
    lro: str,
    base_seed: int,
) -> Dict[str, Any]:
    pre_seed = int(base_seed)
    ft_seed = int(base_seed) + int(args.finetune_seed_offset)
    eval_seed = int(base_seed) + int(args.eval_seed_offset)
    im_task = str(args.task)
    fam = "mnist" if im_task.startswith("mnist") else "cifar"
    nhc = int(args.num_head_classes)
    pack = _load_image_laststep_numpy(
        fam, im_task, t_dim, str(args.binary_target),
        n_tr, n_vap, n_trf, n_vaf, n_te, pre_seed, ft_seed, eval_seed,
        num_head_classes=nhc,
    )
    Xtr, ytr, Xva, yva, Xte, yte, Xtrf, ytrf, Xvaf, yvaf, d_in = pack

    tr_p = LS.DFALastDS(Xtr, ytr, d_in)
    va_p = LS.DFALastDS(Xva, yva, d_in)
    te_ds = LS.DFALastDS(Xte, yte, d_in)
    tr_f = LS.DFALastDS(Xtrf, ytrf, d_in)
    va_f = LS.DFALastDS(Xvaf, yvaf, d_in)

    m_ste, ste_pre_sel, _ = LS.ste_sweep_laststep(
        tr_p,
        va_p,
        te_ds,
        L=int(args.L),
        P_rec=int(args.P_rec),
        P_last=int(args.P_last),
        K_parallel=int(args.K_parallel),
        last_layer_readout=lro,
        ste_epochs=s_pre,
        batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name),
        beta_leak=float(args.beta_leak),
        threshold=float(args.threshold),
        seed=pre_seed,
        ste_lr_grid=args.ste_lr_grid,
        ste_beta_grid=args.ste_beta_grid,
        num_head_classes=nhc,
    )
    b_ste = float(ste_pre_sel["beta"])
    ev_ste = _id_only_metrics(m_ste, va_p, te_ds, b_ste)
    br_w = LS._extract_laststep_branch_weights(m_ste)

    if nhc <= 1:
        cvx_fs = LS.cvx_binary_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="pretraining",
            pretrained_weights=br_w,
            seed=pre_seed,
        )
    else:
        cvx_fs = LS.cvx_multiclass_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="pretraining",
            pretrained_weights=br_w,
            seed=pre_seed,
            num_classes=nhc,
        )
    ood_cvx_fs = _cvx_stage_no_ood(cvx_fs)
    m_sf, ste_ft_ste_sel, _ = LS.ste_sweep_laststep(
        tr_f,
        va_f,
        te_ds,
        L=int(args.L),
        P_rec=int(args.P_rec),
        P_last=int(args.P_last),
        K_parallel=int(args.K_parallel),
        last_layer_readout=lro,
        ste_epochs=s_ft,
        batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name),
        beta_leak=float(args.beta_leak),
        threshold=float(args.threshold),
        seed=ft_seed,
        ste_lr_grid=args.ste_lr_grid,
        ste_beta_grid=args.ste_beta_grid,
        num_head_classes=nhc,
        init_state_dict={k: v.detach().cpu().clone() for k, v in m_ste.state_dict().items()},
    )
    b_sf = float(ste_ft_ste_sel["beta"])
    ev_sf = _id_only_metrics(m_sf, va_f, te_ds, b_sf)

    if nhc <= 1:
        cvx_g = LS.cvx_binary_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="gaussian",
            pretrained_weights=None,
            seed=pre_seed,
        )
        w_cvxg = np.asarray(cvx_g["w"], dtype=np.float64)
    else:
        cvx_g = LS.cvx_multiclass_lif_laststep_sweep(
            x_tr=Xtr,
            y_tr=ytr,
            x_va=Xva,
            y_va=yva,
            x_te=Xte,
            y_te=yte,
            L=int(args.L),
            P_rec=int(args.P_rec),
            P_last=int(args.P_last),
            K_parallel=int(args.K_parallel),
            last_layer_readout=lro,
            beta_leak=float(args.beta_leak),
            threshold=float(args.threshold),
            beta_grid=args.cvx_beta_grid,
            bias_grid=args.cvx_bias_grid,
            init_mode="gaussian",
            pretrained_weights=None,
            seed=pre_seed,
            num_classes=nhc,
        )
        w_cvxg = np.asarray(cvx_g["w"], dtype=np.float64).T
    ood_cvx_g = _cvx_stage_no_ood(cvx_g)
    m_cx, ste_ft_cvx_sel, _ = LS.ste_sweep_laststep(
        tr_f,
        va_f,
        te_ds,
        L=int(args.L),
        P_rec=int(args.P_rec),
        P_last=int(args.P_last),
        K_parallel=int(args.K_parallel),
        last_layer_readout=lro,
        ste_epochs=s_ft,
        batch_size=int(args.batch_size),
        optimizer_name=str(args.optimizer_name),
        beta_leak=float(args.beta_leak),
        threshold=float(args.threshold),
        seed=ft_seed,
        ste_lr_grid=args.ste_lr_grid,
        ste_beta_grid=args.ste_beta_grid,
        num_head_classes=nhc,
        init_state_dict=None,
        head_weight=w_cvxg,
    )
    b_cx = float(ste_ft_cvx_sel["beta"])
    ev_cx = _id_only_metrics(m_cx, va_f, te_ds, b_cx)

    out_meta: Dict[str, Any] = {
        "seed": int(base_seed),
        "task": im_task,
        "T": t_dim,
        "d_in": d_in,
        "num_head_classes": nhc,
        "ste_loss": "hinge" if nhc == 1 else "ce (multiclass; STE + LIF readout)",
        "split_seeds": {
            "pretrain_train": pre_seed + 11,
            "pretrain_val": pre_seed + 29,
            "finetune_train": ft_seed + 11,
            "finetune_val": ft_seed + 29,
            "eval_test": eval_seed + 47,
        },
    }
    if nhc <= 1:
        out_meta["binary_target"] = str(args.binary_target)
    return {
        **out_meta,
        "stages": {
            "ste_pretrain": {"selected_params": ste_pre_sel, **ev_ste},
            "cvx_from_ste_pretrain": LS._public_cvx_record(cvx_fs, ood_cvx_fs),
            "ste_finetune_from_ste_pretrain_new_train": {"selected_params": ste_ft_ste_sel, **ev_sf},
            "cvx_pretrain_gaussian": LS._public_cvx_record(cvx_g, ood_cvx_g),
            "ste_finetune_from_cvx_pretrain_new_train": {"selected_params": ste_ft_cvx_sel, **ev_cx},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="General last-step hybrid: DFA (binary) or image seq (binary hinge or multiclass CE); same 5 stages as dfa laststep."
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument(
        "--task",
        type=str,
        default="dfa",
        help="dfa | mnist_seq | mnist_perm_seq | cifar_seq (DFA: set --dfa_spec, e.g. tomita_3, first_last_xor, parity_2).",
    )
    ap.add_argument("--dfa_spec", type=str, default="tomita_3", help="Used when --task dfa")
    ap.add_argument(
        "--binary_target",
        type=str,
        default="even_odd",
        choices=["even_odd", "high_low"],
        help="For image tasks: class%%2 vs class<5 (MNIST and CIFAR). Ignored when --num_head_classes>1 (raw class ids).",
    )
    ap.add_argument(
        "--num_head_classes",
        type=int,
        default=1,
        help="1: last-step binary head (hinge). >=2: multiclass CE (e.g. 10 for MNIST/CIFAR-10). DFA task is binary only (use 1).",
    )
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--n_train_pre", type=int, default=2304)
    ap.add_argument("--n_val_pre", type=int, default=512)
    ap.add_argument("--n_train_ft", type=int, default=2304)
    ap.add_argument("--n_val_ft", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--n_test_ood", type=int, default=1024)
    ap.add_argument("--ood_T_multipliers", type=int, nargs="*", default=[2, 5, 10])
    ap.add_argument("--finetune_seed_offset", type=int, default=1000)
    ap.add_argument("--eval_seed_offset", type=int, default=2000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=256)
    ap.add_argument("--P_last", type=int, default=512)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--last_layer_readout", choices=["membrane", "spike"], default="spike")
    ap.add_argument("--optimizer_name", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--ste_pretrain_epochs", type=int, default=100)
    ap.add_argument("--ste_finetune_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=-1)
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=list(LR_GRID_DEFAULT))
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    if str(args.task) not in ("dfa", "mnist_seq", "mnist_perm_seq", "cifar_seq"):
        raise ValueError(f"Unknown --task {args.task!r}; use dfa|mnist_seq|mnist_perm_seq|cifar_seq.")
    nhc = int(args.num_head_classes)
    if nhc < 1:
        raise ValueError("num_head_classes must be >= 1.")
    if str(args.task) == "dfa" and nhc != 1:
        raise ValueError("The DFA last-step path is binary only; set --num_head_classes 1.")
    if str(args.task) == "dfa" and str(args.dfa_spec) == "dyck1_unbounded":
        raise ValueError("Use bounded dyck1_d* for this benchmark.")
    t_dim = int(args.T)
    n_tr, n_vap, n_trf, n_vaf, n_te, n_ood = (
        int(args.n_train_pre),
        int(args.n_val_pre),
        int(args.n_train_ft),
        int(args.n_val_ft),
        int(args.n_test),
        int(args.n_test_ood),
    )
    s_pre, s_ft = int(args.ste_pretrain_epochs), int(args.ste_finetune_epochs)
    if bool(args.debug):
        n_tr, n_vap, n_trf, n_vaf, n_te, n_ood = 64, 32, 64, 32, 32, 32
        s_pre, s_ft = 2, 2
        if nhc > 1:
            # Enough rows per class for multiclass LIF features on a short debug run; grids stay at argparse defaults.
            n_tr, n_vap, n_trf, n_vaf, n_te, n_ood = 256, 128, 256, 128, 128, 128

    def _cap(n: int, cap: int) -> int:
        if int(cap) > 0:
            return min(int(n), int(cap))
        return int(n)

    mtr, mva, mte = int(args.max_train_samples), int(args.max_val_samples), int(args.max_test_samples)
    if mtr > 0:
        n_tr = _cap(n_tr, mtr)
        n_trf = _cap(n_trf, mtr)
    if mva > 0:
        n_vap = _cap(n_vap, mva)
        n_vaf = _cap(n_vaf, mva)
    if mte > 0:
        n_te = _cap(n_te, mte)
        n_ood = _cap(n_ood, mte)
    for name, n in (
        ("n_train_pre (after caps)", n_tr),
        ("n_val_pre (after caps)", n_vap),
        ("n_train_ft (after caps)", n_trf),
        ("n_val_ft (after caps)", n_vaf),
        ("n_test (after caps)", n_te),
        ("n_test_ood (after caps)", n_ood),
    ):
        if n < 1:
            raise ValueError(f"Invalid {name}={n}; increase sizes or relax max_*_samples caps.")

    ood_m = [int(m) for m in args.ood_T_multipliers]
    lro = str(args.last_layer_readout)
    is_dfa = str(args.task) == "dfa"
    ood_mode = "dfa: length ood" if is_dfa else "image/cifar: id only (d_in depends on T)"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_dfa:
        out_tag = f"genlaststep_dfa_{args.dfa_spec}"
    else:
        out_tag = (
            f"genlaststep_{args.task}_C{nhc}"
            if nhc > 1
            else f"genlaststep_{args.task}_{args.binary_target}"
        )
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root)).expanduser().resolve()
    else:
        out_root = Path.cwd() / "sweep_results" / f"{out_tag}_T{t_dim}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    config_dump: Dict[str, Any] = {
        "seeds": list(int(s) for s in args.seeds),
        "task": str(args.task),
        "dfa_spec": str(args.dfa_spec) if is_dfa else None,
        "binary_target": (None if is_dfa or nhc > 1 else str(args.binary_target)),
        "num_head_classes": nhc,
        "T": t_dim,
        "n_train_pre": n_tr,
        "n_val_pre": n_vap,
        "n_train_ft": n_trf,
        "n_val_ft": n_vaf,
        "n_test": n_te,
        "n_test_ood": n_ood,
        "ood_T_multipliers": ood_m,
        "ood_mode": ood_mode,
        "finetune_seed_offset": int(args.finetune_seed_offset),
        "eval_seed_offset": int(args.eval_seed_offset),
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "last_layer_readout": lro,
        "beta_leak": float(args.beta_leak),
        "threshold": float(args.threshold),
        "optimizer_name": str(args.optimizer_name),
        "ste_pretrain_epochs": s_pre,
        "ste_finetune_epochs": s_ft,
        "ste_lr_grid": list(args.ste_lr_grid),
        "ste_beta_grid": list(args.ste_beta_grid),
        "cvx_beta_grid": list(args.cvx_beta_grid),
        "cvx_bias_grid": list(args.cvx_bias_grid),
        "max_train_samples": mtr,
        "max_val_samples": mva,
        "max_test_samples": mte,
        "stages": [
            "ste_pretrain",
            "cvx_from_ste_pretrain",
            "ste_finetune_from_ste_pretrain_new_train",
            "cvx_pretrain_gaussian",
            "ste_finetune_from_cvx_pretrain_new_train",
        ],
        "debug": bool(args.debug),
    }
    (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2, default=str) + "\n")

    seed_payloads: List[Dict[str, Any]] = []
    for base_seed in [int(s) for s in args.seeds]:
        if is_dfa:
            print(f"[gen_laststep] dfa {args.dfa_spec} seed={base_seed}", flush=True)
            pl = _run_one_seed_dfa(
                args, t_dim, n_tr, n_vap, n_trf, n_vaf, n_te, n_ood, ood_m, s_pre, s_ft, lro, base_seed
            )
        else:
            print(f"[gen_laststep] {args.task} seed={base_seed}", flush=True)
            pl = _run_one_seed_image(args, t_dim, n_tr, n_vap, n_trf, n_vaf, n_te, s_pre, s_ft, lro, base_seed)
        seed_payloads.append(pl)
        sdir = out_root / f"seed_{int(base_seed)}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "metrics.json").write_text(json.dumps(pl, indent=2, default=str) + "\n")
        print(f"[gen_laststep] wrote {sdir / 'metrics.json'}", flush=True)

    root_payload: Dict[str, Any] = {
        "run_config": config_dump,
        "out_root": str(out_root),
        "n_seeds": int(len(seed_payloads)),
        "seeds": seed_payloads,
    }
    (out_root / "metrics.json").write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    oj = str(args.output_json).strip()
    if oj:
        Path(oj).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    print(json.dumps({"out_root": str(out_root), "n_seeds": len(seed_payloads)}, indent=2), flush=True)


if __name__ == "__main__":
    main()