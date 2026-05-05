#!/usr/bin/env python3
"""
DFA (Tomita / bounded Dyck) autoregressive hybrid bench — same five-stage structure as
``run_arithmetic_add_carry_finetune_bench.py``:

1. STE pretrain (split A)
2. CVX from STE pretrain
3. STE fine-tune (split B) from (1)
4. CVX from Gaussian init
5. STE fine-tune (split B) from (4)

State (next transition) targets map to the addition ``sum`` head (CE); per-step accept maps
to ``carry`` (hinge). The solvers scale **sum** loss by ``lambda_sum`` and **carry** loss by
``lambda_carry`` — **``lambda_sum`` is the state-head weight** and **``lambda_carry`` is the
label-head weight**. In this DFA script we **sweep** ``lambda_sum`` (state), and keep
``lambda_carry`` fixed (label) — the opposite of the addition bench, which sweeps the carry
slot (there: digit-sum vs carry; here: we care about the state / sum slot).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data_loaders.dfa_data_loader import (
    DFA,
    get_dfa,
    get_dfa_start_state_index,
    make_dfa_autoregressive_dataset,
    strings_to_autoregressive_tensors,
)
from solvers import cvx_carry_teacher_solve as cvx_cts
from solvers import cvx_parallel_Solve as cvx_par
from solvers import ste_carry_teacher_solve as ste_cts
from solvers.cvx_solve import InitializationConfig, _build_feature_map

from solver_grids import BETA_GRID_DEFAULT, LR_GRID_DEFAULT


# ---------------------------------------------------------------------------#
# load addition hybrid module (shared STE/CVX/dataset/aggregate)
# ---------------------------------------------------------------------------#


def _load_arith_carry_module() -> Any:
    """
    Load the addition hybrid bench. Plain import works when ``atomic`` is on ``sys.path``.
    Fallback uses importlib with the real module name registered in ``sys.modules`` *before*
    ``exec_module`` so ``@dataclass`` (and Python 3.14’s dataclasses) can resolve
    ``cls.__module__``.
    """
    try:
        import run_arithmetic_add_carry_finetune_bench as m

        return m
    except ImportError:
        pass
    name = "run_arithmetic_add_carry_finetune_bench"
    p = Path(__file__).resolve().parent / f"{name}.py"
    s = importlib.util.spec_from_file_location(name, p)
    if s is None or s.loader is None:
        raise ImportError("Cannot load run_arithmetic_add_carry_finetune_bench.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


ARITH = _load_arith_carry_module()


# ---------------------------------------------------------------------------#
# DFA data -> CarryAugmentedDataset
# ---------------------------------------------------------------------------#


@dataclass
class _Meta:
    dfa: DFA
    d_sym: int
    start_state_index: int


def build_dfa_carry_augmented_dataset_from_seeds(
    *,
    dfa_spec: str,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed_train: int,
    seed_val: int,
    seed_test: int,
    verify_count: int,
) -> Tuple[Any, _Meta]:
    p_tr = make_dfa_autoregressive_dataset(
        dfa_spec, n_train, T, seed=seed_train, balanced=True, return_strings=bool(verify_count > 0)
    )
    Xtr, yst_tr, yl_tr, n_sc, d_in, _spec, str_tr = p_tr
    p_va = make_dfa_autoregressive_dataset(
        dfa_spec, n_val, T, seed=seed_val, balanced=True, return_strings=False
    )
    Xva, yst_va, yl_va, n_sc2, d_in2, _, _ = p_va
    p_te = make_dfa_autoregressive_dataset(
        dfa_spec, n_test, T, seed=seed_test, balanced=True, return_strings=False
    )
    Xte, yst_te, yl_te, n_sc3, d_in3, _, _ = p_te
    if n_sc != n_sc2 or n_sc != n_sc3 or d_in != d_in2 or d_in != d_in3:
        raise ValueError("Train/val/test DFA autoregressive shape mismatch.")
    dfa = get_dfa(dfa_spec)
    d_sym = len(dfa.alphabet)
    if d_in != n_sc + d_sym:
        raise ValueError(f"Expected d_in = n_state + |Σ| = {n_sc + d_sym}, got {d_in}.")
    if str_tr is not None:
        for i in range(min(int(verify_count), len(str_tr))):
            verify_ar(dfa, [str_tr[i]], Xtr[i : i + 1], yst_tr[i : i + 1], yl_tr[i : i + 1], T)

    ds = ARITH.CarryAugmentedDataset(
        X_train=Xtr,
        y_sum_train=yst_tr,
        y_carry_train=yl_tr,
        X_val=Xva,
        y_sum_val=yst_va,
        y_carry_val=yl_va,
        X_test=Xte,
        y_sum_test=yst_te,
        y_carry_test=yl_te,
        num_sum_classes=int(n_sc),
        d_in=int(d_in),
        T=int(T),
        dataset_name=f"dfa_state_label::{dfa_spec}::T{T}",
    )
    meta = _Meta(dfa=dfa, d_sym=d_sym, start_state_index=int(get_dfa_start_state_index(dfa)))
    return ds, meta


def verify_ar(
    dfa: DFA,
    strings: Sequence[Sequence[str]],
    X: np.ndarray,
    y_state: np.ndarray,
    y_label: np.ndarray,
    T: int,
) -> None:
    X2, ys2, yl2, n_sc, _d_in = strings_to_autoregressive_tensors(dfa, list(strings), T)
    if X2.shape != X.shape or not np.allclose(X2, X, rtol=1e-5, atol=1e-6):
        raise ValueError("Autoregressive tensor mismatch vs strings.")
    if not np.array_equal(ys2, y_state) or not np.array_equal(yl2, y_label):
        raise ValueError("y_state / y_label mismatch.")


def _make_dfa_ood_tensors(
    dfa_spec: str,
    T_ood: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    X, ys, yl, n_sc, d_in, _, _ = make_dfa_autoregressive_dataset(
        dfa_spec, n_test, T_ood, seed=seed, balanced=True, return_strings=False
    )
    return X, ys, yl, n_sc, d_in


# ---------------------------------------------------------------------------#
# DFA autoregressive rollout (state one-hot, not add channel 2)
# ---------------------------------------------------------------------------#


def _build_cvx_features_for_eval(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if int(getattr(init_cfg, "K_parallel", 1)) > 1:
        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        _, _, d_test, _ = cvx_par._build_feature_map(x_train, x_val, x_test, pic, all_timesteps=True)
    else:
        _, _, d_test, _ = _build_feature_map(x_train, x_val, x_test, init_cfg, all_timesteps=True)
    return d_test


def ste_predict_dfa_state_label_autoregressive(
    model: ste_cts.CarryAugmentedSNN,
    x_seq: np.ndarray,
    *,
    num_sum_classes: int,
    d_in: int,
    start_state_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_sc = int(num_sum_classes)
    d_sym = int(d_in) - n_sc
    if d_sym < 1:
        raise ValueError("d_in - num_sum_classes must be |alphabet|.")
    device = next(model.parameters()).device
    n, T, dtot = x_seq.shape
    if dtot != d_in:
        raise ValueError(f"x_seq d_in {dtot} != {d_in}.")
    x_roll = x_seq.astype(np.float32, copy=True)
    e0 = np.zeros(n_sc, dtype=np.float32)
    e0[int(start_state_index)] = 1.0
    x_roll[:, 0, :n_sc] = e0
    sum_pred = np.zeros((n, T), dtype=np.int64)
    carry_pred = np.zeros((n, T), dtype=np.int64)
    b = int(num_sum_classes)
    model.eval()
    with torch.no_grad():
        for t in range(T):
            xt = torch.tensor(x_roll, dtype=torch.float32, device=device)
            sum_logits, carry_logits = model(xt)
            sum_t = sum_logits[:, t, :].detach().cpu().numpy()
            carry_t = carry_logits[:, t, :].detach().cpu().numpy()
            sum_pred[:, t] = ARITH._decode_sum(sum_t, b)
            carry_pred[:, t] = ARITH._decode_carry(carry_t)
            if t + 1 < T:
                x_roll[:, t + 1, :n_sc] = 0.0
                x_roll[np.arange(n, dtype=np.int64), t + 1, sum_pred[:, t]] = 1.0
    return sum_pred, carry_pred


def cvx_predict_dfa_state_label_autoregressive(
    *,
    w_sum: np.ndarray,
    w_carry: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_seq: np.ndarray,
    init_cfg: InitializationConfig,
    base: int,
    d_in: int,
    start_state_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_sc = int(base)
    d_sym = int(d_in) - n_sc
    if d_sym < 1:
        raise ValueError("invalid d_in vs base for DFA ar rollout.")
    n, T, dtot = x_seq.shape
    if dtot != d_in:
        raise ValueError("x_seq d_in mismatch.")
    x_roll = x_seq.astype(np.float32, copy=True)
    e0 = np.zeros(n_sc, dtype=np.float32)
    e0[int(start_state_index)] = 1.0
    x_roll[:, 0, :n_sc] = e0
    sum_pred = np.zeros((n, T), dtype=np.int64)
    carry_pred = np.zeros((n, T), dtype=np.int64)
    b = int(base)
    for t in range(T):
        d_test = _build_cvx_features_for_eval(
            x_train=x_train, x_val=x_val, x_test=x_roll, init_cfg=init_cfg
        )
        sum_scores = d_test @ w_sum
        carry_scores = d_test @ w_carry
        if b == 2:
            sum_scores_t = sum_scores.reshape(n, T)[:, t]
        else:
            sum_scores_t = sum_scores.reshape(n, T, b)[:, t, :]
        carry_scores_t = carry_scores.reshape(n, T)[:, t]
        sum_pred[:, t] = ARITH._decode_sum(sum_scores_t, b)
        carry_pred[:, t] = ARITH._decode_carry(carry_scores_t)
        if t + 1 < T:
            x_roll[:, t + 1, :n_sc] = 0.0
            x_roll[np.arange(n, dtype=np.int64), t + 1, sum_pred[:, t]] = 1.0
    return sum_pred, carry_pred


def _rename_dfa_keys(d: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        nk = k.replace("sum_", "state_").replace("carry_", "label_")
        out[nk] = v
    return out


def _print_dfa_hybrid_train_ok(*, base_seed: int, lambda_sum: float, stage_key: str) -> None:
    print(
        f"[dfa_hybrid] train_ok seed={int(base_seed)} lambda_sum={float(lambda_sum):.8g} stage={stage_key}",
        flush=True,
    )


def _print_dfa_hybrid_eval_ok(*, base_seed: int, lambda_sum: float, stage_key: str) -> None:
    print(
        f"[dfa_hybrid] eval_ok seed={int(base_seed)} lambda_sum={float(lambda_sum):.8g} stage={stage_key}",
        flush=True,
    )


# ---------------------------------------------------------------------------#
# eval (parallel to arith _eval_stage_single_mode)
# ---------------------------------------------------------------------------#


def _eval_dfa_single_mode(
    *,
    eval_mode: str,
    ds_fit: ARITH.CarryAugmentedDataset,
    ds_eval: ARITH.CarryAugmentedDataset,
    ste_model: Optional[ste_cts.CarryAugmentedSNN] = None,
    cvx_bundle: Optional[Dict[str, Any]] = None,
    ood_T_multipliers: Sequence[int],
    n_test_ood: int,
    dfa_spec: str,
    eval_seed: int,
    n_state_classes: int,
    d_in: int,
    start_state_index: int,
) -> Dict[str, Any]:
    if (ste_model is None) == (cvx_bundle is None):
        raise ValueError("Provide exactly one of ste_model or cvx_bundle.")
    b = int(n_state_classes)

    def _tf_metrics_ste(x: np.ndarray, y_s: np.ndarray, y_c: np.ndarray) -> Dict[str, float]:
        sp, cp = ste_cts.ste_predict_sum_carry(ste_model, x)  # type: ignore[arg-type]
        return _rename_dfa_keys(ste_cts.carry_teacher_forcing_token_metrics(sp, cp, y_s, y_c))

    def _tf_metrics_cvx(x: np.ndarray, y_s: np.ndarray, y_c: np.ndarray) -> Dict[str, float]:
        sp, cp = cvx_cts.cvx_predict_two_head_ood(
            w_sum=cvx_bundle["sum_weights"],  # type: ignore[index]
            w_carry=cvx_bundle["carry_weights"],  # type: ignore[index]
            x_train=ds_fit.X_train,
            x_val=ds_fit.X_val,
            x_ood=x,
            init_cfg=cvx_bundle["init_cfg"],  # type: ignore[index]
            base=b,
        )
        return _rename_dfa_keys(ste_cts.carry_teacher_forcing_token_metrics(sp, cp, y_s, y_c))

    def _ar_metrics_ste(x: np.ndarray, y_s: np.ndarray, y_c: np.ndarray) -> Dict[str, float]:
        sp, cp = ste_predict_dfa_state_label_autoregressive(
            ste_model,  # type: ignore[arg-type]
            x,
            num_sum_classes=b,
            d_in=int(d_in),
            start_state_index=int(start_state_index),
        )
        return _rename_dfa_keys(ARITH.carry_autoregressive_metrics(sp, cp, y_s, y_c))

    def _ar_metrics_cvx(x: np.ndarray, y_s: np.ndarray, y_c: np.ndarray) -> Dict[str, float]:
        sp, cp = cvx_predict_dfa_state_label_autoregressive(
            w_sum=cvx_bundle["sum_weights"],  # type: ignore[index]
            w_carry=cvx_bundle["carry_weights"],  # type: ignore[index]
            x_train=ds_fit.X_train,
            x_val=ds_fit.X_val,
            x_seq=x,
            init_cfg=cvx_bundle["init_cfg"],  # type: ignore[index]
            base=b,
            d_in=int(d_in),
            start_state_index=int(start_state_index),
        )
        return _rename_dfa_keys(ARITH.carry_autoregressive_metrics(sp, cp, y_s, y_c))

    if str(eval_mode) == "teacher_forcing":
        metric_fn = _tf_metrics_ste if ste_model is not None else _tf_metrics_cvx
    else:
        metric_fn = _ar_metrics_ste if ste_model is not None else _ar_metrics_cvx

    ood: Dict[str, Any] = {}
    t_train = int(ds_eval.T)
    for m in ood_T_multipliers:
        T_ood = int(m) * t_train
        xo, yso, yco, n_sc, di = _make_dfa_ood_tensors(
            dfa_spec, T_ood, int(n_test_ood), int(eval_seed) + 10_000 + int(m) * 97
        )
        if n_sc != n_state_classes or di != d_in:
            raise ValueError(f"OOD layout mismatch: T_ood={T_ood} n_sc={n_sc} d_in={di}")
        key = f"Ttrain{t_train}_Tood{T_ood}_x{int(m)}"
        ood[key] = {
            "T_train": t_train,
            "T_ood": T_ood,
            "multiplier": int(m),
            "n_test": int(n_test_ood),
            "metrics": metric_fn(xo, yso, yco),
        }
    return {
        "id_metrics": metric_fn(ds_eval.X_test, ds_eval.y_sum_test, ds_eval.y_carry_test),
        "ood_eval": ood,
    }


def _eval_dfa_all_modes(**kwargs: Any) -> Dict[str, Any]:
    k_tf = {k: v for k, v in kwargs.items() if k != "eval_mode"}
    return {
        "teacher_forcing": _eval_dfa_single_mode(eval_mode="teacher_forcing", **k_tf),
        "autoregressive": _eval_dfa_single_mode(eval_mode="autoregressive", **k_tf),
    }


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_dfa_eval_mode_payload(mode_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Like addition bench aggregate, but OOD blocks use DFA keys (T_train, T_ood, …) not n_digits."""
    if not mode_payloads:
        return {}
    id_metrics = ARITH._float_dict_mean_std([sp["id_metrics"] for sp in mode_payloads])
    ood0 = mode_payloads[0]["ood_eval"]
    ood_agg: Dict[str, Any] = {}
    for key in ood0:
        blocks = [sp["ood_eval"][key] for sp in mode_payloads]
        b0 = blocks[0]
        meta = {k: b0[k] for k in b0 if k != "metrics"}
        ood_agg[key] = {**meta, "metrics": ARITH._float_dict_mean_std([b["metrics"] for b in blocks])}
    return {"id_metrics": id_metrics, "ood_eval": ood_agg}


def _aggregate_dfa_stage(stage_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not stage_payloads:
        return {}
    return {
        "teacher_forcing": _aggregate_dfa_eval_mode_payload([sp["teacher_forcing"] for sp in stage_payloads]),
        "autoregressive": _aggregate_dfa_eval_mode_payload([sp["autoregressive"] for sp in stage_payloads]),
    }


def _aggregate_dfa_lambda_sweep(seed_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not seed_payloads:
        return []
    base_grid = [float(e["lambda_sum"]) for e in seed_payloads[0]["lambda_sweep"]]
    for sp in seed_payloads[1:]:
        if [float(e["lambda_sum"]) for e in sp["lambda_sweep"]] != base_grid:
            raise ValueError("lambda_sum sweep order differs across seeds")
    out: List[Dict[str, Any]] = []
    for idx, _ in enumerate(base_grid):
        entries = [sp["lambda_sweep"][idx] for sp in seed_payloads]
        out.append(
            {
                "lambda_sum": float(base_grid[idx]),
                "lambda_carry_fixed": float(entries[0].get("lambda_carry", 0.0)),
                "ste_pretrain": _aggregate_dfa_stage([e["ste_pretrain"] for e in entries]),
                "cvx_from_ste_pretrain": _aggregate_dfa_stage([e["cvx_from_ste_pretrain"] for e in entries]),
                "ste_finetune_from_ste_pretrain_new_train": _aggregate_dfa_stage(
                    [e["ste_finetune_from_ste_pretrain_new_train"] for e in entries]
                ),
                "cvx_pretrain": _aggregate_dfa_stage([e["cvx_pretrain"] for e in entries]),
                "ste_finetune_from_cvx_pretrain_new_train": _aggregate_dfa_stage(
                    [e["ste_finetune_from_cvx_pretrain_new_train"] for e in entries]
                ),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DFA autoregressive hybrid (same 5 stages as add carry bench). "
        "lambda_sum scales the sum/state head; lambda_carry scales the carry/label head. "
        "Sweep is over lambda_sum (state); lambda_carry is fixed."
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--dfa_spec", type=str, default="tomita_6")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--n_train_pre", type=int, default=2304)
    ap.add_argument("--n_val_pre", type=int, default=512)
    ap.add_argument("--n_train_ft", type=int, default=2304)
    ap.add_argument("--n_val_ft", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--n_test_ood", type=int, default=1024)
    ap.add_argument("--ood_T_multipliers", type=int, nargs="*", default=[2,5,10])
    ap.add_argument("--verify_samples", type=int, default=3)
    ap.add_argument("--finetune_seed_offset", type=int, default=1000)
    ap.add_argument("--eval_seed_offset", type=int, default=2000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="If >0, cap n_train for both pretrain and finetune (after --debug). 0 = no cap.",
    )
    ap.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="If >0, cap n_val for pretrain and finetune. 0 = no cap.",
    )
    ap.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="If >0, cap n_test and n_test_ood. 0 = no cap.",
    )

    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--P_rec", type=int, default=256)
    ap.add_argument("--P_last", type=int, default=512)
    ap.add_argument("--K_parallel", type=int, default=2)
    ap.add_argument("--beta_leak", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--ste_last_layer_readout", choices=["membrane", "spike"], default="spike")
    ap.add_argument("--cvx_last_layer_readout", choices=["membrane", "spike"], default="spike")
    ap.add_argument("--optimizer_name", choices=["adam", "sgd"], default="adam")

    ap.add_argument("--ste_pretrain_epochs", type=int, default=100)
    ap.add_argument("--ste_finetune_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=-1)
    ap.add_argument(
        "--lambda_carry",
        type=float,
        default=1.0,
        help="Fixed weight on the carry head = per-step accept (hinge). Not swept in this script.",
    )
    ap.add_argument(
        "--lambda_sum_grid",
        type=float,
        nargs="*",
        default=[0.125, 0.75, 1.0, 4.0],
        help="Swept weight on the sum head = next-state (CE). This is the state / lambda_sum grid.",
    )
    ap.add_argument("--ste_lr_grid", type=float, nargs="*", default=list(LR_GRID_DEFAULT))
    ap.add_argument("--ste_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_beta_grid", type=float, nargs="*", default=list(BETA_GRID_DEFAULT))
    ap.add_argument("--cvx_bias_grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--cvx_device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--ste_sum_loss", choices=["auto", "hinge", "ce", "hinge_ovr"], default="ce")
    ap.add_argument("--ste_carry_loss", choices=["auto", "hinge", "ce"], default="hinge")
    ap.add_argument("--cvx_sum_loss", choices=["auto", "hinge", "ce", "hinge_ovr"], default="ce")
    ap.add_argument("--cvx_carry_loss", choices=["auto", "hinge", "ce"], default="hinge")
    ap.add_argument(
        "--tf_objective", choices=["joint", "lambda_weighted", "mean_pair", "lambda_normalized"], default="joint"
    )
    ap.add_argument("--ste_time_loss", choices=["uniform", "ramp"], default="ramp")
    ap.add_argument("--cvx_time_loss", choices=["uniform", "ramp"], default="ramp")

    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    if not args.lambda_sum_grid:
        raise ValueError("lambda_sum_grid must be non-empty.")
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
        args.ste_lr_grid = [0.01]
        args.ste_beta_grid = [0.0]
        args.cvx_beta_grid = [0.0]
        args.lambda_sum_grid = [1.0]

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
            raise ValueError(f"Invalid {name}={n}; increase dataset sizes or relax max_*_samples caps.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root).strip():
        out_root = Path(str(args.out_root)).expanduser().resolve()
    else:
        out_root = Path.cwd() / "sweep_results" / f"dfa_hybrid_{args.dfa_spec}_T{t_dim}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    cvx_device = ARITH._resolve_cvx_device(str(args.cvx_device))

    config_dump = {
        "seeds": list(args.seeds),
        "dfa_spec": str(args.dfa_spec),
        "T": t_dim,
        "n_train_pre": n_tr,
        "n_val_pre": n_vap,
        "n_train_ft": n_trf,
        "n_val_ft": n_vaf,
        "n_test": n_te,
        "n_test_ood": n_ood,
        "ood_T_multipliers": list(int(x) for x in args.ood_T_multipliers),
        "finetune_seed_offset": int(args.finetune_seed_offset),
        "eval_seed_offset": int(args.eval_seed_offset),
        "L": int(args.L),
        "P_rec": int(args.P_rec),
        "P_last": int(args.P_last),
        "K_parallel": int(args.K_parallel),
        "ste_last_layer_readout": str(args.ste_last_layer_readout),
        "cvx_last_layer_readout": str(args.cvx_last_layer_readout),
        "optimizer_name": str(args.optimizer_name),
        "ste_pretrain_epochs": s_pre,
        "ste_finetune_epochs": s_ft,
        "lambda_carry_fixed_label_head": float(args.lambda_carry),
        "lambda_sum_grid_swept_state_head": [float(x) for x in args.lambda_sum_grid],
        "ste_lr_grid": list(args.ste_lr_grid),
        "ste_beta_grid": list(args.ste_beta_grid),
        "cvx_beta_grid": list(args.cvx_beta_grid),
        "cvx_bias_grid": list(args.cvx_bias_grid),
        "ste_sum_loss": str(args.ste_sum_loss),
        "ste_carry_loss": str(args.ste_carry_loss),
        "cvx_sum_loss": str(args.cvx_sum_loss),
        "cvx_carry_loss": str(args.cvx_carry_loss),
        "tf_objective": str(args.tf_objective),
        "ste_time_loss": str(args.ste_time_loss),
        "cvx_time_loss": str(args.cvx_time_loss),
        "max_train_samples": int(mtr),
        "max_val_samples": int(mva),
        "max_test_samples": int(mte),
        "naming": "sum head = state (y_sum); carry head = label (y_carry). "
        "Swept: lambda_sum (state). Fixed: lambda_carry (label). "
        "Differs from addition bench, which sweeps the carry slot.",
        "evaluation_modes": ["teacher_forcing", "autoregressive"],
        "stages": [
            "ste_pretrain",
            "cvx_from_ste_pretrain",
            "ste_finetune_from_ste_pretrain_new_train",
            "cvx_pretrain",
            "ste_finetune_from_cvx_pretrain_new_train",
        ],
        "debug": bool(args.debug),
    }
    (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n")

    seed_payloads: List[Dict[str, Any]] = []

    for base_seed in args.seeds:
        pre_seed = int(base_seed)
        ft_seed = int(base_seed) + int(args.finetune_seed_offset)
        eval_seed = int(base_seed) + int(args.eval_seed_offset)
        ds_pre, meta_pre = build_dfa_carry_augmented_dataset_from_seeds(
            dfa_spec=str(args.dfa_spec),
            T=t_dim,
            n_train=n_tr,
            n_val=n_vap,
            n_test=n_te,
            seed_train=pre_seed + 11,
            seed_val=pre_seed + 29,
            seed_test=eval_seed + 47,
            verify_count=int(args.verify_samples),
        )
        ds_ft, _meta_ft = build_dfa_carry_augmented_dataset_from_seeds(
            dfa_spec=str(args.dfa_spec),
            T=t_dim,
            n_train=n_trf,
            n_val=n_vaf,
            n_test=n_te,
            seed_train=ft_seed + 11,
            seed_val=ft_seed + 29,
            seed_test=eval_seed + 47,
            verify_count=0,
        )
        b_sc = int(ds_pre.num_sum_classes)
        d_in = int(ds_pre.d_in)
        st_0 = int(meta_pre.start_state_index)

        lambda_sweep_payloads: List[Dict[str, Any]] = []
        n_ls = list(float(x) for x in args.lambda_sum_grid)
        for li, lambda_sum in enumerate(n_ls):
            _set_seed(pre_seed)
            print(
                f"[dfa_hybrid] start seed={int(base_seed)} "
                f"lambda_sum={float(lambda_sum):.8g} ({li + 1}/{len(n_ls)} for this seed)",
                flush=True,
            )
            ste_pre_model, ste_pre_sel, _ = ARITH.ste_sweep_and_train_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=s_pre,
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=pre_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(lambda_sum),
                lambda_carry=float(args.lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=None,
            )
            ste_pre_weights = ARITH._extract_carry_weight_list(ste_pre_model)
            _print_dfa_hybrid_train_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="ste_pretrain"
            )

            cvx_from_ste_bundle, cvx_from_ste_sel, cvx_from_ste_tf_test = ARITH.cvx_fit_shared_two_head_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                seed=pre_seed,
                beta_grid=args.cvx_beta_grid,
                bias_grid=args.cvx_bias_grid,
                lambda_sum=float(lambda_sum),
                lambda_carry=float(args.lambda_carry),
                cvx_device=cvx_device,
                cvx_sum_loss=str(args.cvx_sum_loss),
                cvx_carry_loss=str(args.cvx_carry_loss),
                cvx_time_loss=str(args.cvx_time_loss),
                init_mode="pretraining",
                pretrained_weights=ste_pre_weights,
            )
            _print_dfa_hybrid_train_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="cvx_from_ste_pretrain"
            )

            ste_ft_from_ste_model, ste_ft_from_ste_sel, _ = ARITH.ste_sweep_and_train_init(
                ds=ds_ft,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=s_ft,
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=ft_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(lambda_sum),
                lambda_carry=float(args.lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=ste_pre_weights,
            )
            _print_dfa_hybrid_train_ok(
                base_seed=base_seed,
                lambda_sum=lambda_sum,
                stage_key="ste_finetune_from_ste_pretrain_new_train",
            )

            cvx_pre_bundle, cvx_pre_sel, cvx_pre_tf_test = ARITH.cvx_fit_shared_two_head_init(
                ds=ds_pre,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                cvx_last_layer_readout=str(args.cvx_last_layer_readout),
                seed=pre_seed,
                beta_grid=args.cvx_beta_grid,
                bias_grid=args.cvx_bias_grid,
                lambda_sum=float(lambda_sum),
                lambda_carry=float(args.lambda_carry),
                cvx_device=cvx_device,
                cvx_sum_loss=str(args.cvx_sum_loss),
                cvx_carry_loss=str(args.cvx_carry_loss),
                cvx_time_loss=str(args.cvx_time_loss),
                init_mode="gaussian",
                pretrained_weights=None,
            )
            cvx_pre_weights = ARITH._cvx_bundle_to_carry_weights(
                bundle=cvx_pre_bundle,
                d_in=d_in,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                base=b_sc,
            )
            _print_dfa_hybrid_train_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="cvx_pretrain"
            )
            ste_ft_from_cvx_model, ste_ft_from_cvx_sel, _ = ARITH.ste_sweep_and_train_init(
                ds=ds_ft,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                ste_last_layer_readout=str(args.ste_last_layer_readout),
                ste_epochs=s_ft,
                batch_size=int(args.batch_size),
                optimizer_name=str(args.optimizer_name),
                beta_leak=float(args.beta_leak),
                threshold=float(args.threshold),
                seed=ft_seed,
                ste_lr_grid=args.ste_lr_grid,
                ste_beta_grid=args.ste_beta_grid,
                lambda_sum=float(lambda_sum),
                lambda_carry=float(args.lambda_carry),
                ste_sum_loss=str(args.ste_sum_loss),
                ste_carry_loss=str(args.ste_carry_loss),
                tf_objective=str(args.tf_objective),
                ste_time_loss=str(args.ste_time_loss),
                pretrained_weights=cvx_pre_weights,
            )
            _print_dfa_hybrid_train_ok(
                base_seed=base_seed,
                lambda_sum=lambda_sum,
                stage_key="ste_finetune_from_cvx_pretrain_new_train",
            )

            com = dict(
                ood_T_multipliers=args.ood_T_multipliers,
                n_test_ood=int(n_ood),
                dfa_spec=str(args.dfa_spec),
                eval_seed=eval_seed,
                n_state_classes=b_sc,
                d_in=d_in,
                start_state_index=st_0,
            )
            ste_pre_eval = _eval_dfa_all_modes(
                ds_fit=ds_pre, ds_eval=ds_pre, ste_model=ste_pre_model, **com
            )
            _print_dfa_hybrid_eval_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="ste_pretrain"
            )
            cvx_from_ste_eval = _eval_dfa_all_modes(
                ds_fit=ds_pre, ds_eval=ds_pre, cvx_bundle=cvx_from_ste_bundle, **com
            )
            _print_dfa_hybrid_eval_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="cvx_from_ste_pretrain"
            )
            ste_ft_from_ste_eval = _eval_dfa_all_modes(
                ds_fit=ds_ft, ds_eval=ds_ft, ste_model=ste_ft_from_ste_model, **com
            )
            _print_dfa_hybrid_eval_ok(
                base_seed=base_seed,
                lambda_sum=lambda_sum,
                stage_key="ste_finetune_from_ste_pretrain_new_train",
            )
            cvx_pre_eval = _eval_dfa_all_modes(
                ds_fit=ds_pre, ds_eval=ds_pre, cvx_bundle=cvx_pre_bundle, **com
            )
            _print_dfa_hybrid_eval_ok(
                base_seed=base_seed, lambda_sum=lambda_sum, stage_key="cvx_pretrain"
            )
            ste_ft_from_cvx_eval = _eval_dfa_all_modes(
                ds_fit=ds_ft, ds_eval=ds_ft, ste_model=ste_ft_from_cvx_model, **com
            )
            _print_dfa_hybrid_eval_ok(
                base_seed=base_seed,
                lambda_sum=lambda_sum,
                stage_key="ste_finetune_from_cvx_pretrain_new_train",
            )

            cvx_from_ste_tf = _rename_dfa_keys({str(k): float(v) for k, v in cvx_from_ste_tf_test.items()})
            cvx_pre_tf = _rename_dfa_keys({str(k): float(v) for k, v in cvx_pre_tf_test.items()})
            lambda_sweep_payloads.append(
                {
                    "lambda_sum": float(lambda_sum),
                    "lambda_carry": float(args.lambda_carry),
                    "ste_pretrain": {"selected_params": ste_pre_sel, **ste_pre_eval},
                    "cvx_from_ste_pretrain": {
                        "selected_params": cvx_from_ste_sel,
                        "teacher_forcing_pretrain_test_metrics": cvx_from_ste_tf,
                        "diagnostics": {
                            "primal_value": float(cvx_from_ste_bundle["primal_value"]),
                            "dual_value": float(cvx_from_ste_bundle["dual_value"]),
                            "gap": float(cvx_from_ste_bundle["gap"]),
                        },
                        **cvx_from_ste_eval,
                    },
                    "ste_finetune_from_ste_pretrain_new_train": {
                        "selected_params": ste_ft_from_ste_sel,
                        **ste_ft_from_ste_eval,
                    },
                    "cvx_pretrain": {
                        "selected_params": cvx_pre_sel,
                        "teacher_forcing_pretrain_test_metrics": cvx_pre_tf,
                        "diagnostics": {
                            "primal_value": float(cvx_pre_bundle["primal_value"]),
                            "dual_value": float(cvx_pre_bundle["dual_value"]),
                            "gap": float(cvx_pre_bundle["gap"]),
                        },
                        **cvx_pre_eval,
                    },
                    "ste_finetune_from_cvx_pretrain_new_train": {
                        "selected_params": ste_ft_from_cvx_sel,
                        **ste_ft_from_cvx_eval,
                    },
                }
            )
            print(
                f"[dfa_hybrid] finished seed={int(base_seed)} lambda_sum={float(lambda_sum):.8g} "
                f"(all 5 stages evaluated and payload appended)",
                flush=True,
            )

        seed_payload = {
            "seed": int(base_seed),
            "dfa_spec": str(args.dfa_spec),
            "T": t_dim,
            "d_in": d_in,
            "n_state_classes": b_sc,
            "meta": {"d_sym": meta_pre.d_sym, "start_state_index": st_0},
            "split_seeds": {
                "pretrain_train": pre_seed + 11,
                "pretrain_val": pre_seed + 29,
                "finetune_train": ft_seed + 11,
                "finetune_val": ft_seed + 29,
                "eval_test": eval_seed + 47,
            },
            "lambda_sweep": lambda_sweep_payloads,
        }
        seed_payloads.append(seed_payload)
        sdir = out_root / f"seed_{int(base_seed)}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "metrics.json").write_text(json.dumps(seed_payload, indent=2, default=str) + "\n")
        print(
            f"[dfa_hybrid] seed {int(base_seed)} complete: "
            f"wrote {sdir / 'metrics.json'} ({len(lambda_sweep_payloads)} lambda values)",
            flush=True,
        )

    aggregate = {
        "n_seeds": int(len(seed_payloads)),
        "lambda_sweep": _aggregate_dfa_lambda_sweep(seed_payloads),
    }
    root_payload = {
        "run_config": config_dump,
        "out_root": str(out_root),
        "seeds": seed_payloads,
        "aggregate": aggregate,
    }
    (out_root / "metrics.json").write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    oj = str(args.output_json).strip()
    if oj:
        Path(oj).expanduser().write_text(json.dumps(root_payload, indent=2, default=str) + "\n")
    print(json.dumps({"aggregate": aggregate, "out_root": str(out_root)}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
