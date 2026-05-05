#!/usr/bin/env python3
"""
Base-b addition (``--arith_base`` in 2, 3, 5, 7, 10): train on fixed ``n_digits_train``, sweep
STE (lr × beta) and CVX (beta × bias; lr only for ``--cvx_method=sgd``), then token/seq accuracy on
longer OOD ``n_digits``. **Default reporting** always includes
:func:`add_rule_context_diagnostics` on in-distribution ``X_test`` and on each OOD length: a
context-conditioned local-rule analysis over the observed triples
``(a_t, b_t, c_t^{in})`` on the **sum** positions only, alongside full-sequence
``_detailed_arithmetic_metrics`` and optional ``--ood_block_size`` timeline blocks. The diagnostics
report per-context accuracy, per-block context tables, and a blockwise invariance summary to help
distinguish local rule learning from positional memorization. Decoded operands from ``x`` must match
``y`` (reconstruction check).

By default the SNN uses **membrane** readout on the last hidden layer (``--ste_last_layer_readout``)
and the convex LIF feature path uses **spike** readout (``--cvx_last_layer_readout``).

By default writes metric JSON under the run directory (see ``--no_save_metrics``) and does **not**
write ``.npz`` weight files; pass ``--save_weights`` to export ``ste_best.npz`` / ``cvx_best.npz``.
Trained parameters are always **in memory** on the solve results: ``ste_res.model`` (PyTorch) and
``cvx_res.trained_model["weights"]`` (``numpy``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    from data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        _build_split,
        _stack,
        generate_samples_for_op_base_seq,
        load_arithmetic_dataset,
        verify_sample_seq,
    )
    from fine_tune import _extract_weight_list
    from solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from solvers import cvx_parallel_Solve as cvx_par
    from solvers import ste_parallel_Solve as ste_par
    from solvers.cvx_solve import InitializationConfig, SolveConfig, _build_feature_map, cvx_solve
    from solvers.ste_solve import SNNBaselineSeq, SteModelConfig, SteSolveConfig, ste_solve
else:
    from .data_loaders.arithmetic_data_loader import (
        SUPPORTED_BASES,
        _build_split,
        _stack,
        generate_samples_for_op_base_seq,
        load_arithmetic_dataset,
        verify_sample_seq,
    )
    from .fine_tune import _extract_weight_list
    from .solver_grids import BETA_GRID_DEFAULT, BIAS_GRID_DEFAULT, LR_GRID_DEFAULT, cvx_lr_sweep_values
    from .solvers import cvx_parallel_Solve as cvx_par
    from .solvers import ste_parallel_Solve as ste_par
    from .solvers.cvx_solve import InitializationConfig, SolveConfig, _build_feature_map, cvx_solve
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
            raise ValueError("Requested cvx_device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("Requested cvx_device=mps but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown cvx_device={device_name!r}")


def _add_timesteps(n_digits: int) -> int:
    return int(n_digits) + 1


def _make_test_xy(
    *,
    op: str,
    base: int,
    n_digits: int,
    n_samples: int,
    seed: int,
    verify_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = generate_samples_for_op_base_seq(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_samples,
        seed=seed,
    )
    vc = min(verify_count, len(samples))
    for i in range(vc):
        verify_sample_seq(samples[i])
    return _stack(samples, base)


def _mask_carry_inputs(x: np.ndarray) -> np.ndarray:
    """Zero out channel 2 (carry-in) of input tensors with shape (N, T, d_in).

    Channels 0/1 are operand digits; channel 2 is the **true carry-in** at column t
    (and at the final extra row, the **MSD carry-out** target). Masking forces the
    model to track the carry chain itself rather than reading it from the input.
    Channel 2 at the final timestep also makes the MSD carry-out target trivially
    copyable from the input — masking removes that, too.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape (N, T, d_in), got {x.shape}.")
    if x.shape[2] < 3:
        raise ValueError(
            f"Expected d_in>=3 to mask carry-in channel, got d_in={x.shape[2]}."
        )
    out = np.array(x, copy=True)
    out[:, :, 2] = 0.0
    return out


def _build_lengthgen_split(
    *,
    op: str,
    base: int,
    n_digits: int,
    n_samples: int,
    seed: int,
    verify_count: int,
    mask_carry: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap the data loader's ``_build_split`` and apply optional carry masking.

    Returns ``(x_for_model, x_unmasked, y)``:
    - ``x_for_model`` has channel 2 zeroed iff ``mask_carry`` is True; that's what gets
      fed to STE/CVX training and prediction.
    - ``x_unmasked`` is always the raw 3-channel input from the data builder; that's what
      ``add_rule_context_diagnostics`` needs (to read the **true** carry-in column at t=0
      and reconstruct the per-step carry chain for the per-context analysis). Keeping
      both lets us mask the model's view without breaking the diagnostic.

    When ``mask_carry`` is False the two arrays are the same (no extra memory cost: same
    underlying buffer).
    """
    x_unmasked, y = _build_split(
        op=op,
        base=base,
        n_digits=n_digits,
        n_samples=n_samples,
        seed=seed,
        verify_count=verify_count,
    )
    x_for_model = _mask_carry_inputs(x_unmasked) if mask_carry else x_unmasked
    return x_for_model, x_unmasked, y


def _detailed_arithmetic_metrics(
    preds: np.ndarray,
    y: np.ndarray,
    *,
    block_size: int = 5,
) -> Dict[str, Any]:
    """
    Global token/seq accuracy, plus **non-overlapping** per-block stats along time (OOD, etc.).

    For add, ``y`` has length ``n_digits + 1`` (0-based row indices in ``target_tokens``):
    timesteps ``0..n_digits-1`` are per-column **sum digits**; the last is **MSD carry-out**.
    With ``block_size=5`` and e.g. ``n_digits=10`` (11 steps, 1-based T=1..11): blocks are
    T=1..5, T=6..10, and T=11 (often a single final-carry token in the last block).
    """
    if int(block_size) < 1:
        raise ValueError(f"block_size must be >=1, got {block_size}.")
    if preds.shape != y.shape:
        raise ValueError(f"preds shape {preds.shape} != y shape {y.shape}")
    n, steps = y.shape
    last_t0 = int(steps) - 1
    match = preds == y
    token_acc = float(match.mean())
    seq_acc = float(match.all(axis=1).mean())
    wrong_rows = ~match.all(axis=1)
    n_wrong = int(wrong_rows.sum())
    first_err: List[int] = []
    for i in range(n):
        if wrong_rows[i]:
            t = int(np.argmax(~match[i]))
            first_err.append(t)
    mean_first_wrong_timestep = float(np.mean(first_err)) if first_err else float("nan")
    std_first_wrong_timestep = float(np.std(first_err)) if len(first_err) > 1 else float("nan")

    bs = int(block_size)
    per_block: List[Dict[str, Any]] = []
    start = 0
    block_idx = 0
    while start < steps:
        end = min(start + bs, steps)
        block_match = match[:, start:end]
        t_acc_b = float(block_match.mean())
        s_acc_b = float(block_match.all(axis=1).mean())
        per_block.append(
            {
                "block_index": block_idx,
                "block_size": bs,
                "n_tokens_in_block": int(end - start),
                "timestep_start_0based": int(start),
                "timestep_end_0based_inclusive": int(end - 1),
                "timestep_start_1based": int(start + 1),
                "timestep_end_1based_inclusive": int(end),
                "includes_msd_carry_out_token": bool(last_t0 >= start and last_t0 < end),
                "token_acc": t_acc_b,
                "seq_acc": s_acc_b,
                "token_loss": float(1.0 - t_acc_b),
                "seq_loss": float(1.0 - s_acc_b),
            }
        )
        block_idx += 1
        start = end

    return {
        "n_samples": n,
        "n_timesteps": steps,
        "block_size": bs,
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "n_sequences_with_any_error": n_wrong,
        "mean_first_wrong_timestep_among_wrong": mean_first_wrong_timestep,
        "std_first_wrong_timestep_among_wrong": std_first_wrong_timestep,
        "per_five_timestep_blocks": per_block,
    }


def _dec_x_digit(v: float, b: int) -> int:
    b1 = float(b - 1) if b > 1 else 1.0
    d = int(np.round(float(v) * b1))
    if d < 0 or d > b - 1:
        raise ValueError(f"Decoded digit {d} not in [0, {b - 1}].")
    return d


def _add_y_rebuild_matches_global(x: np.ndarray, y: np.ndarray, b: int, n_d: int) -> None:
    """Raises if batch ``y`` is not a column add on operands decoded from ``x`` (debug / consistency)."""
    n, st = y.shape
    if st != n_d + 1:
        raise ValueError(f"Expected y shape [*, {n_d + 1}], got {y.shape}.")
    for i in range(n):
        a = [_dec_x_digit(float(x[i, t, 0]), b) for t in range(n_d)]
        b_ = [_dec_x_digit(float(x[i, t, 1]), b) for t in range(n_d)]
        c0 = _dec_x_digit(float(x[i, 0, 2]), b)
        c = c0
        for t in range(n_d):
            s = (a[t] + b_[t] + c) % b
            if s != y[i, t]:
                raise ValueError(
                    f"Reconstruction mismatch at i={i} t={t}: y={y[i, t]!r} != column_add={s!r}."
                )
            c = (a[t] + b_[t] + c) // b
        if c != y[i, n_d]:
            raise ValueError(
                f"Reconstruction final carry mismatch at i={i}: y={y[i, n_d]!r} != {c!r}."
            )


def _decode_add_columns(x: np.ndarray, base: int, n_digits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode operand/carry-in columns from normalized ``x`` for the first ``n_digits`` sum positions."""
    b = int(base)
    n_d = int(n_digits)
    if n_d < 1:
        raise ValueError(f"n_digits must be >=1, got {n_d}.")
    scale = float(b - 1) if b > 1 else 1.0
    x_sum = x[:, :n_d, :]
    a = np.rint(x_sum[:, :, 0] * scale).astype(np.int64)
    b_ = np.rint(x_sum[:, :, 1] * scale).astype(np.int64)
    c_in = np.rint(x_sum[:, :, 2] * scale).astype(np.int64)
    return a, b_, c_in


def _ctx_key_to_payload(ctx: Tuple[int, int, int], base: int) -> Dict[str, Any]:
    a, b_, c_in = (int(ctx[0]), int(ctx[1]), int(ctx[2]))
    base_i = int(base)
    return {
        "context": [a, b_, c_in],
        "context_str": f"({a},{b_},{c_in})",
        "expected_sum": int((a + b_ + c_in) % base_i),
        "expected_carry_out": int((a + b_ + c_in) // base_i),
    }


def add_rule_context_diagnostics(
    pred: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    base: int,
    n_digits: int,
    *,
    block_size: int = 5,
    min_count_per_context: int = 10,
) -> Dict[str, Any]:
    """
    Context-conditioned diagnostics for the local addition rule on **sum** positions only.

    For each observed local triple ``(a_t, b_t, c_t^{in})``, report overall token accuracy,
    blockwise token accuracy, and an invariance summary measuring how much the same context's
    accuracy changes across blocks. This helps distinguish local rule learning from positional
    memorization while staying in the current full-sequence evaluation regime.
    """
    b = int(base)
    n_d = int(n_digits)
    bs = int(block_size)
    if bs < 1:
        raise ValueError(f"block_size must be >=1, got {bs}.")
    if pred.shape != y.shape:
        raise ValueError(f"pred shape {pred.shape} != y shape {y.shape}")
    n, steps = y.shape
    if steps != n_d + 1:
        raise ValueError(f"add: steps must be n_digits+1={n_d + 1}, got {steps}.")
    _add_y_rebuild_matches_global(x, y, b, n_d)

    a, b_, c_in = _decode_add_columns(x, b, n_d)
    sum_true = y[:, :n_d].astype(np.int64)
    sum_pred = pred[:, :n_d].astype(np.int64)
    sum_match = (sum_pred == sum_true)
    carry_out_true = ((a + b_ + c_in) // b).astype(np.int64)
    final_carry_acc = float((pred[:, n_d].astype(np.int64) == y[:, n_d].astype(np.int64)).mean())

    ctx_overall: Dict[Tuple[int, int, int], Dict[str, int]] = {}
    ctx_by_block: Dict[int, Dict[Tuple[int, int, int], Dict[str, int]]] = {}
    n_blocks = (n_d + bs - 1) // bs
    for blk in range(n_blocks):
        ctx_by_block[blk] = {}

    for i in range(n):
        for t in range(n_d):
            ctx = (int(a[i, t]), int(b_[i, t]), int(c_in[i, t]))
            ok = int(sum_match[i, t])
            rec = ctx_overall.setdefault(ctx, {"count": 0, "correct": 0})
            rec["count"] += 1
            rec["correct"] += ok
            blk = int(t // bs)
            brec = ctx_by_block[blk].setdefault(ctx, {"count": 0, "correct": 0})
            brec["count"] += 1
            brec["correct"] += ok

    per_context_overall: List[Dict[str, Any]] = []
    for ctx in sorted(ctx_overall.keys()):
        rec = ctx_overall[ctx]
        base_payload = _ctx_key_to_payload(ctx, b)
        per_context_overall.append(
            {
                **base_payload,
                "count": int(rec["count"]),
                "token_acc": float(rec["correct"] / max(1, rec["count"])),
                "token_loss": float(1.0 - (rec["correct"] / max(1, rec["count"]))),
            }
        )

    per_block_context_metrics: List[Dict[str, Any]] = []
    ctx_gaps: List[Dict[str, Any]] = []
    for blk in range(n_blocks):
        start = int(blk * bs)
        end = int(min(start + bs, n_d))
        blk_match = sum_match[:, start:end]
        blk_payload: Dict[str, Any] = {
            "block_index": blk,
            "n_tokens_in_block": int(end - start),
            "timestep_start_0based": start,
            "timestep_end_0based_inclusive": int(end - 1),
            "timestep_start_1based": int(start + 1),
            "timestep_end_1based_inclusive": int(end),
            "token_acc": float(blk_match.mean()),
            "seq_acc": float(blk_match.all(axis=1).mean()),
            "contexts": [],
        }
        for ctx in sorted(ctx_by_block[blk].keys()):
            rec = ctx_by_block[blk][ctx]
            acc = rec["correct"] / max(1, rec["count"])
            blk_payload["contexts"].append(
                {
                    **_ctx_key_to_payload(ctx, b),
                    "count": int(rec["count"]),
                    "token_acc": float(acc),
                    "token_loss": float(1.0 - acc),
                }
            )
        per_block_context_metrics.append(blk_payload)

    for ctx in sorted(ctx_overall.keys()):
        valid: List[Tuple[int, int, float]] = []
        total_valid_count = 0
        for blk in range(n_blocks):
            rec = ctx_by_block[blk].get(ctx)
            if rec is None:
                continue
            if int(rec["count"]) < int(min_count_per_context):
                continue
            acc = rec["correct"] / max(1, rec["count"])
            valid.append((blk, int(rec["count"]), float(acc)))
            total_valid_count += int(rec["count"])
        if len(valid) >= 2:
            accs = [x[2] for x in valid]
            gap = float(max(accs) - min(accs))
            ctx_gaps.append(
                {
                    **_ctx_key_to_payload(ctx, b),
                    "n_valid_blocks": int(len(valid)),
                    "total_count_in_valid_blocks": int(total_valid_count),
                    "max_gap": gap,
                    "block_accs": [
                        {"block_index": int(blk), "count": int(cnt), "token_acc": float(acc)}
                        for blk, cnt, acc in valid
                    ],
                }
            )

    if ctx_gaps:
        worst = max(ctx_gaps, key=lambda z: float(z["max_gap"]))
        weight_sum = float(sum(int(z["total_count_in_valid_blocks"]) for z in ctx_gaps))
        weighted_mean_gap = float(
            sum(float(z["max_gap"]) * int(z["total_count_in_valid_blocks"]) for z in ctx_gaps) / max(1.0, weight_sum)
        )
        max_gap = float(worst["max_gap"])
    else:
        worst = None
        weighted_mean_gap = float("nan")
        max_gap = float("nan")

    return {
        "n_digits": int(n_d),
        "n_samples": int(n),
        "block_size": int(bs),
        "sum_positions_only": True,
        "n_contexts_observed": int(len(per_context_overall)),
        "sum_token_acc_over_sum_positions": float(sum_match.mean()),
        "sum_seq_acc_over_sum_positions": float(sum_match.all(axis=1).mean()),
        "final_carry_token_acc": final_carry_acc,
        "per_context_overall": per_context_overall,
        "per_block_context_metrics": per_block_context_metrics,
        "invariance_summary": {
            "min_count_per_context": int(min_count_per_context),
            "n_blocks_over_sum_positions": int(n_blocks),
            "n_contexts_with_multi_block_min_count": int(len(ctx_gaps)),
            "weighted_mean_gap": weighted_mean_gap,
            "max_gap": max_gap,
            "worst_context_by_gap": worst,
        },
    }


def _print_rule_context_summary(prefix: str, diag: Dict[str, Any]) -> None:
    inv = diag["invariance_summary"]
    print(
        f"{prefix} sum_token_acc={diag['sum_token_acc_over_sum_positions']:.4f} "
        f"sum_seq_acc={diag['sum_seq_acc_over_sum_positions']:.4f} "
        f"final_carry_acc={diag['final_carry_token_acc']:.4f} "
        f"contexts={diag['n_contexts_observed']} "
        f"ctx_multi_block={inv['n_contexts_with_multi_block_min_count']} "
        f"weighted_gap={inv['weighted_mean_gap']:.4f} max_gap={inv['max_gap']:.4f}",
        flush=True,
    )
    worst = inv.get("worst_context_by_gap")
    if worst is not None:
        print(
            f"      worst_context={worst['context_str']} exp_sum={worst['expected_sum']} "
            f"exp_carry={worst['expected_carry_out']} max_gap={worst['max_gap']:.4f}",
            flush=True,
        )


def _ste_predict_all_tokens(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(xt)
    if logits.shape[-1] == 1:
        preds = (logits[:, :, 0] >= 0.0).long()
    else:
        preds = logits.argmax(dim=2)
    return preds.detach().cpu().numpy().astype(np.int64)


def _cvx_predict_all_tokens(
    *,
    w: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    n, steps = y_ood.shape
    kpar = int(getattr(init_cfg, "K_parallel", 1))
    if kpar > 1:
        pic = cvx_par.InitializationConfig(**asdict(init_cfg))
        d_tr, d_va, d_te, _ = cvx_par._build_feature_map(
            x_train, x_val, x_ood, pic, all_timesteps=True
        )
    else:
        d_tr, d_va, d_te, _ = _build_feature_map(
            x_train, x_val, x_ood, init_cfg, all_timesteps=True
        )
    scores = d_te @ w
    pred_flat = np.argmax(scores, axis=1)
    if int(pred_flat.shape[0]) != n * steps:
        raise ValueError(
            f"CVX flat preds {pred_flat.shape[0]} != n*steps={n * steps} for OOD shape {x_ood.shape}."
        )
    return pred_flat.reshape(n, steps)


def _run_ste_sweep_and_best(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    last_layer_readout: str,
    loss_type: str,
    optimizer_name: str,
    ste_epochs: int,
    batch_size: int,
    seed: int,
    ste_lr_grid: Sequence[float],
    ste_beta_grid: Sequence[float],
    pretrained_weights: Optional[List[np.ndarray]] = None,
) -> Tuple[object, Dict[str, float]]:
    """Run STE (lr × beta) sweep and refit best. ``pretrained_weights`` is forwarded to
    ``ste_solve`` on every call (sweep + refit), so finetune-style runs warm-start from
    the same initialization across the grid before training."""
    best_score = float("inf")
    best_params: Dict[str, float] | None = None
    for ste_lr in ste_lr_grid:
        for ste_beta in ste_beta_grid:
            _set_seed(seed)
            out = ste_solve(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                model_cfg=SteModelConfig(
                    d_in=d_in,
                    num_classes=num_classes,
                    L=L,
                    P_rec=P_rec,
                    P_last=P_last,
                    K_parallel=K_parallel,
                    last_layer_readout=last_layer_readout,
                ),
                solve_cfg=SteSolveConfig(
                    loss_name=loss_type,
                    optimizer_name=optimizer_name,
                    lr=float(ste_lr),
                    epochs=ste_epochs,
                    batch_size=None if batch_size == -1 else int(batch_size),
                    log_every=0,
                    weight_decay=0.0,
                    beta_path_reg=float(ste_beta),
                ),
                pretrained_weights=pretrained_weights,
            )
            score = float(out.best_losses["val_loss"]) + float(ste_beta)
            print(
                f"  [ste sweep] lr={ste_lr} beta={ste_beta} val_loss={out.best_losses['val_loss']:.6f} "
                f"score={score:.6f}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_params = {"lr": float(ste_lr), "beta": float(ste_beta)}
    if best_params is None:
        raise RuntimeError("STE sweep found no candidate.")
    print(
        f"[ste] selected lr={best_params['lr']:.6g} beta={best_params['beta']:.6g} "
        f"(score~val_loss+beta={best_score:.6f})",
        flush=True,
    )
    _set_seed(seed)
    best_out = ste_solve(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        model_cfg=SteModelConfig(
            d_in=d_in,
            num_classes=num_classes,
            L=L,
            P_rec=P_rec,
            P_last=P_last,
            K_parallel=K_parallel,
            last_layer_readout=last_layer_readout,
        ),
        solve_cfg=SteSolveConfig(
            loss_name=loss_type,
            optimizer_name=optimizer_name,
            lr=float(best_params["lr"]),
            epochs=ste_epochs,
            batch_size=None if batch_size == -1 else int(batch_size),
            log_every=0,
            weight_decay=0.0,
            beta_path_reg=float(best_params["beta"]),
        ),
        pretrained_weights=pretrained_weights,
    )
    return best_out, best_params


def _run_cvx_sweep_and_best(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    last_layer_readout: str,
    loss_type: str,
    optimizer_name: str,
    cvx_method: str,
    cvx_epochs: int,
    batch_size: int,
    seed: int,
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    cvx_lr_grid: Sequence[float],
    cvx_device: torch.device | None,
    init_mode: str = "gaussian",
    pretrained_weights: Optional[List[np.ndarray]] = None,
    compute_ce_dual: bool = False,
) -> Tuple[object, InitializationConfig, Dict[str, float | None]]:
    """Run CVX (beta × bias [× lr if sgd]) sweep with the chosen ``init_mode``.

    ``init_mode='gaussian'`` (default) builds LIF features from a fresh seeded init.
    ``init_mode='pretraining'`` requires ``pretrained_weights`` (a list of hidden-layer
    weight tensors in branch-major order, optionally with a trailing classifier head
    that the CVX feature builder ignores) — same convention as the hybrid bench.
    """
    if str(init_mode) not in ("gaussian", "pretraining"):
        raise ValueError(f"init_mode must be 'gaussian' or 'pretraining', got {init_mode!r}.")
    if init_mode == "pretraining" and (pretrained_weights is None or len(pretrained_weights) == 0):
        raise ValueError("init_mode='pretraining' requires non-empty pretrained_weights.")
    cvx_lr_eff = cvx_lr_sweep_values(cvx_method, tuple(float(x) for x in cvx_lr_grid))
    best_score = float("inf")
    best_params: Dict[str, float | None] | None = None
    for cvx_beta in beta_grid:
        for cvx_lr in cvx_lr_eff:
            for cvx_bias in bias_grid:
                _set_seed(seed)
                init_cfg = InitializationConfig(
                    mode=str(init_mode),
                    seed=seed,
                    L=L,
                    P_rec=P_rec,
                    P_last=P_last,
                    K_parallel=K_parallel,
                    feature_count=P_last,
                    last_layer_readout=last_layer_readout,
                    bias=float(cvx_bias),
                    pretrained_weights=pretrained_weights,
                )
                out = cvx_solve(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    y_test=y_test,
                    init_cfg=init_cfg,
                    solve_cfg=SolveConfig(
                        method=cvx_method,
                        loss_name=loss_type,
                        beta=float(cvx_beta),
                        lr=float(cvx_lr),
                        optimizer_name=optimizer_name,
                        epochs=cvx_epochs,
                        batch_size=None if batch_size == -1 else int(batch_size),
                        log_every=0,
                        compute_ce_dual=bool(compute_ce_dual),
                    ),
                    device=cvx_device,
                )
                val_obj = float(out.final_losses["val_objective"])
                d = out.diagnostics
                if bool(compute_ce_dual):
                    diag_str = (
                        f"primal={float(d.primal_value):.8g} dual={float(d.dual_value):.8g} "
                        f"gap={float(d.gap):.8g}"
                    )
                else:
                    diag_str = f"primal={float(d.primal_value):.8g} dual=skipped gap=skipped"
                print(
                    f"  [cvx sweep] beta={cvx_beta} lr={cvx_lr} bias={cvx_bias} val_objective={val_obj:.6f} "
                    + diag_str,
                    flush=True,
                )
                if val_obj < best_score:
                    best_score = val_obj
                    best_params = {
                        "lr": None if cvx_method == "cvx" else float(cvx_lr),
                        "beta": float(cvx_beta),
                        "bias": float(cvx_bias),
                    }
    if best_params is None:
        raise RuntimeError("CVX sweep found no candidate.")
    lr_note = best_params["lr"] if best_params["lr"] is not None else 0.0
    print(
        f"[cvx] selected beta={best_params['beta']:.6g} bias={best_params['bias']:.6g} "
        f"lr={lr_note} (val_objective={best_score:.6f}, method={cvx_method}, init={init_mode})",
        flush=True,
    )
    _set_seed(seed)
    best_init = InitializationConfig(
        mode=str(init_mode),
        seed=seed,
        L=L,
        P_rec=P_rec,
        P_last=P_last,
        K_parallel=K_parallel,
        feature_count=P_last,
        last_layer_readout=last_layer_readout,
        bias=float(best_params["bias"]),
        pretrained_weights=pretrained_weights,
    )
    best_out = cvx_solve(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        init_cfg=best_init,
        solve_cfg=SolveConfig(
            method=cvx_method,
            loss_name=loss_type,
            beta=float(best_params["beta"]),
            lr=float(best_params["lr"]) if best_params["lr"] is not None else 0.0,
            optimizer_name=optimizer_name,
            epochs=cvx_epochs,
            batch_size=None if batch_size == -1 else int(batch_size),
            log_every=0,
            compute_ce_dual=bool(compute_ce_dual),
        ),
        device=cvx_device,
    )
    _print_cvx_diagnostics("[cvx] best model (after retrain)", best_out)
    return best_out, best_init, best_params


def _gaussian_lif_hidden_weights_out_in(
    *,
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    seed: int,
    variant: str = "standard",
) -> List[np.ndarray]:
    """Same Gaussian LIF init that ``cvx_parallel_Solve._build_feature_map`` produces in
    ``mode='gaussian'``, returned in PyTorch ``nn.Linear`` (out, in) layout so it can be
    handed to ``ste_solve(pretrained_weights=...)`` as the hidden stack.
    """
    k = int(K_parallel)
    sub_p_rec = cvx_par._parallel_branch_width(int(P_rec), k, "P_rec")
    sub_p_last = cvx_par._parallel_branch_width(int(P_last), k, "P_last")
    branch_hidden_dims = cvx_par._hidden_dims_like_snn_p2(int(L), int(sub_p_rec), int(sub_p_last))
    rng = np.random.default_rng(int(seed))
    out: List[np.ndarray] = []
    for _ in range(k):
        in_dim = int(d_in)
        for h in branch_hidden_dims:
            w_in_out = cvx_par._sample_weight_matrix(rng, in_dim, int(h), str(variant))
            out.append(np.asarray(w_in_out.T, dtype=np.float64))
            in_dim = int(h)
    return out


def _ste_pretrained_from_cvx_bundle(
    *,
    d_in: int,
    L: int,
    P_rec: int,
    P_last: int,
    K_parallel: int,
    seed: int,
    cvx_classifier_w_p_last_x_classes: np.ndarray,
    variant: str = "standard",
) -> List[np.ndarray]:
    """Build an STE-loadable weight list from a CVX-pretrain Gaussian bundle.

    The CVX path uses Gaussian-init LIF features (deterministic from ``seed``) and fits a
    linear convex classifier whose weights have shape ``(P_last, num_classes)``. To
    initialize an STE model with this exact starting point, we regenerate the Gaussian
    hidden weights (transposed to ``(out, in)``) and append the CVX classifier weights
    transposed to ``(num_classes, P_last)`` — the layout ``ste_solve`` expects.
    """
    if cvx_classifier_w_p_last_x_classes.ndim != 2:
        raise ValueError(
            f"Expected CVX classifier weight shape (P_last, num_classes); got "
            f"{cvx_classifier_w_p_last_x_classes.shape}."
        )
    if int(cvx_classifier_w_p_last_x_classes.shape[0]) != int(P_last):
        raise ValueError(
            f"CVX classifier rows {cvx_classifier_w_p_last_x_classes.shape[0]} != P_last={P_last}."
        )
    hidden = _gaussian_lif_hidden_weights_out_in(
        d_in=d_in, L=L, P_rec=P_rec, P_last=P_last, K_parallel=K_parallel, seed=seed, variant=variant
    )
    head_out_in = np.asarray(cvx_classifier_w_p_last_x_classes.T, dtype=np.float64)
    return [*hidden, head_out_in]


def _ste_predict_and_full_metrics(
    *,
    ste_model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    base: int,
    n_digits: int,
    block_size: int,
    x_for_rule_context: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run STE prediction on ``x`` (which may be masked), then compute detailed and
    rule-context metrics. ``x_for_rule_context`` overrides the input fed to
    ``add_rule_context_diagnostics`` — pass an **unmasked** copy when ``x`` was masked,
    so the per-context analysis can still read the true carry chain. Defaults to ``x``
    (correct when no masking is applied).
    """
    pred = _ste_predict_all_tokens(ste_model, x)
    detail = _detailed_arithmetic_metrics(pred, y, block_size=block_size)
    x_rule = x if x_for_rule_context is None else x_for_rule_context
    rule = add_rule_context_diagnostics(pred, y, x_rule, int(base), int(n_digits), block_size=block_size)
    return {**detail, "rule_context": rule}


def _cvx_predict_and_full_metrics(
    *,
    w: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    init_cfg: InitializationConfig,
    base: int,
    n_digits: int,
    block_size: int,
    x_eval_for_rule_context: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Like :func:`_ste_predict_and_full_metrics`, but for the CVX path. Features are
    rebuilt from ``x_train, x_val, x_eval`` (which may all be masked); pass
    ``x_eval_for_rule_context`` (unmasked) to keep the per-context analysis valid.
    """
    pred = _cvx_predict_all_tokens(
        w=w, x_train=x_train, x_val=x_val, x_ood=x_eval, y_ood=y_eval, init_cfg=init_cfg
    )
    detail = _detailed_arithmetic_metrics(pred, y_eval, block_size=block_size)
    x_rule = x_eval if x_eval_for_rule_context is None else x_eval_for_rule_context
    rule = add_rule_context_diagnostics(pred, y_eval, x_rule, int(base), int(n_digits), block_size=block_size)
    return {**detail, "rule_context": rule}


def _print_cvx_diagnostics(prefix: str, out: object) -> None:
    """Print primal / dual objectives and gap from ``cvx_solve`` / parallel CVX result."""
    d = out.diagnostics
    print(
        f"{prefix} primal={float(d.primal_value):.8g} dual={float(d.dual_value):.8g} gap={float(d.gap):.8g}",
        flush=True,
    )


def _save_ste_npz(path: Path, model: torch.nn.Module) -> None:
    weights = _extract_weight_list(model)
    kw = {f"w{i}": weights[i] for i in range(len(weights))}
    np.savez_compressed(path, **kw)


def _save_cvx_npz(path: Path, trained_model: object) -> None:
    if not isinstance(trained_model, dict) or "weights" not in trained_model:
        raise TypeError(f"Expected CVX trained_model dict with 'weights', got {type(trained_model)}")
    np.savez_compressed(path, W=trained_model["weights"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--arith_base",
        type=int,
        default=2,
        help=f"Operand radix (addition in this base). Supported: {SUPPORTED_BASES}.",
    )
    p.add_argument("--n_digits_train", type=int, default=5, help="Training / val digit width (in-distribution).")
    p.add_argument(
        "--ood_digits",
        type=int,
        nargs="+",
        default=[10, 15, 25],
        help="OOD n_digits to evaluate; default 10,15,25 (5-column train, longer test).",
    )
    p.add_argument(
        "--ood_block_size",
        type=int,
        default=5,
        help="OOD: non-overlapping time blocks in full-sequence per-block stats (1-accuracy).",
    )
    p.add_argument("--n_train", type=int, default=8192)
    p.add_argument("--n_val", type=int, default=2000, help="Val samples for training n_digits only.")
    p.add_argument("--n_test", type=int, default=2000, help="Test samples per OOD digit width.")
    p.add_argument(
        "--variants",
        choices=("minimal", "full5"),
        default="full5",
        help=(
            "minimal: 2 variants (STE-from-scratch + CVX-from-Gaussian) — original behavior. "
            "full5: 5 variants mirroring the carry/state hybrid bench — "
            "ste_pretrain (split A), cvx_from_ste_pretrain, ste_finetune_from_ste_pretrain (split B), "
            "cvx_pretrain (Gaussian), ste_finetune_from_cvx_pretrain (split B). Default: full5."
        ),
    )
    p.add_argument(
        "--n_train_pre",
        type=int,
        default=None,
        help="Pretrain (split A) training samples; default = --n_train.",
    )
    p.add_argument(
        "--n_train_ft",
        type=int,
        default=None,
        help="Finetune (split B) training samples; default = --n_train.",
    )
    p.add_argument(
        "--n_val_pre",
        type=int,
        default=None,
        help="Pretrain (split A) val samples; default = --n_val.",
    )
    p.add_argument(
        "--n_val_ft",
        type=int,
        default=None,
        help="Finetune (split B) val samples; default = --n_val.",
    )
    p.add_argument(
        "--finetune_seed_offset",
        type=int,
        default=1000,
        help="Split-B seed offset (added to base seed for finetune dataset).",
    )
    p.add_argument(
        "--eval_seed_offset",
        type=int,
        default=2000,
        help="Test-split seed offset (eval_seed=base_seed+offset; used for ID test seeded as eval_seed+47).",
    )
    mask_grp = p.add_mutually_exclusive_group()
    mask_grp.add_argument(
        "--mask_carry_in",
        dest="mask_carry_in",
        action="store_true",
        help=(
            "Zero out input channel 2 (true carry-in column) in train/val/test/OOD. "
            "Default ON — forces the model to track the carry chain itself, comparable "
            "to hybrid AR mode."
        ),
    )
    mask_grp.add_argument(
        "--no_mask_carry_in",
        dest="mask_carry_in",
        action="store_false",
        help=(
            "Keep the true carry-in column in input channel 2 (turn off masking). "
            "This is the hybrid TF analogue (model reads the true carry every step)."
        ),
    )
    p.set_defaults(mask_carry_in=True)
    cvx_dual_grp = p.add_mutually_exclusive_group()
    cvx_dual_grp.add_argument(
        "--cvx_ce_dual",
        dest="cvx_ce_dual",
        action="store_true",
        help=(
            "With --cvx_method cvx and --loss_type ce: also run the conic dual after the "
            "primal solve (slower; populates dual_value/gap diagnostics). Default OFF."
        ),
    )
    cvx_dual_grp.add_argument(
        "--no_cvx_ce_dual",
        dest="cvx_ce_dual",
        action="store_false",
        help="Silence the CE dual solve (default). Diagnostics dual/gap will be NaN.",
    )
    p.set_defaults(cvx_ce_dual=False)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--P_rec", type=int, default=256)
    p.add_argument("--P_last", type=int, default=512)
    p.add_argument("--K_parallel", type=int, default=2)
    p.add_argument("--loss_type", choices=("ce", "hinge", "hinge_ovr", "squared"), default="hinge_ovr")
    p.add_argument("--optimizer_name", choices=("adam", "sgd"), default="sgd")
    p.add_argument("--cvx_method", choices=("cvx", "sgd"), default="cvx")
    p.add_argument("--cvx_device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--batch_size", type=int, default=-1)
    p.add_argument("--cvx_epochs", type=int, default=200)
    p.add_argument("--ste_epochs", type=int, default=200)
    p.add_argument(
        "--ste_last_layer_readout",
        choices=("membrane", "spike"),
        default="membrane",
        help="SNN (STE) last hidden readout into the classifier.",
    )
    p.add_argument(
        "--cvx_last_layer_readout",
        choices=("membrane", "spike"),
        default="spike",
        help="CVX LIF stack last hidden readout used for the linear convex layer (features D).",
    )
    p.add_argument(
        "--ste_lr_grid",
        type=float,
        nargs="+",
        default=None,
        help=f"Default: {list(LR_GRID_DEFAULT)}",
    )
    p.add_argument(
        "--ste_beta_grid",
        type=float,
        nargs="+",
        default=None,
        help=f"Default: {list(BETA_GRID_DEFAULT)}",
    )
    p.add_argument(
        "--cvx_beta_grid",
        type=float,
        nargs="+",
        default=None,
        help=f"Default: {list(BETA_GRID_DEFAULT)}",
    )
    p.add_argument(
        "--cvx_bias_grid",
        type=float,
        nargs="+",
        default=None,
        help=f"Default: {list(BIAS_GRID_DEFAULT)}",
    )
    p.add_argument("--out_root", type=str, default="", help="Directory for this run; default: sweep_results/...")
    p.add_argument("--verify_samples", type=int, default=8)
    p.add_argument(
        "--no_save_metrics",
        action="store_true",
        help="If set, do not write metrics JSON (seed_*/metrics.json, metrics.json, summary_all_seeds.json).",
    )
    p.add_argument(
        "--save_weights",
        action="store_true",
        help="Write ste_best.npz and cvx_best.npz under each seed_* (default: off; models only in ste_res / cvx_res).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.arith_base) not in SUPPORTED_BASES:
        raise ValueError(
            f"Unsupported --arith_base={args.arith_base!r}. "
            f"Use one of: {SUPPORTED_BASES} (data loader SUPPORTED_BASES)."
        )
    T_expect = _add_timesteps(args.n_digits_train)
    ste_lr_grid = tuple(args.ste_lr_grid) if args.ste_lr_grid is not None else LR_GRID_DEFAULT
    ste_beta_grid = tuple(args.ste_beta_grid) if args.ste_beta_grid is not None else BETA_GRID_DEFAULT
    cvx_beta_grid = tuple(args.cvx_beta_grid) if args.cvx_beta_grid is not None else BETA_GRID_DEFAULT
    cvx_bias_grid = tuple(args.cvx_bias_grid) if args.cvx_bias_grid is not None else BIAS_GRID_DEFAULT

    atomic_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root).strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = (
            atomic_dir
            / "sweep_results"
            / f"arithmetic_add_b{int(args.arith_base)}_len_gen_L{args.L}_traind{args.n_digits_train}_K{args.K_parallel}_{stamp}"
        )
    out_root.mkdir(parents=True, exist_ok=True)

    cvx_dev = _resolve_cvx_device(args.cvx_device)
    save_metrics = not bool(args.no_save_metrics)
    save_weights = bool(args.save_weights)

    n_train_pre = int(args.n_train_pre) if args.n_train_pre is not None else int(args.n_train)
    n_train_ft = int(args.n_train_ft) if args.n_train_ft is not None else int(args.n_train)
    n_val_pre = int(args.n_val_pre) if args.n_val_pre is not None else int(args.n_val)
    n_val_ft = int(args.n_val_ft) if args.n_val_ft is not None else int(args.n_val)
    if str(args.variants) not in ("minimal", "full5"):
        raise ValueError(f"Unknown --variants={args.variants!r}")

    config_dump = {
        "seeds": list(args.seeds),
        "arith_op": "add",
        "arith_base": args.arith_base,
        "n_digits_train": args.n_digits_train,
        "T_train": T_expect,
        "ood_digits": list(args.ood_digits),
        "ood_block_size": int(args.ood_block_size),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test_per_ood": args.n_test,
        "n_train_pre": n_train_pre,
        "n_train_ft": n_train_ft,
        "n_val_pre": n_val_pre,
        "n_val_ft": n_val_ft,
        "finetune_seed_offset": int(args.finetune_seed_offset),
        "eval_seed_offset": int(args.eval_seed_offset),
        "mask_carry_in": bool(args.mask_carry_in),
        "cvx_ce_dual": bool(args.cvx_ce_dual),
        "variants": str(args.variants),
        "L": args.L,
        "P_rec": args.P_rec,
        "P_last": args.P_last,
        "K_parallel": args.K_parallel,
        "loss_type": args.loss_type,
        "cvx_method": args.cvx_method,
        "cvx_epochs": args.cvx_epochs,
        "ste_epochs": args.ste_epochs,
        "ste_last_layer_readout": str(args.ste_last_layer_readout),
        "cvx_last_layer_readout": str(args.cvx_last_layer_readout),
        "ste_lr_grid": list(ste_lr_grid),
        "ste_beta_grid": list(ste_beta_grid),
        "cvx_beta_grid": list(cvx_beta_grid),
        "cvx_bias_grid": list(cvx_bias_grid),
        "cvx_lr_grid_for_sgd": list(ste_lr_grid),
        "cvx_lr_effective_in_run": list(cvx_lr_sweep_values(args.cvx_method, ste_lr_grid)),
        "note_cvx_lr": "For cvx_method=cvx, lr is not used by the solver; sweep is beta × bias. "
        "For cvx_method=sgd, lr values match ste_lr_grid.",
        "save_weights": save_weights,
        "default_reporting_metrics": {
            "add_rule_context_diagnostics": True,
            "where": "in-distribution `X_test` split and each OOD `n_digits` (context-conditioned local-rule analysis on sum positions)",
        },
        "stages_full5": [
            "ste_pretrain",
            "cvx_from_ste_pretrain",
            "ste_finetune_from_ste_pretrain_new_train",
            "cvx_pretrain",
            "ste_finetune_from_cvx_pretrain_new_train",
        ],
    }
    (out_root / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n")

    all_seeds_summary: List[Dict[str, Any]] = []
    all_seeds_payloads: List[Dict[str, Any]] = []

    def _flush_root_metrics() -> None:
        if not save_metrics:
            return
        root_metrics = {
            "run_config": config_dump,
            "out_root": str(out_root),
            "seeds": list(all_seeds_payloads),
        }
        (out_root / "metrics.json").write_text(json.dumps(root_metrics, indent=2, default=str) + "\n")

    for seed in args.seeds:
        print("\n" + "=" * 100, flush=True)
        print(f"SEED {seed} (variants={args.variants}, mask_carry_in={bool(args.mask_carry_in)})", flush=True)
        print("=" * 100, flush=True)

        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Build train / val / test splits. For variants=full5 we use independent seeds
        # for the pretrain and finetune training splits but share the test split via
        # ``eval_seed_offset``, mirroring the carry/state hybrid bench.
        pre_seed = int(seed)
        ft_seed = int(seed) + int(args.finetune_seed_offset)
        eval_seed = int(seed) + int(args.eval_seed_offset)
        x_train, x_train_unmasked, y_train = _build_lengthgen_split(
            op="add",
            base=int(args.arith_base),
            n_digits=int(args.n_digits_train),
            n_samples=n_train_pre,
            seed=pre_seed + 11,
            verify_count=int(args.verify_samples),
            mask_carry=bool(args.mask_carry_in),
        )
        x_val, x_val_unmasked, y_val = _build_lengthgen_split(
            op="add",
            base=int(args.arith_base),
            n_digits=int(args.n_digits_train),
            n_samples=n_val_pre,
            seed=pre_seed + 29,
            verify_count=int(args.verify_samples),
            mask_carry=bool(args.mask_carry_in),
        )
        x_test, x_test_unmasked, y_test = _build_lengthgen_split(
            op="add",
            base=int(args.arith_base),
            n_digits=int(args.n_digits_train),
            n_samples=int(args.n_test),
            seed=eval_seed + 47,
            verify_count=int(args.verify_samples),
            mask_carry=bool(args.mask_carry_in),
        )
        if int(x_train.shape[1]) != T_expect:
            raise ValueError(
                f"Train T={x_train.shape[1]} != expected {T_expect} for n_digits={args.n_digits_train} add."
            )
        d_in = int(x_train.shape[2])
        # Number of token classes is base for sum positions; the MSD carry-out also takes values in [0, base).
        num_classes = int(args.arith_base)

        # Stage 1: STE pretrain on split A. (Same training that the original "minimal"
        # variant runs.)
        print(
            f"[phase] STE pretrain (split A, last_layer_readout={args.ste_last_layer_readout})...",
            flush=True,
        )
        ste_res, ste_sel = _run_ste_sweep_and_best(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            d_in=d_in,
            L=args.L,
            P_rec=args.P_rec,
            P_last=args.P_last,
            K_parallel=args.K_parallel,
            last_layer_readout=str(args.ste_last_layer_readout),
            loss_type=args.loss_type,
            optimizer_name=args.optimizer_name,
            ste_epochs=args.ste_epochs,
            batch_size=args.batch_size,
            seed=pre_seed,
            ste_lr_grid=ste_lr_grid,
            ste_beta_grid=ste_beta_grid,
        )
        ste_model = ste_res.model
        if not isinstance(ste_model, (SNNBaselineSeq, ste_par.SNNBaselineSeq)):
            raise TypeError(f"Unexpected STE model type {type(ste_model)}")

        print(
            "[ste_pretrain] train done. token_acc="
            f"{ste_res.best_losses.get('train_token_acc')} "
            f"val={ste_res.best_losses.get('val_token_acc')} "
            f"test={ste_res.best_losses.get('test_token_acc')} "
            f"seq_acc test={ste_res.best_losses.get('test_seq_acc')}",
            flush=True,
        )
        if save_weights:
            _save_ste_npz(seed_dir / "ste_best.npz", ste_model)
            print(f"[saved] {seed_dir / 'ste_best.npz'}", flush=True)
        ste_pretrain_weights = _extract_weight_list(ste_model)

        # Stage 4: CVX pretrain (Gaussian) on split A.
        print(
            f"[phase] CVX pretrain (Gaussian, last_layer_readout={args.cvx_last_layer_readout})...",
            flush=True,
        )
        cvx_res, cvx_init, cvx_sel = _run_cvx_sweep_and_best(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            L=args.L,
            P_rec=args.P_rec,
            P_last=args.P_last,
            K_parallel=args.K_parallel,
            last_layer_readout=str(args.cvx_last_layer_readout),
            loss_type=args.loss_type,
            optimizer_name=args.optimizer_name,
            cvx_method=args.cvx_method,
            cvx_epochs=args.cvx_epochs,
            batch_size=args.batch_size,
            seed=pre_seed,
            beta_grid=cvx_beta_grid,
            bias_grid=cvx_bias_grid,
            cvx_lr_grid=ste_lr_grid,
            cvx_device=cvx_dev,
            init_mode="gaussian",
            pretrained_weights=None,
            compute_ce_dual=bool(args.cvx_ce_dual),
        )
        print(
            "[cvx_pretrain] train done. token_acc="
            f"{cvx_res.final_losses.get('test_token_acc')} "
            f"seq_acc test={cvx_res.final_losses.get('test_seq_acc')}",
            flush=True,
        )
        if save_weights:
            _save_cvx_npz(seed_dir / "cvx_best.npz", cvx_res.trained_model)
            print(f"[saved] {seed_dir / 'cvx_best.npz'}", flush=True)

        w_cvx = cvx_res.trained_model["weights"]

        # Extra stages (variants=full5): build the finetune-split data first, then run
        # the 3 additional models — CVX-from-STE-pretrain (split A), STE-finetune-from-STE
        # (split B), and STE-finetune-from-CVX (split B). Each stage has its own param
        # selection and weights, and each is evaluated independently below.
        stage_results: Dict[str, Dict[str, Any]] = {
            "ste_pretrain": {
                "kind": "ste",
                "model": ste_model,
                "init_cfg": None,
                "fit_x_train": x_train,
                "fit_x_val": x_val,
                "selected_params": ste_sel,
            },
            "cvx_pretrain": {
                "kind": "cvx",
                "w": w_cvx,
                "init_cfg": cvx_init,
                "fit_x_train": x_train,
                "fit_x_val": x_val,
                "selected_params": {
                    k: (float(v) if v is not None else None) for k, v in cvx_sel.items()
                },
                "diagnostics": asdict(cvx_res.diagnostics),
                "train_final_losses": {
                    k: float(v) for k, v in cvx_res.final_losses.items() if isinstance(v, (float, int))
                },
            },
        }
        if str(args.variants) == "full5":
            print(
                f"[phase] CVX from STE pretrain (split A, last_layer_readout={args.cvx_last_layer_readout})...",
                flush=True,
            )
            cvx_from_ste_res, cvx_from_ste_init, cvx_from_ste_sel = _run_cvx_sweep_and_best(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                L=args.L,
                P_rec=args.P_rec,
                P_last=args.P_last,
                K_parallel=args.K_parallel,
                last_layer_readout=str(args.cvx_last_layer_readout),
                loss_type=args.loss_type,
                optimizer_name=args.optimizer_name,
                cvx_method=args.cvx_method,
                cvx_epochs=args.cvx_epochs,
                batch_size=args.batch_size,
                seed=pre_seed,
                beta_grid=cvx_beta_grid,
                bias_grid=cvx_bias_grid,
                cvx_lr_grid=ste_lr_grid,
                cvx_device=cvx_dev,
                init_mode="pretraining",
                pretrained_weights=ste_pretrain_weights,
                compute_ce_dual=bool(args.cvx_ce_dual),
            )
            w_cvx_from_ste = cvx_from_ste_res.trained_model["weights"]
            stage_results["cvx_from_ste_pretrain"] = {
                "kind": "cvx",
                "w": w_cvx_from_ste,
                "init_cfg": cvx_from_ste_init,
                "fit_x_train": x_train,
                "fit_x_val": x_val,
                "selected_params": {
                    k: (float(v) if v is not None else None) for k, v in cvx_from_ste_sel.items()
                },
                "diagnostics": asdict(cvx_from_ste_res.diagnostics),
                "train_final_losses": {
                    k: float(v) for k, v in cvx_from_ste_res.final_losses.items() if isinstance(v, (float, int))
                },
            }

            # Build the finetune split (split B). Same digit length as pretrain.
            x_train_ft, _x_train_ft_unmasked, y_train_ft = _build_lengthgen_split(
                op="add",
                base=int(args.arith_base),
                n_digits=int(args.n_digits_train),
                n_samples=n_train_ft,
                seed=ft_seed + 11,
                verify_count=int(args.verify_samples),
                mask_carry=bool(args.mask_carry_in),
            )
            x_val_ft, _x_val_ft_unmasked, y_val_ft = _build_lengthgen_split(
                op="add",
                base=int(args.arith_base),
                n_digits=int(args.n_digits_train),
                n_samples=n_val_ft,
                seed=ft_seed + 29,
                verify_count=int(args.verify_samples),
                mask_carry=bool(args.mask_carry_in),
            )

            print(
                f"[phase] STE finetune from STE pretrain (split B, last_layer_readout={args.ste_last_layer_readout})...",
                flush=True,
            )
            ste_ft_from_ste_res, ste_ft_from_ste_sel = _run_ste_sweep_and_best(
                x_train=x_train_ft,
                y_train=y_train_ft,
                x_val=x_val_ft,
                y_val=y_val_ft,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                d_in=d_in,
                L=args.L,
                P_rec=args.P_rec,
                P_last=args.P_last,
                K_parallel=args.K_parallel,
                last_layer_readout=str(args.ste_last_layer_readout),
                loss_type=args.loss_type,
                optimizer_name=args.optimizer_name,
                ste_epochs=args.ste_epochs,
                batch_size=args.batch_size,
                seed=ft_seed,
                ste_lr_grid=ste_lr_grid,
                ste_beta_grid=ste_beta_grid,
                pretrained_weights=ste_pretrain_weights,
            )
            ste_ft_from_ste_model = ste_ft_from_ste_res.model
            stage_results["ste_finetune_from_ste_pretrain_new_train"] = {
                "kind": "ste",
                "model": ste_ft_from_ste_model,
                "init_cfg": None,
                "fit_x_train": x_train_ft,
                "fit_x_val": x_val_ft,
                "selected_params": ste_ft_from_ste_sel,
                "train_final_losses": {
                    k: float(v) for k, v in ste_ft_from_ste_res.best_losses.items() if isinstance(v, (float, int))
                },
            }

            # STE-from-CVX-pretrain handoff: rebuild the Gaussian LIF init that the CVX
            # path used, append the CVX-fit linear classifier transposed to (out, in),
            # and feed both to ste_solve via pretrained_weights.
            ste_pretrained_from_cvx = _ste_pretrained_from_cvx_bundle(
                d_in=d_in,
                L=int(args.L),
                P_rec=int(args.P_rec),
                P_last=int(args.P_last),
                K_parallel=int(args.K_parallel),
                seed=pre_seed,
                cvx_classifier_w_p_last_x_classes=w_cvx,
            )
            print(
                f"[phase] STE finetune from CVX pretrain (split B, last_layer_readout={args.ste_last_layer_readout})...",
                flush=True,
            )
            ste_ft_from_cvx_res, ste_ft_from_cvx_sel = _run_ste_sweep_and_best(
                x_train=x_train_ft,
                y_train=y_train_ft,
                x_val=x_val_ft,
                y_val=y_val_ft,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                d_in=d_in,
                L=args.L,
                P_rec=args.P_rec,
                P_last=args.P_last,
                K_parallel=args.K_parallel,
                last_layer_readout=str(args.ste_last_layer_readout),
                loss_type=args.loss_type,
                optimizer_name=args.optimizer_name,
                ste_epochs=args.ste_epochs,
                batch_size=args.batch_size,
                seed=ft_seed,
                ste_lr_grid=ste_lr_grid,
                ste_beta_grid=ste_beta_grid,
                pretrained_weights=ste_pretrained_from_cvx,
            )
            ste_ft_from_cvx_model = ste_ft_from_cvx_res.model
            stage_results["ste_finetune_from_cvx_pretrain_new_train"] = {
                "kind": "ste",
                "model": ste_ft_from_cvx_model,
                "init_cfg": None,
                "fit_x_train": x_train_ft,
                "fit_x_val": x_val_ft,
                "selected_params": ste_ft_from_cvx_sel,
                "train_final_losses": {
                    k: float(v) for k, v in ste_ft_from_cvx_res.best_losses.items() if isinstance(v, (float, int))
                },
            }

        # ID-test rule-context (always reported) for ste_pretrain + cvx_pretrain (legacy
        # back-compat keys).
        pred_ste_id = _ste_predict_all_tokens(ste_model, x_test)
        pred_cvx_id = _cvx_predict_all_tokens(
            w=w_cvx,
            x_train=x_train,
            x_val=x_val,
            x_ood=x_test,
            y_ood=y_test,
            init_cfg=cvx_init,
        )
        nd_id = int(args.n_digits_train)
        rule_context_in_dist = {
            "n_digits": nd_id,
            "ste": add_rule_context_diagnostics(
                pred_ste_id,
                y_test,
                x_test_unmasked,
                int(args.arith_base),
                nd_id,
                block_size=int(args.ood_block_size),
            ),
            "cvx": add_rule_context_diagnostics(
                pred_cvx_id,
                y_test,
                x_test_unmasked,
                int(args.arith_base),
                nd_id,
                block_size=int(args.ood_block_size),
            ),
        }
        _print_rule_context_summary(
            f"[in-dist test] rule_context  n_digits={nd_id}  STE", rule_context_in_dist["ste"]
        )
        _print_rule_context_summary(
            f"[in-dist test] rule_context  n_digits={nd_id}  CVX", rule_context_in_dist["cvx"]
        )

        # Per-stage ID metrics (TF-style — true carry-in input unless --mask_carry_in).
        stage_id_metrics: Dict[str, Dict[str, Any]] = {}
        for stage_key, sr in stage_results.items():
            ood_bs = int(args.ood_block_size)
            if sr["kind"] == "ste":
                m = _ste_predict_and_full_metrics(
                    ste_model=sr["model"],
                    x=x_test,
                    y=y_test,
                    base=int(args.arith_base),
                    n_digits=nd_id,
                    block_size=ood_bs,
                    x_for_rule_context=x_test_unmasked,
                )
            else:
                m = _cvx_predict_and_full_metrics(
                    w=sr["w"],
                    x_train=sr["fit_x_train"],
                    x_val=sr["fit_x_val"],
                    x_eval=x_test,
                    y_eval=y_test,
                    init_cfg=sr["init_cfg"],
                    base=int(args.arith_base),
                    n_digits=nd_id,
                    block_size=ood_bs,
                    x_eval_for_rule_context=x_test_unmasked,
                )
            stage_id_metrics[stage_key] = m

        ood_report: Dict[str, Any] = {}
        for nd in args.ood_digits:
            te_seed = seed + 10_000 + int(nd) * 97
            x_ood_unmasked, y_ood = _make_test_xy(
                op="add",
                base=args.arith_base,
                n_digits=nd,
                n_samples=args.n_test,
                seed=te_seed,
                verify_count=args.verify_samples,
            )
            x_ood = _mask_carry_inputs(x_ood_unmasked) if bool(args.mask_carry_in) else x_ood_unmasked
            T_ood = _add_timesteps(nd)
            if x_ood.shape[1] != T_ood:
                raise ValueError(f"OOD T mismatch for nd={nd}: got {x_ood.shape[1]}, expect {T_ood}")
            if x_ood.shape[2] != d_in:
                raise ValueError(f"OOD d_in {x_ood.shape[2]} != train d_in {d_in}")

            pred_ste = _ste_predict_all_tokens(ste_model, x_ood)
            pred_cvx = _cvx_predict_all_tokens(
                w=w_cvx,
                x_train=x_train,
                x_val=x_val,
                x_ood=x_ood,
                y_ood=y_ood,
                init_cfg=cvx_init,
            )

            ood_bs = int(args.ood_block_size)
            ste_m = _detailed_arithmetic_metrics(pred_ste, y_ood, block_size=ood_bs)
            cvx_m = _detailed_arithmetic_metrics(pred_cvx, y_ood, block_size=ood_bs)
            ste_rule = add_rule_context_diagnostics(
                pred_ste,
                y_ood,
                x_ood,
                int(args.arith_base),
                int(nd),
                block_size=ood_bs,
            )
            cvx_rule = add_rule_context_diagnostics(
                pred_cvx,
                y_ood,
                x_ood,
                int(args.arith_base),
                int(nd),
                block_size=ood_bs,
            )
            # Per-OOD all-stage eval: legacy ``ste``/``cvx`` keys map to ste_pretrain /
            # cvx_pretrain; full5 adds the other 3 stage keys.
            ood_bs = int(args.ood_block_size)
            ood_report[f"n_digits_{nd}"] = {}
            for stage_key, sr in stage_results.items():
                if sr["kind"] == "ste":
                    sm = _ste_predict_and_full_metrics(
                        ste_model=sr["model"],
                        x=x_ood,
                        y=y_ood,
                        base=int(args.arith_base),
                        n_digits=int(nd),
                        block_size=ood_bs,
                        x_for_rule_context=x_ood_unmasked,
                    )
                else:
                    sm = _cvx_predict_and_full_metrics(
                        w=sr["w"],
                        x_train=sr["fit_x_train"],
                        x_val=sr["fit_x_val"],
                        x_eval=x_ood,
                        y_eval=y_ood,
                        init_cfg=sr["init_cfg"],
                        base=int(args.arith_base),
                        n_digits=int(nd),
                        block_size=ood_bs,
                        x_eval_for_rule_context=x_ood_unmasked,
                    )
                ood_report[f"n_digits_{nd}"][stage_key] = sm
            ood_report[f"n_digits_{nd}"]["ste"] = ood_report[f"n_digits_{nd}"]["ste_pretrain"]
            ood_report[f"n_digits_{nd}"]["cvx"] = ood_report[f"n_digits_{nd}"]["cvx_pretrain"]

            ste_m = ood_report[f"n_digits_{nd}"]["ste_pretrain"]
            cvx_m = ood_report[f"n_digits_{nd}"]["cvx_pretrain"]
            print(
                f"[ood] n_digits={nd}  STE-pretrain  token_acc={ste_m['token_acc']:.4f} "
                f"seq_acc={ste_m['seq_acc']:.4f} "
                f"mean_first_wrong(among wrong)={ste_m['mean_first_wrong_timestep_among_wrong']:.4f}",
                flush=True,
            )
            _print_rule_context_summary("      STE-pretrain rule_context", ste_m["rule_context"])
            print(
                f"[ood] n_digits={nd}  CVX-Gauss     token_acc={cvx_m['token_acc']:.4f} seq_acc={cvx_m['seq_acc']:.4f} "
                f"mean_first_wrong(among wrong)={cvx_m['mean_first_wrong_timestep_among_wrong']:.4f}",
                flush=True,
            )
            _print_rule_context_summary("      CVX-Gauss rule_context", cvx_m["rule_context"])
            if str(args.variants) == "full5":
                for stage_key in (
                    "cvx_from_ste_pretrain",
                    "ste_finetune_from_ste_pretrain_new_train",
                    "ste_finetune_from_cvx_pretrain_new_train",
                ):
                    sm = ood_report[f"n_digits_{nd}"][stage_key]
                    print(
                        f"[ood] n_digits={nd}  {stage_key:<48}  token_acc={sm['token_acc']:.4f} "
                        f"seq_acc={sm['seq_acc']:.4f}",
                        flush=True,
                    )

        # Per-stage payload (selected_params, id_metrics, ood_eval) for full5 reporting.
        stages_payload: Dict[str, Any] = {}
        for stage_key, sr in stage_results.items():
            stage_id = stage_id_metrics[stage_key]
            stage_ood = {
                k: ood_report[k][stage_key] for k in ood_report.keys()
            }
            stages_payload[stage_key] = {
                "selected_params": sr["selected_params"],
                "id_metrics": stage_id,
                "ood_eval": stage_ood,
            }
            if "diagnostics" in sr:
                stages_payload[stage_key]["diagnostics"] = sr["diagnostics"]
            if "train_final_losses" in sr:
                stages_payload[stage_key]["train_final_losses"] = sr["train_final_losses"]

        seed_payload: Dict[str, Any] = {
            "seed": seed,
            "variants": str(args.variants),
            "mask_carry_in": bool(args.mask_carry_in),
            "split_seeds": {
                "pretrain_train": pre_seed + 11,
                "pretrain_val": pre_seed + 29,
                "finetune_train": ft_seed + 11,
                "finetune_val": ft_seed + 29,
                "eval_test": eval_seed + 47,
            },
            "add_rule_context_in_dist_test": rule_context_in_dist,
            "ste_last_layer_readout": str(args.ste_last_layer_readout),
            "cvx_last_layer_readout": str(args.cvx_last_layer_readout),
            "ste_selected_params": ste_sel,
            "ste_train_best_losses": {k: float(v) for k, v in ste_res.best_losses.items() if isinstance(v, (float, int))},
            "cvx_selected_params": {k: (float(v) if v is not None else None) for k, v in cvx_sel.items()},
            "cvx_diagnostics_best": asdict(cvx_res.diagnostics),
            "cvx_train_final_losses": {k: float(v) for k, v in cvx_res.final_losses.items() if isinstance(v, (float, int))},
            "stages": stages_payload,
            "ood_eval": ood_report,
            "artifacts": (
                {
                    "ste_npz": str((seed_dir / "ste_best.npz").resolve()),
                    "cvx_npz": str((seed_dir / "cvx_best.npz").resolve()),
                }
                if save_weights
                else None
            ),
            "note_artifacts": "STE: ste_res.model. CVX: cvx_res.trained_model['weights']. "
            "npz files only if --save_weights.",
        }
        if save_metrics:
            (seed_dir / "metrics.json").write_text(json.dumps(seed_payload, indent=2, default=str) + "\n")
            all_seeds_payloads.append(seed_payload)
            _flush_root_metrics()
            print(f"[saved] {seed_dir / 'metrics.json'}", flush=True)
            print(f"[saved] {out_root / 'metrics.json'} (aggregate, {len(all_seeds_payloads)} seed(s))", flush=True)
        all_seeds_summary.append({
            "seed": seed,
            "variants": str(args.variants),
            "mask_carry_in": bool(args.mask_carry_in),
            "ood_eval": ood_report,
            "ste_sel": ste_sel,
            "cvx_sel": cvx_sel,
            "stages": stages_payload,
        })

    if save_metrics:
        (out_root / "summary_all_seeds.json").write_text(json.dumps(all_seeds_summary, indent=2, default=str) + "\n")
        print(f"\n[done] Wrote summary_all_seeds.json and metrics.json under {out_root}", flush=True)
    else:
        print(f"\n[done] Run finished (--no_save_metrics); metrics JSON skipped. Output dir: {out_root}", flush=True)


if __name__ == "__main__":
    main()
