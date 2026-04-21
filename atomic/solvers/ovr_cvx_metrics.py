"""
NumPy-only metrics for multiclass CVX OVR (one binary problem per class).

Kept separate from ``loss_functions`` so solver modules can import without pulling
the full torch/cvxpy stack ordering that can break in some entrypoints.
"""

from __future__ import annotations

import numpy as np


def multiclass_ovr_cvx_data_loss(loss_name: str, scores: np.ndarray, y: np.ndarray) -> float:
    """
    Data-only loss for the multiclass CVX **OVR reduction** (one binary ``solve_binary_l1_primal_dual``
    per class on ±1 targets).

    This matches the term ``sum_c (1/n) sum_i l(y_bin, d_i^T w_c)`` optimized by that stack, and
    differs from ``LossFunction.hinge_ovr`` / ``squared`` on 2D logits, which use a **single** mean
    over ``n * num_classes`` entries (OVR-style targets but **without** the per-class ``(1/n)``
    scaling stacked from independent binary problems).

    For **squared**, the solver fits ±1 labels per column; the one-hot mean-squared metric is not
    the same objective and must not be used when reporting CVX OVR primal alignment.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected scores (N, C), got shape={scores.shape}.")
    n, num_classes = scores.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"Expected y shape (N,), got {y.shape} for scores {scores.shape}.")
    y_i = y.astype(np.int64)
    total = 0.0
    for c in range(num_classes):
        y_bin = np.where(y_i == c, 1.0, -1.0).astype(np.float64)
        s = scores[:, c]
        if loss_name in ("hinge", "hinge_ovr"):
            margins = y_bin * s
            total += float((1.0 / n) * np.sum(np.maximum(0.0, 1.0 - margins)))
        elif loss_name == "squared":
            total += float((1.0 / n) * np.sum((s - y_bin) ** 2))
        else:
            raise ValueError(f"multiclass_ovr_cvx_data_loss: unsupported loss_name={loss_name!r}.")
    return float(total)
