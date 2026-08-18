"""
Criticality tuning and ridge readout for the LSM reservoir baseline.

Why this module exists
----------------------
For a *spiking* reservoir with no explicit recurrent weight matrix (this
architecture carries temporal state only through the LIF membrane leak), the
standard criticality criterion is NOT the ESN spectral radius --- there is no
W_rec to rescale --- but the **branching ratio** sigma of the spike dynamics
(Maass/Legenstein edge-of-chaos; Beggs & Plenz avalanche criticality; Wilting
& Priesemann 2018 multistep estimator). A reservoir is critical when sigma
~= 1: each spike triggers on average one spike at the next step, so activity
neither dies out (sigma < 1, subcritical) nor explodes (sigma > 1,
supercritical). Criticality maximizes the separation property and memory of
the liquid.

The tunable knobs that actually exist in this architecture are:

* ``beta_leak``   -- membrane retention (the effective recurrence/memory).
* ``threshold``   -- firing gate (controls firing rate / avalanche size).
* ``input_scale`` -- gain on the frozen W_in (drives the reservoir harder/softer).

This module:

1. :func:`estimate_branching_ratio` -- measures sigma from spike trains produced
   by a reservoir on real input, by regressing per-step descendant spike counts
   on ancestor spike counts (the standard regression estimator; robust to
   subsampling).
2. :func:`tune_reservoir_criticality` -- sweeps (beta_leak, threshold,
   input_scale) and selects the config whose measured sigma is closest to 1,
   returning a criticality-tuned :class:`LSMModelConfig` plus the selected
   input-scale multiplier and the full sweep table (for the appendix).
3. :func:`lsm_ridge_solve` -- the canonical LSM readout: a closed-form ridge
   (L2) regression on the frozen reservoir features. This is the "R" condition
   in the R / R-CVX design; R-CVX keeps the same tuned reservoir but swaps the
   ridge readout for the convex L1 solve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .LSM import (
    LSMBaselineSeq,
    LSMModelConfig,
    LSMSolveResult,
    _compute_eval_token_seq_stats,
    _sequence_accuracy_from_logits,
    choose_device,
)


# ----------------------------------------------------------------------------- #
# 1. Branching-ratio estimation                                                 #
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def _collect_branch_spike_trains(
    model: LSMBaselineSeq, x_seq: torch.Tensor
) -> List[torch.Tensor]:
    """Run the reservoir and return, per branch, the last-layer spike train.

    Returns a list of length ``K_parallel``; each entry is a ``(N, T, width)``
    binary spike tensor for that branch's terminal LIF layer. We instrument the
    same forward recursion as ``LSMBaselineSeq.forward`` but capture spikes
    rather than the membrane/spike readout, so the measured dynamics are exactly
    the ones the readout sees.
    """
    model.eval()
    _, steps, _ = x_seq.shape
    branch_mems = [
        [lif.init_leaky().to(x_seq.device) for lif in branch.lifs]
        for branch in model.branches
    ]
    per_branch_spikes: List[List[torch.Tensor]] = [[] for _ in model.branches]
    for t in range(steps):
        x_t = x_seq[:, t, :]
        for b_idx, branch in enumerate(model.branches):
            h = x_t
            last_spk: Optional[torch.Tensor] = None
            for i, (fc, lif) in enumerate(zip(branch.fcs, branch.lifs)):
                spk, mem = lif(fc(h), branch_mems[b_idx][i])
                branch_mems[b_idx][i] = mem
                h = spk
                last_spk = spk
            assert last_spk is not None, "Branch had zero layers — impossible for L >= 1."
            per_branch_spikes[b_idx].append(last_spk)
    return [torch.stack(s, dim=1) for s in per_branch_spikes]  # each (N, T, width)


def estimate_branching_ratio(
    model: LSMBaselineSeq,
    x_seq: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Estimate the branching ratio sigma of the reservoir on ``x_seq``.

    Definition. Let ``a_t`` be the total number of spikes emitted across the
    reservoir at step ``t`` (summed over neurons and, here, averaged over the
    batch to reduce variance). The branching ratio is the expected number of
    step-(t+1) spikes triggered per step-t spike, estimated by the
    least-squares slope of ``a_{t+1}`` on ``a_t`` over the sequence:

        sigma = sum_t (a_t * a_{t+1}) / sum_t (a_t^2).

    This is the standard regression estimator for sigma (equivalent to the
    Wilting-Priesemann multistep estimator at lag 1). ``sigma < 1`` is
    subcritical, ``sigma > 1`` supercritical, ``sigma ~= 1`` critical.

    We aggregate spike counts across all branches (they are independent
    reservoirs, so their activity simply adds) and across the batch.
    """
    spike_trains = _collect_branch_spike_trains(model, x_seq)  # list of (N,T,w)
    total: Optional[torch.Tensor] = None
    for st in spike_trains:
        counts = st.sum(dim=2)  # (N, T)
        total = counts if total is None else total + counts
    assert total is not None
    a = total.float().mean(dim=0)  # (T,)
    if a.numel() < 2:
        return float("nan")
    a_t = a[:-1]
    a_next = a[1:]
    denom = float((a_t * a_t).sum().item())
    if denom < eps:
        return 0.0  # reservoir silent: treat as maximally subcritical
    sigma = float((a_t * a_next).sum().item()) / denom
    return sigma


# ----------------------------------------------------------------------------- #
# 2. Criticality tuning sweep                                                   #
# ----------------------------------------------------------------------------- #

@dataclass
class CriticalityGrid:
    """Search grid for the criticality knobs.

    Defaults span sub- to super-critical regimes for typical LIF settings. The
    input-scale multiplies the frozen ``W_in`` at measurement time (see
    :func:`_scaled_model`), so it does not change which reservoir is stored ---
    only how hard it is driven --- keeping the frozen-weight interchange with
    CVX intact.
    """

    beta_leak: Sequence[float] = (0.90, 0.95, 0.99, 0.995)
    threshold: Sequence[float] = (0.5, 1.0, 1.5, 2.0)
    input_scale: Sequence[float] = (0.5, 1.0, 2.0, 4.0)


@dataclass
class CriticalityResult:
    tuned_config: LSMModelConfig
    input_scale: float
    branching_ratio: float
    table: List[Dict[str, float]]  # every (beta,thr,scale) -> sigma, for logging


def _scaled_model(cfg: LSMModelConfig, input_scale: float, device: torch.device) -> LSMBaselineSeq:
    """Build an LSM from ``cfg`` and multiply every frozen W_in by ``input_scale``.

    Scaling the input weights is the criticality knob that leaves the *identity*
    of the reservoir (its random draw, hence its CVX-interchange weight list)
    unchanged up to a scalar gain, while moving the operating point through the
    branching-ratio axis.
    """
    model = LSMBaselineSeq(cfg).to(device)
    if input_scale != 1.0:
        with torch.no_grad():
            for branch in model.branches:
                for fc in branch.fcs:
                    fc.weight.mul_(float(input_scale))
    return model


def tune_reservoir_criticality(
    base_cfg: LSMModelConfig,
    x_probe: np.ndarray,
    *,
    grid: Optional[CriticalityGrid] = None,
    target_sigma: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> CriticalityResult:
    """Select (beta_leak, threshold, input_scale) whose sigma is closest to 1.

    ``x_probe`` is a representative input batch (e.g. the training inputs or a
    subsample). We measure the branching ratio for every grid point on this
    fixed probe and pick the configuration minimizing ``|sigma - target_sigma|``.

    Returns a :class:`CriticalityResult` whose ``tuned_config`` has the selected
    ``beta_leak`` and ``threshold`` folded in, plus the selected ``input_scale``
    (applied to W_in at model-build time via :func:`build_tuned_model`). The full
    (knob -> sigma) table is returned for reporting in the appendix, which is
    exactly the "tuned by the criticality criterion" evidence the rebuttal
    promises.
    """
    grid = grid or CriticalityGrid()
    run_device = choose_device() if device is None else device
    probe = torch.tensor(x_probe, dtype=torch.float32, device=run_device)

    table: List[Dict[str, float]] = []
    best_key: Tuple[float, float, float, float, float] | None = None  # (dev, beta, thr, scale, sigma)
    for beta in grid.beta_leak:
        for thr in grid.threshold:
            cfg = replace(base_cfg, beta_leak=float(beta), threshold=float(thr))
            for scale in grid.input_scale:
                model = _scaled_model(cfg, float(scale), run_device)
                sigma = estimate_branching_ratio(model, probe)
                dev = abs(sigma - target_sigma) if np.isfinite(sigma) else float("inf")
                table.append(
                    {
                        "beta_leak": float(beta),
                        "threshold": float(thr),
                        "input_scale": float(scale),
                        "branching_ratio": float(sigma),
                        "abs_dev_from_target": float(dev),
                    }
                )
                if verbose:
                    print(
                        f"[criticality] beta={beta:.3f} thr={thr:.2f} scale={scale:.2f} "
                        f"-> sigma={sigma:.4f} (|dev|={dev:.4f})",
                        flush=True,
                    )
                if best_key is None or dev < best_key[0]:
                    best_key = (float(dev), float(beta), float(thr), float(scale), float(sigma))
    assert best_key is not None
    _, beta_sel, thr_sel, scale_sel, sigma_sel = best_key
    tuned_cfg = replace(base_cfg, beta_leak=beta_sel, threshold=thr_sel)
    if verbose:
        print(
            f"[criticality] SELECTED beta={beta_sel:.3f} thr={thr_sel:.2f} "
            f"input_scale={scale_sel:.2f} sigma={sigma_sel:.4f}",
            flush=True,
        )
    return CriticalityResult(
        tuned_config=tuned_cfg,
        input_scale=scale_sel,
        branching_ratio=sigma_sel,
        table=table,
    )


def build_tuned_model(
    result: CriticalityResult, device: Optional[torch.device] = None
) -> LSMBaselineSeq:
    """Materialize the criticality-tuned reservoir (knobs + input scaling applied)."""
    run_device = choose_device() if device is None else device
    return _scaled_model(result.tuned_config, result.input_scale, run_device)


# ----------------------------------------------------------------------------- #
# 3. Ridge (L2) readout --- the canonical "R" condition                          #
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def _reservoir_features(model: LSMBaselineSeq, x_seq: torch.Tensor) -> torch.Tensor:
    """Return the concatenated last-layer readout features, shape ``(N, T, P_last)``.

    This is exactly the tensor ``LSMBaselineSeq.forward`` feeds to its classifier,
    captured before the linear head so a closed-form ridge solve can replace SGD.
    """
    model.eval()
    _, steps, _ = x_seq.shape
    branch_mems = [
        [lif.init_leaky().to(x_seq.device) for lif in branch.lifs]
        for branch in model.branches
    ]
    feats: List[torch.Tensor] = []
    for t in range(steps):
        x_t = x_seq[:, t, :]
        readouts: List[torch.Tensor] = []
        for b_idx, branch in enumerate(model.branches):
            h = x_t
            last_spk: Optional[torch.Tensor] = None
            last_mem: Optional[torch.Tensor] = None
            for i, (fc, lif) in enumerate(zip(branch.fcs, branch.lifs)):
                spk, mem = lif(fc(h), branch_mems[b_idx][i])
                branch_mems[b_idx][i] = mem
                h = spk
                last_spk = spk
                last_mem = mem
            readouts.append(last_mem if model.last_layer_readout == "membrane" else last_spk)
        feats.append(torch.cat(readouts, dim=1))
    return torch.stack(feats, dim=1)  # (N, T, P_last)


def _pm_one_targets(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """Build the +/-1 one-vs-rest regression target matrix (rank-1 labels).

    Standard Lukoševičius-Jaeger ridge-classification code: correct class -> +1,
    all others -> -1. Prediction is ``argmax(Phi @ W)`` regardless of the sign
    convention, so this reduces to Fisher's linear discriminant in the OVR case.
    """
    Y = -torch.ones((y.shape[0], num_classes), device=device)
    Y[torch.arange(y.shape[0]), y.long()] = 1.0
    return Y


def lsm_ridge_solve(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: LSMBaselineSeq,
    num_classes: int,
    ridge_lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    device: Optional[torch.device] = None,
) -> LSMSolveResult:
    """Closed-form ridge readout on frozen reservoir features (the "R" baseline).

    Fits ``W_cls`` by ridge regression on the reservoir features and writes the
    solution back into ``model.classifier`` so downstream evaluation (accuracy,
    token/seq stats) uses the identical code path as the Adam LSM and the CVX
    methods. ``ridge_lambdas`` is swept and selected on validation MSE --- the
    L2 analog of the CVX lambda sweep, so R and R-CVX are tuned on the same
    footing.

    Readout target: for final-step classification (rank-1 labels) we regress the
    last-timestep feature onto one-vs-rest +/-1 targets. For per-token sequence
    labels (rank-2) we stack all timesteps and regress token-wise. This mirrors
    how the CVX readout consumes the same feature matrix.
    """
    run_device = choose_device() if device is None else device
    model = model.to(run_device)

    def feats_of(x: np.ndarray) -> torch.Tensor:
        xf = torch.tensor(x, dtype=torch.float32, device=run_device)
        return _reservoir_features(model, xf)  # (N,T,P_last)

    F_tr = feats_of(x_train)
    F_va = feats_of(x_val)
    F_te = feats_of(x_test)
    P_last = F_tr.shape[2]

    y_tr_t = torch.tensor(y_train, device=run_device)
    y_va_t = torch.tensor(y_val, device=run_device)
    y_te_t = torch.tensor(y_test, device=run_device)
    seq_labels = (y_tr_t.ndim == 2)

    if seq_labels:
        Phi = F_tr.reshape(-1, P_last)
        yt = y_tr_t.reshape(-1)
        Y = -torch.ones((Phi.shape[0], num_classes), device=run_device)
        Y[torch.arange(Phi.shape[0]), yt.long()] = 1.0
    else:
        Phi = F_tr[:, -1, :]  # last-timestep features
        Y = _pm_one_targets(y_tr_t, num_classes, run_device)

    # Precompute the Gram matrix once; sweep lambda cheaply.
    G = Phi.t() @ Phi                      # (P_last, P_last)
    RHS = Phi.t() @ Y                      # (P_last, C)
    eye = torch.eye(P_last, device=run_device, dtype=G.dtype)

    def val_mse(W: torch.Tensor) -> float:
        if seq_labels:
            Fv = F_va.reshape(-1, P_last)
            yv = y_va_t.reshape(-1)
            Yv = -torch.ones((Fv.shape[0], num_classes), device=run_device)
            Yv[torch.arange(Fv.shape[0]), yv.long()] = 1.0
            pred = Fv @ W
            return float(((pred - Yv) ** 2).mean().item())
        Fv = F_va[:, -1, :]
        Yv = _pm_one_targets(y_va_t, num_classes, run_device)
        pred = Fv @ W
        return float(((pred - Yv) ** 2).mean().item())

    best: Tuple[float, float, torch.Tensor] | None = None
    for lam in ridge_lambdas:
        W = torch.linalg.solve(G + float(lam) * eye, RHS)  # (P_last, C)
        vm = val_mse(W)
        print(f"[lsm-ridge] lambda={lam:.4g} val_mse={vm:.6f}", flush=True)
        if best is None or vm < best[0]:
            best = (vm, float(lam), W)
    assert best is not None
    val_mse_sel, lam_sel, W_sel = best
    print(f"[lsm-ridge] SELECTED lambda={lam_sel:.4g}", flush=True)

    with torch.no_grad():
        model.classifier.weight.copy_(W_sel.t().contiguous())

    def acc3(Feat3d: torch.Tensor, y_ref: torch.Tensor) -> float:
        logits = Feat3d @ W_sel  # (N,T,C)
        return _sequence_accuracy_from_logits(logits, y_ref)

    best_losses: Dict[str, float] = {
        "ridge_lambda": lam_sel,
        "val_mse": val_mse_sel,
        "train_acc": acc3(F_tr, y_tr_t),
        "val_acc": acc3(F_va, y_va_t),
        "test_acc": acc3(F_te, y_te_t),
    }
    if seq_labels:
        for split, x_np, y_ref in (
            ("train", x_train, y_tr_t),
            ("val", x_val, y_va_t),
            ("test", x_test, y_te_t),
        ):
            stats = _compute_eval_token_seq_stats(
                model, torch.tensor(x_np, dtype=torch.float32, device=run_device), y_ref
            )
            best_losses[f"{split}_token_acc"] = stats["token_acc"]
            best_losses[f"{split}_seq_acc"] = stats["seq_acc"]

    return LSMSolveResult(
        model=model,
        loss_history=[],
        best_losses=best_losses,
        final_train_objective=float(val_mse_sel),
    )


# ----------------------------------------------------------------------------- #
# 4. End-to-end convenience: tune -> ridge (the full "R" pipeline)               #
# ----------------------------------------------------------------------------- #

def run_R_baseline(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    base_cfg: LSMModelConfig,
    num_classes: int,
    grid: Optional[CriticalityGrid] = None,
    ridge_lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    probe_subsample: Optional[int] = 256,
    device: Optional[torch.device] = None,
) -> Tuple[LSMSolveResult, CriticalityResult]:
    """Full R condition: criticality-tune the reservoir, then ridge-fit the readout.

    Returns ``(ridge_result, criticality_result)``. To obtain R-CVX, take
    ``criticality_result.tuned_config`` and ``.input_scale``, build the same
    reservoir, export its weights with :func:`solvers.LSM.extract_lsm_weight_list`,
    and feed them to ``cvx_solve`` as ``pretrained_weights`` --- the reservoir is
    identical, only the readout differs (ridge here, convex L1 there), which is
    the clean R-vs-R-CVX readout isolation the rebuttal describes.
    """
    run_device = choose_device() if device is None else device
    probe = x_train
    if probe_subsample is not None and x_train.shape[0] > probe_subsample:
        idx = np.random.default_rng(0).choice(x_train.shape[0], size=probe_subsample, replace=False)
        probe = x_train[idx]

    crit = tune_reservoir_criticality(base_cfg, probe, grid=grid, device=run_device)
    model = build_tuned_model(crit, device=run_device)
    ridge = lsm_ridge_solve(
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        model=model, num_classes=num_classes,
        ridge_lambdas=ridge_lambdas, device=run_device,
    )
    return ridge, crit
