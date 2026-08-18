"""
Liquid State Machine (LSM) reservoir baseline — spiking-reservoir architecture.

Design
------
For a *spiking* reservoir the temporal state is carried by the LIF membrane
leak; there is no explicit recurrent weight matrix ``W_rec``. This aligns with
the SNN-reservoir convention used in Wijesinghe et al. 2019 / Zhou et al. 2020
and with the criticality-tuning literature (Bertschinger & Natschläger 2004;
Legenstein & Maass 2007; Wilting & Priesemann 2018): the reservoir's operating
point is set by the **branching ratio** sigma of the spike dynamics — not by
an ESN-style spectral radius — which is what
:mod:`atomic.solvers.lsm_criticality` measures and tunes.

Tunable reservoir knobs (all frozen after init; the readout is the only
trainable tensor):

* ``beta_leak``   — membrane retention factor (the effective recurrence/memory).
* ``threshold``   — LIF firing gate (controls firing rate / avalanche size).
* ``reservoir_variant`` — distribution of the frozen input weights ``W_in``:
  ``standard`` (i.i.d. N(0,1); bit-exact against
  :func:`solvers.cvx_solve._sample_weight_matrix` with the same seed, so R-CVX
  weight transfer is exact), ``normalized`` (column-normalized N(0,1); the
  standard ESN/LSM recipe), or ``orthogonal`` (QR of a Gaussian).

The **input scale** — a scalar multiplying every ``W_in`` — is applied post-hoc
by :func:`solvers.lsm_criticality._scaled_model` at model-build time, so the
reservoir's identity (its random draw, hence its CVX-interchange weight list)
is preserved up to a scalar gain. This is what lets R and R-CVX share the exact
same criticality-tuned reservoir while differing only in the readout objective.

Architecture layout
-------------------
Ditto structural copy of :class:`ste_parallel_Solve._SNNParallelBranch` /
:class:`ste_parallel_Solve.SNNBaselineSeq`: ``K_parallel`` independent branches,
each an ``L``-deep stack of ``snn.Leaky`` layers with dimensions
``[P_rec, P_rec, ..., P_last]`` (per-branch widths = total / K_parallel). The
per-branch ``.fcs`` / ``.lifs`` attributes are exposed exactly as in the STE
baseline so the pretrained-weight lists produced by
:func:`extract_lsm_weight_list` feed directly into
``InitializationConfig(pretrained_weights=...)`` for R-CVX.

Public surface
--------------
* :class:`LSMModelConfig`, :class:`LSMSolveConfig`, :class:`LSMSolveResult`.
* :class:`LSMBaselineSeq` — K-parallel LIF reservoir + linear readout.
* :func:`lsm_solve` — Adam/SGD fit of the linear readout (fallback; the
  canonical R baseline uses the closed-form ridge in
  :func:`solvers.lsm_criticality.lsm_ridge_solve` instead).
* :func:`extract_lsm_weight_list` — export ``[W_in per (branch, layer)...,
  classifier]`` for R-CVX weight transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss_functions import LossFunction


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LSMModelConfig:
    """LSM reservoir architecture spec.

    Criticality knobs (all set at build time; :mod:`lsm_criticality` sweeps
    them and picks the combination whose measured branching ratio sigma is
    closest to 1):

    * ``beta_leak`` — LIF membrane retention. Higher = longer memory.
    * ``threshold`` — LIF firing threshold. Higher = sparser spikes.
    * ``reservoir_variant`` — ``standard`` / ``normalized`` / ``orthogonal``
      distribution of the frozen ``W_in``. ``standard`` matches CVX Gaussian
      init exactly; the others are common LSM recipes.
    * ``reservoir_seed`` — deterministic seed for every frozen ``W_in`` draw.

    The **input scale** is *not* a config field on purpose: it is applied by
    :func:`solvers.lsm_criticality._scaled_model` as a post-hoc multiplier on
    ``fc.weight`` so the reservoir's random draw is preserved (only the input
    drive is rescaled), which is what makes the R / R-CVX weight interchange
    clean.
    """

    d_in: int
    num_classes: int
    L: int
    P_rec: int
    P_last: int
    K_parallel: int = 1
    beta_leak: float = 0.99
    threshold: float = 1.0
    last_layer_readout: str = "membrane"
    reservoir_seed: int = 0
    reservoir_variant: str = "standard"


@dataclass
class LSMSolveConfig:
    """Same schema as :class:`ste_solve.SteSolveConfig` so call sites can swap 1:1.

    Only used by the Adam/SGD fallback :func:`lsm_solve`; the canonical ridge
    readout in :mod:`lsm_criticality` ignores this dataclass and sweeps its
    own ``ridge_lambdas`` grid.
    """

    loss_name: str = "hinge_ovr"
    optimizer_name: str = "adam"
    lr: float = 1e-3
    epochs: int = 100
    batch_size: Optional[int] = None
    log_every: int = 10
    weight_decay: float = 0.0
    beta_path_reg: float = 0.0


@dataclass
class LSMSolveResult:
    model: nn.Module
    loss_history: List[float]
    best_losses: Dict[str, float]
    final_train_objective: float


# ---------------------------------------------------------------------------#
# helpers
# ---------------------------------------------------------------------------#


def _parallel_branch_width(total_width: int, k_parallel: int, name: str) -> int:
    if k_parallel <= 0:
        raise ValueError(f"K_parallel must be positive, got {k_parallel}.")
    if total_width % k_parallel != 0:
        raise ValueError(
            f"{name}={total_width} must be divisible by K_parallel={k_parallel} "
            "for equal-width parallel subnetworks."
        )
    return total_width // k_parallel


def _hidden_dims_like_snn_p2(L: int, P_rec: int, P_last: int) -> List[int]:
    if L <= 1:
        return [P_last]
    return [P_rec] * max(L - 2, 0) + [P_last]


def _col_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=0, keepdims=True) + eps
    return mat / norms


def _sample_input_matrix(
    rng: np.random.Generator, in_dim: int, out_dim: int, variant: str
) -> np.ndarray:
    """Sample a frozen ``(in_dim, out_dim)`` input weight matrix ``W_in``.

    Matches :func:`solvers.cvx_solve._sample_weight_matrix` for
    ``variant='standard'`` so R-CVX weight transfer is bit-exact against CVX
    Gaussian init with the same seed.
    """
    w = rng.standard_normal((in_dim, out_dim)).astype(np.float64)
    if variant == "standard":
        return w
    if variant == "normalized":
        return _col_normalize(w)
    if variant == "orthogonal":
        q, _ = np.linalg.qr(w)
        if q.shape[1] >= out_dim:
            return q[:, :out_dim]
        extra = rng.standard_normal((in_dim, out_dim - q.shape[1])).astype(np.float64)
        return np.concatenate([q, _col_normalize(extra)], axis=1)
    raise ValueError(f"Unknown reservoir_variant={variant}.")


def _reservoir_layer_seed(base_seed: int, branch_idx: int, layer_idx: int) -> int:
    """Per-(branch, layer) seed derived from the config's base seed."""
    x = int(base_seed) * 1_000_003 + int(branch_idx) * 10_007 + int(layer_idx) * 97
    return int(x) & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------#
# LSM branch + full model
# ---------------------------------------------------------------------------#


class _LSMReservoirBranch(nn.Module):
    """Per-branch frozen reservoir: ``L`` stacked ``snn.Leaky`` layers with fixed ``W_in``.

    Attribute layout mirrors ``ste_parallel_Solve._SNNParallelBranch`` — ``.fcs``
    is an ``nn.ModuleList`` of ``nn.Linear(in_dim, h, bias=False)`` and
    ``.lifs`` is an ``nn.ModuleList`` of ``snn.Leaky``. The only difference is
    ``fc.weight.requires_grad_(False)`` in ``__init__``, so a plain
    ``optimizer = torch.optim.Adam(model.parameters(), ...)`` naturally trains
    only the classifier head above.

    The :mod:`lsm_criticality` module accesses ``.fcs`` and ``.lifs`` directly
    (spike-train capture, feature capture, input-scale multiplication), which
    is why this layout is exactly the STE-baseline shape.
    """

    def __init__(
        self,
        *,
        d_in: int,
        hidden_dims: List[int],
        beta_leak: float,
        threshold: float,
        reservoir_seed: int,
        branch_idx: int,
        variant: str,
    ) -> None:
        super().__init__()
        fcs: List[nn.Module] = []
        lifs: List[nn.Module] = []
        in_dim = int(d_in)
        for l, h_dim in enumerate(hidden_dims):
            layer_seed = _reservoir_layer_seed(reservoir_seed, branch_idx, l)
            rng = np.random.default_rng(layer_seed)
            fc = nn.Linear(in_dim, int(h_dim), bias=False)
            W_np = _sample_input_matrix(rng, in_dim=in_dim, out_dim=int(h_dim), variant=variant)
            with torch.no_grad():
                fc.weight.copy_(torch.from_numpy(W_np.T).to(fc.weight.dtype))
            fc.weight.requires_grad_(False)
            fcs.append(fc)
            lifs.append(snn.Leaky(beta=float(beta_leak), threshold=float(threshold)))
            in_dim = int(h_dim)
        self.fcs = nn.ModuleList(fcs)
        self.lifs = nn.ModuleList(lifs)


class LSMBaselineSeq(nn.Module):
    """K-parallel LSM: ``K_parallel`` frozen reservoir branches + trained linear readout.

    Forward
    -------
    Per-timestep, per-branch, per-layer ``snn.Leaky`` step (matches
    ``ste_parallel_Solve.SNNBaselineSeq`` exactly). Every branch's last-layer
    readout (membrane or spike per ``last_layer_readout``) is concatenated to
    length ``P_last`` and fed into the shared ``classifier`` to produce logits.
    """

    def __init__(self, cfg: LSMModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.k_parallel = int(cfg.K_parallel)
        sub_p_rec = _parallel_branch_width(int(cfg.P_rec), self.k_parallel, "P_rec")
        sub_p_last = _parallel_branch_width(int(cfg.P_last), self.k_parallel, "P_last")
        hidden_dims = _hidden_dims_like_snn_p2(int(cfg.L), sub_p_rec, sub_p_last)
        self.hidden_dims = list(hidden_dims)
        self.last_layer_readout = str(cfg.last_layer_readout)
        self.branches = nn.ModuleList(
            [
                _LSMReservoirBranch(
                    d_in=int(cfg.d_in),
                    hidden_dims=self.hidden_dims,
                    beta_leak=float(cfg.beta_leak),
                    threshold=float(cfg.threshold),
                    reservoir_seed=int(cfg.reservoir_seed),
                    branch_idx=int(k),
                    variant=str(cfg.reservoir_variant),
                )
                for k in range(self.k_parallel)
            ]
        )
        self.classifier = nn.Linear(int(cfg.P_last), int(cfg.num_classes), bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, T, _ = x_seq.shape
        branch_mems = [
            [lif.init_leaky().to(x_seq.device) for lif in branch.lifs] for branch in self.branches
        ]
        logits_seq: List[torch.Tensor] = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            readouts: List[torch.Tensor] = []
            for b_idx, branch in enumerate(self.branches):
                h = x_t
                last_spk: Optional[torch.Tensor] = None
                last_mem: Optional[torch.Tensor] = None
                for i, (fc, lif) in enumerate(zip(branch.fcs, branch.lifs)):
                    spk, mem = lif(fc(h), branch_mems[b_idx][i])
                    branch_mems[b_idx][i] = mem
                    h = spk
                    last_spk = spk
                    last_mem = mem
                readouts.append(
                    last_mem if self.last_layer_readout == "membrane" else last_spk
                )
            logits_seq.append(self.classifier(torch.cat(readouts, dim=1)))
        return torch.stack(logits_seq, dim=1)


# ---------------------------------------------------------------------------#
# Regularization / eval helpers
# ---------------------------------------------------------------------------#


def _lsm_readout_reg(model: LSMBaselineSeq) -> torch.Tensor:
    return torch.sqrt((model.classifier.weight ** 2).sum() + 1e-12)


def _compute_eval_loss(
    model: LSMBaselineSeq,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_name: str,
    beta_path_reg: float = 0.0,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = LossFunction.compute(name=loss_name, y=y, f_x=logits).value
        if beta_path_reg > 0.0:
            loss = loss + float(beta_path_reg) * _lsm_readout_reg(model)
        return float(loss.item())


def _sequence_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.ndim != 3:
        raise ValueError(f"Expected sequence logits with shape (N,T,C), got {tuple(logits.shape)}.")
    if y.ndim == 1:
        last_logits = logits[:, -1, :]
        if last_logits.shape[1] == 1:
            preds = (last_logits[:, 0] >= 0.0).long()
        else:
            preds = torch.argmax(last_logits, dim=1)
        return float((preds == y.long()).float().mean().item())
    if y.ndim == 2:
        if tuple(logits.shape[:2]) != tuple(y.shape):
            raise ValueError(
                f"Sequence label shape mismatch: logits={tuple(logits.shape)}, y={tuple(y.shape)}."
            )
        if logits.shape[2] == 1:
            preds = (logits[:, :, 0] >= 0.0).long()
        else:
            preds = torch.argmax(logits, dim=2)
        return float((preds == y.long()).float().mean().item())
    raise ValueError(f"Expected labels with rank 1 or 2, got rank {y.ndim}.")


def _compute_eval_accuracy(model: LSMBaselineSeq, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return _sequence_accuracy_from_logits(logits, y)


def _token_seq_stats_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if logits.ndim != 3 or y.ndim != 2:
        raise ValueError(f"Expected logits rank-3 and y rank-2, got {tuple(logits.shape)} and {tuple(y.shape)}.")
    if tuple(logits.shape[:2]) != tuple(y.shape):
        raise ValueError(
            f"Token/seq stat shape mismatch: logits={tuple(logits.shape)}, y={tuple(y.shape)}."
        )
    if logits.shape[2] == 1:
        preds = (logits[:, :, 0] >= 0.0).long()
    else:
        preds = torch.argmax(logits, dim=2)
    match = preds == y.long()
    token_acc = float(match.float().mean().item())
    seq_acc = float(match.all(dim=1).float().mean().item())
    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "token_loss": float(1.0 - token_acc),
        "seq_loss": float(1.0 - seq_acc),
    }


def _compute_eval_token_seq_stats(
    model: LSMBaselineSeq, x: torch.Tensor, y: torch.Tensor
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return _token_seq_stats_from_logits(logits, y)


def _apply_pretrained_weights(
    model: LSMBaselineSeq, pretrained_weights: List[np.ndarray]
) -> None:
    """Overwrite ``fc.weight`` matrices and the classifier from a numpy weight list.

    Layout: branch-major ``[W_in[0], W_in[1], ..., classifier]``. Provided for
    interchange with the STE surrogate baseline's export format so an LSM can
    be seeded from a STE-pretrain checkpoint.
    """
    n_fcs_total = sum(len(b.fcs) for b in model.branches)
    expected = n_fcs_total + 1
    if len(pretrained_weights) != expected:
        raise ValueError(f"Expected {expected} tensors in pretrained_weights, got {len(pretrained_weights)}.")
    with torch.no_grad():
        flat_idx = 0
        for b_idx, branch in enumerate(model.branches):
            for l_idx, fc in enumerate(branch.fcs):
                src = torch.from_numpy(pretrained_weights[flat_idx]).to(
                    fc.weight.device, dtype=fc.weight.dtype
                )
                if tuple(src.shape) != tuple(fc.weight.shape):
                    raise ValueError(
                        f"Branch-{b_idx} layer-{l_idx} fc.weight shape mismatch: "
                        f"{tuple(src.shape)} != {tuple(fc.weight.shape)}"
                    )
                fc.weight.copy_(src)
                fc.weight.requires_grad_(False)
                flat_idx += 1
        src_cls = torch.from_numpy(pretrained_weights[-1]).to(
            model.classifier.weight.device, dtype=model.classifier.weight.dtype
        )
        if tuple(src_cls.shape) != tuple(model.classifier.weight.shape):
            raise ValueError("Classifier weight shape mismatch in pretrained initialization.")
        model.classifier.weight.copy_(src_cls)


# ---------------------------------------------------------------------------#
# lsm_solve — Adam/SGD training of the readout (fallback; use ridge in
# lsm_criticality for the canonical R baseline)
# ---------------------------------------------------------------------------#


def lsm_solve(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_cfg: LSMModelConfig,
    solve_cfg: LSMSolveConfig,
    pretrained_weights: Optional[List[np.ndarray]] = None,
    device: Optional[torch.device] = None,
) -> LSMSolveResult:
    """Adam/SGD fit of the LSM's linear readout (fallback path).

    The canonical R baseline uses closed-form ridge instead — see
    :func:`solvers.lsm_criticality.lsm_ridge_solve`. This function is kept for
    (a) API parity with :func:`ste_solve.ste_solve` and (b) hinge/CE loss
    experiments that don't map naturally onto ridge regression.

    Only ``classifier.weight`` is trainable — every ``fc.weight`` in every
    branch is set to ``requires_grad=False`` at construction time, so the
    optimizer's parameter list is filtered to exactly that one tensor.
    """
    run_device = choose_device() if device is None else device
    model = LSMBaselineSeq(model_cfg).to(run_device)
    if pretrained_weights is not None:
        _apply_pretrained_weights(model, pretrained_weights)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=run_device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=run_device)
    test_x = torch.tensor(x_test, dtype=torch.float32, device=run_device)
    train_y = torch.tensor(y_train, dtype=torch.long, device=run_device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=run_device)
    test_y = torch.tensor(y_test, dtype=torch.long, device=run_device)

    n_train = int(train_x.shape[0])
    if solve_cfg.batch_size is not None and int(solve_cfg.batch_size) != n_train:
        raise ValueError(
            f"LSM enforces full-batch training: expected batch_size={n_train}, "
            f"got {int(solve_cfg.batch_size)}."
        )
    batch_size = n_train
    print(
        (
            f"[lsm-run] K_parallel={int(model_cfg.K_parallel)} L={int(model_cfg.L)} "
            f"P_rec={int(model_cfg.P_rec)} P_last={int(model_cfg.P_last)} "
            f"beta_leak={float(model_cfg.beta_leak):.4g} threshold={float(model_cfg.threshold):.4g} "
            f"reservoir_seed={int(model_cfg.reservoir_seed)} "
            f"reservoir_variant={model_cfg.reservoir_variant} "
            f"readout={model_cfg.last_layer_readout} "
            f"beta_path_reg={float(solve_cfg.beta_path_reg):.6g} "
            f"lr={float(solve_cfg.lr):.6g} weight_decay={float(solve_cfg.weight_decay):.6g} "
            f"batch_size=full({batch_size}) n_train={n_train} epochs={int(solve_cfg.epochs)}"
        ),
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("LSM has no trainable parameters — expected classifier head to be trainable.")

    loss_history: List[float] = []
    best_state = None
    n_samples = n_train
    if int(solve_cfg.epochs) == 0:
        if pretrained_weights is None:
            raise ValueError("lsm_solve(epochs=0) requires pretrained_weights (no training steps).")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        if solve_cfg.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_params, lr=solve_cfg.lr, weight_decay=solve_cfg.weight_decay
            )
        elif solve_cfg.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params, lr=solve_cfg.lr, weight_decay=solve_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer_name={solve_cfg.optimizer_name}.")
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        best_val = float("inf")
        for epoch in range(1, solve_cfg.epochs + 1):
            model.train()
            permutation = torch.randperm(n_samples, device=run_device)
            epoch_loss_accum = 0.0
            for start in range(0, n_samples, batch_size):
                idx = permutation[start : start + batch_size]
                logits = model(train_x[idx])
                loss = LossFunction.compute(name=solve_cfg.loss_name, y=train_y[idx], f_x=logits).value
                if solve_cfg.beta_path_reg > 0.0:
                    loss = loss + float(solve_cfg.beta_path_reg) * _lsm_readout_reg(model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_accum += float(loss.item()) * int(idx.numel())
            epoch_loss = epoch_loss_accum / float(n_samples)
            val_loss = _compute_eval_loss(
                model, val_x, val_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
            )
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if solve_cfg.log_every > 0 and epoch % solve_cfg.log_every == 0:
                train_acc = _compute_eval_accuracy(model, train_x, train_y)
                val_acc = _compute_eval_accuracy(model, val_x, val_y)
                loss_history.append(epoch_loss)
                print(
                    (
                        f"[lsm] epoch={epoch}/{solve_cfg.epochs} "
                        f"train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} "
                        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                    ),
                    flush=True,
                )
        if best_state is None:
            raise RuntimeError("Training did not produce any best_state.")
    model.load_state_dict(best_state)

    final_train_objective = _compute_eval_loss(
        model, train_x, train_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
    )
    best_losses = {
        "train_loss": final_train_objective,
        "train_objective": final_train_objective,
        "val_loss": _compute_eval_loss(
            model, val_x, val_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
        ),
        "test_loss": _compute_eval_loss(
            model, test_x, test_y, solve_cfg.loss_name, beta_path_reg=solve_cfg.beta_path_reg
        ),
    }
    if train_y.ndim == 2 and val_y.ndim == 2 and test_y.ndim == 2:
        tr_stats = _compute_eval_token_seq_stats(model, train_x, train_y)
        va_stats = _compute_eval_token_seq_stats(model, val_x, val_y)
        te_stats = _compute_eval_token_seq_stats(model, test_x, test_y)
        best_losses.update(
            {
                "train_token_acc": tr_stats["token_acc"],
                "val_token_acc": va_stats["token_acc"],
                "test_token_acc": te_stats["token_acc"],
                "train_seq_acc": tr_stats["seq_acc"],
                "val_seq_acc": va_stats["seq_acc"],
                "test_seq_acc": te_stats["seq_acc"],
                "train_token_loss": tr_stats["token_loss"],
                "val_token_loss": va_stats["token_loss"],
                "test_token_loss": te_stats["token_loss"],
                "train_seq_loss": tr_stats["seq_loss"],
                "val_seq_loss": va_stats["seq_loss"],
                "test_seq_loss": te_stats["seq_loss"],
            }
        )

    return LSMSolveResult(
        model=model,
        loss_history=loss_history,
        best_losses=best_losses,
        final_train_objective=final_train_objective,
    )


# ---------------------------------------------------------------------------#
# R-CVX interop
# ---------------------------------------------------------------------------#


def extract_lsm_weight_list(model: LSMBaselineSeq) -> List[np.ndarray]:
    """Branch-major numpy list ``[W_in per (branch, layer)..., classifier]``.

    Consumed by the R-CVX weight-transfer path (i.e. fed into
    :class:`solvers.cvx_solve.InitializationConfig(pretrained_weights=...)`).
    Since this LSM has no ``W_rec``, the transfer is bit-exact against CVX
    Gaussian init when ``reservoir_variant='standard'`` and the input-scale
    multiplier is folded into ``fc.weight`` before export (which is what
    :func:`solvers.lsm_criticality._scaled_model` does).
    """
    weights: List[np.ndarray] = []
    for branch in model.branches:
        for fc in branch.fcs:
            weights.append(fc.weight.detach().cpu().numpy().copy())
    weights.append(model.classifier.weight.detach().cpu().numpy().copy())
    return weights
