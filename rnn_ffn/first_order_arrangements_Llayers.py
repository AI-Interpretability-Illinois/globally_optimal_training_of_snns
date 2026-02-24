# -*- coding: utf-8 -*-
"""
First-order Convex-LASSO training for L-layer threshold networks using stored hyperplane witnesses.

Changes vs your original file:
- Store hyperplane weights (witnesses) explicitly (including representation matrix H).
- Replace CVXPY solvers with first-order solvers:
    * squared loss + L1 via FISTA
    * hinge loss + L1 via proximal subgradient (optional)
- Generalize arrangement construction to L layers (feedforward).
- Keep "reference .py" structure: build patterns on train, store U, reuse U to build test patterns.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------- utils -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_sign_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    y_signed = np.where(y > 0, 1.0, -1.0)
    return y_signed.astype(np.float32)


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def power_iteration_spectral_norm_sq(D: np.ndarray, iters: int = 50, seed: int = 0) -> float:
    """Approximate ||D||_2^2 via power iteration on (D^T D)."""
    rng = np.random.default_rng(seed)
    p = D.shape[1]
    v = rng.normal(size=(p,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = D.T @ (D @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    dv = D @ v
    return float(dv @ dv)


# ----------------------------- datasets -----------------------------
def generate_parity(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, d)).astype(np.float32)
    parity = (x.sum(axis=1) % 2) * 2 - 1
    return x, parity.astype(np.float32)


def generate_gaussian_mixture(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(2, d)).astype(np.float32)
    labels = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, d)).astype(np.float32) + means[labels]
    y = labels * 2 - 1
    return x, y.astype(np.float32)


# ----------------------------- representation + arrangements -----------------------------
@dataclass
class RepresentationSpec:
    H: np.ndarray  # (d, m_repr)


def build_representation_matrix(x: np.ndarray, m_repr: int, seed: int) -> Tuple[np.ndarray, RepresentationSpec]:
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(x.shape[1], m_repr)).astype(np.float32)
    x_tilde = (x @ H >= 0).astype(np.float32)
    return x_tilde, RepresentationSpec(H=H)


def transform_representation_matrix(x: np.ndarray, spec: RepresentationSpec) -> np.ndarray:
    return (x @ spec.H >= 0).astype(np.float32)


def sample_hyperplane_arrangements(z: np.ndarray, num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given z (n x p), sample hyperplanes U (p x num_samples), compute patterns D = 1[zU>=0],
    keep unique columns and return (D_unique, U_unique).
    """
    rng = np.random.default_rng(seed)
    n, p = z.shape
    U = rng.normal(size=(p, num_samples)).astype(np.float32)
    patterns = (z @ U >= 0).astype(np.uint8)  # n x num_samples

    uniq: Dict[bytes, np.ndarray] = {}
    for j in range(patterns.shape[1]):
        key = patterns[:, j].tobytes()
        if key not in uniq:
            uniq[key] = U[:, j].copy()

    # rebuild D from keys (fast and stable)
    D_cols = [np.frombuffer(k, dtype=np.uint8) for k in uniq.keys()]
    D = np.stack(D_cols, axis=1).astype(np.float32)  # n x Puniq
    U_unique = np.stack(list(uniq.values()), axis=1).astype(np.float32)  # p x Puniq
    return D, U_unique


@dataclass
class LArrangementSpec:
    U_layers: List[np.ndarray]  # U_ell: (p_{ell-1} x P_ell)


def build_l_layer_arrangements(
    x_features: np.ndarray,
    P_per_layer: List[int],
    arr_samples: int,
    seed: int,
) -> Tuple[np.ndarray, LArrangementSpec]:
    """
    Build L-layer arrangement dictionary on TRAIN data:
        D0 := x_features
        D_ell := 1[ D_{ell-1} @ U_ell >= 0]
    """
    rng = np.random.default_rng(seed)
    D_prev = x_features.astype(np.float32)
    U_layers: List[np.ndarray] = []

    for ell, P_ell in enumerate(P_per_layer, start=1):
        oversamp = max(arr_samples, int(2.0 * P_ell))
        D_ell, U_ell = sample_hyperplane_arrangements(
            D_prev, oversamp, int(rng.integers(1, 1_000_000_000))
        )

        # cap to P_ell columns (dictionary budget)
        if D_ell.shape[1] > P_ell:
            D_ell = D_ell[:, :P_ell]
            U_ell = U_ell[:, :P_ell]

        U_layers.append(U_ell)
        D_prev = D_ell

    return D_prev, LArrangementSpec(U_layers=U_layers)


def transform_l_layer_arrangements(x_features: np.ndarray, spec: LArrangementSpec) -> np.ndarray:
    """Apply stored U_layers to compute D_L on TEST data."""
    D = x_features.astype(np.float32)
    for U in spec.U_layers:
        D = (D @ U >= 0).astype(np.float32)
    return D


# ----------------------------- first-order convex solvers -----------------------------
def solve_lasso_squared_fista(
    D: np.ndarray,
    y: np.ndarray,
    beta: float,
    max_iter: int = 20000,
    tol: float = 1e-6,
    step: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Solve: min_w 0.5||Dw - y||^2 + beta||w||_1 using FISTA.
    """
    n, p = D.shape
    w = np.zeros(p, dtype=np.float32)
    z = w.copy()
    t = 1.0

    if step is None:
        L = power_iteration_spectral_norm_sq(D, seed=seed)
        step = 1.0 / (L + 1e-12)

    prev_obj = np.inf
    for k in range(max_iter):
        Dz = D @ z
        grad = D.T @ (Dz - y)
        w_next = soft_threshold(z - step * grad, step * beta).astype(np.float32)

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_next + ((t - 1.0) / t_next) * (w_next - w)

        w = w_next
        t = t_next

        r = D @ w - y
        obj = 0.5 * float(r @ r) + beta * float(np.abs(w).sum())
        if abs(prev_obj - obj) <= tol * max(1.0, abs(prev_obj)):
            prev_obj = obj
            break
        prev_obj = obj

    return {"w": w, "primal": prev_obj, "iters": k + 1, "step": float(step)}


# ----------------------------- evaluation -----------------------------
def compute_accuracy_from_arrangements(D: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    preds = np.sign(D @ w)
    return float(np.mean(preds == y))


def plot_accuracies(records: List[Dict[str, float]], title: str, out_path: str) -> None:
    labels = [r["label"] for r in records]
    ste_train = [r["ste_train"] for r in records]
    ste_test = [r["ste_test"] for r in records]
    cvx_train = [r["cvx_train"] for r in records]
    cvx_test = [r["cvx_test"] for r in records]
    x = np.arange(len(labels))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, ste_train, width, label="STE train")
    plt.bar(x - 0.5 * width, ste_test, width, label="STE test")
    plt.bar(x + 0.5 * width, cvx_train, width, label="Convex (1st-order) train")
    plt.bar(x + 1.5 * width, cvx_test, width, label="Convex (1st-order) test")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def append_summary(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# ----------------------------- STE baseline -----------------------------
class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEFFN(nn.Module):
    def __init__(self, input_dim: int, m1: int, m2: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, m1, bias=False)
        self.w2 = nn.Linear(m1, m2, bias=False)
        self.w3 = nn.Linear(m2, 1, bias=False)

    def forward(self, x):
        h1 = ThresholdSTE.apply(self.w1(x))
        h2 = ThresholdSTE.apply(self.w2(h1))
        out = self.w3(h2)
        return out.squeeze(-1)


def train_ste_ffn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    m1: int,
    m2: int,
    beta: float,
    lr: float,
    epochs: int,
) -> Dict[str, float]:
    model = STEFFN(x_train.shape[1], m1, m2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = F.mse_loss(logits, y_train_t) + beta * model.w3.weight.abs().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_pred = model(x_train_t).sign()
        test_pred = model(x_test_t).sign()
    return {
        "train_acc": (train_pred == y_train_t).float().mean().item(),
        "test_acc": (test_pred == y_test_t).float().mean().item(),
    }


# ----------------------------- config -----------------------------
@dataclass
class B4Params:
    n: int
    d: int
    m1: int
    m2: int
    m_star: int
    m_repr: int
    beta: float


def b4_param_grid() -> List[B4Params]:
    return [
        B4Params(n=20, d=120, m1=1020, m2=40, m_star=40, m_repr=1020, beta=1e-3),
        B4Params(n=50, d=70, m1=1020, m2=70, m_star=40, m_repr=1020, beta=1e-3),
        B4Params(n=100, d=40, m1=1020, m2=120, m_star=40, m_repr=1020, beta=1e-3),
    ]


def generate_ground_truth_labels(x: np.ndarray, m_star: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(x.shape[1], m_star)).astype(np.float32)
    w2 = rng.normal(size=(m_star, m_star)).astype(np.float32)
    w3 = rng.normal(size=(m_star,)).astype(np.float32)
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    logits = h2 @ w3
    return to_sign_labels(logits)


# ----------------------------- experiments -----------------------------
def parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def run_ffn_experiments(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    params: B4Params,
    config: Dict[str, float],
    P_per_layer: List[int],
) -> None:
    records = []
    summary_path = config["summary_path"]
    L = len(P_per_layer)

    for name, (x, y) in datasets.items():
        x_train = x[: params.n]
        y_train = y[: params.n]
        x_test = x[params.n : params.n + config["n_test"]]
        y_test = y[params.n : params.n + config["n_test"]]

        x_train_tilde, repr_spec = build_representation_matrix(x_train, params.m_repr, config["seed"])
        x_test_tilde = transform_representation_matrix(x_test, repr_spec)

        ste_metrics = train_ste_ffn(
            x_train_tilde, y_train, x_test_tilde, y_test,
            params.m1, params.m2, params.beta, config["lr"], config["epochs"],
        )

        D_train, arr_spec = build_l_layer_arrangements(
            x_train_tilde, P_per_layer=P_per_layer, arr_samples=config["arr_samples"], seed=config["seed"] + 123
        )
        D_test = transform_l_layer_arrangements(x_test_tilde, arr_spec)

        cvx = solve_lasso_squared_fista(
            D_train, y_train, params.beta,
            max_iter=config["fo_max_iter"], tol=config["fo_tol"], seed=config["seed"] + 999
        )

        train_acc = compute_accuracy_from_arrangements(D_train, cvx["w"], y_train)
        test_acc = compute_accuracy_from_arrangements(D_test, cvx["w"], y_test)

        print(f"[FFN:{name}] L={L} P={P_per_layer} | STE train={ste_metrics['train_acc']:.3f} test={ste_metrics['test_acc']:.3f}")
        print(f"[FFN:{name}] ConvexFO train={train_acc:.3f} test={test_acc:.3f} | iters={cvx['iters']} step={cvx['step']:.3e}")

        append_summary(
            summary_path,
            f"FFN:{name} | L={L} P={P_per_layer} | STE train={ste_metrics['train_acc']:.3f} test={ste_metrics['test_acc']:.3f} | "
            f"ConvexFO train={train_acc:.3f} test={test_acc:.3f} | obj={cvx['primal']:.6f} iters={cvx['iters']}",
        )

        records.append(
            {"label": name, "ste_train": ste_metrics["train_acc"], "ste_test": ste_metrics["test_acc"],
             "cvx_train": train_acc, "cvx_test": test_acc}
        )

    plot_accuracies(records, f"FFN Accuracies (L={L})", config["ffn_plot_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ffn"], required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--L", type=int, default=2, help="Number of arrangement layers (excluding representation).")
    parser.add_argument("--P_per_layer", type=str, default="2000,2000",
                        help="Comma-separated widths per arrangement layer, length must equal L.")
    parser.add_argument("--arr_samples", type=int, default=2000)
    parser.add_argument("--fo_max_iter", type=int, default=20000)
    parser.add_argument("--fo_tol", type=float, default=1e-6)
    args = parser.parse_args()

    set_seed(args.seed)

    P_per_layer = parse_int_list(args.P_per_layer)
    if len(P_per_layer) != args.L:
        raise ValueError(f"--P_per_layer must have length L={args.L}, got {P_per_layer}")

    run_ts = int(time.time())
    run_id = f"{args.mode}_L{args.L}_{run_ts}"

    config = {
        "seed": args.seed,
        "arr_samples": 200 if args.debug else args.arr_samples,
        "epochs": 50 if args.debug else 500,
        "lr": 1e-2,
        "n_test": 200 if args.debug else 3000,
        "fo_max_iter": 2000 if args.debug else args.fo_max_iter,
        "fo_tol": args.fo_tol,
        "ffn_plot_path": str(Path(__file__).with_name(f"ffn_accuracies_{run_id}.png")),
        "summary_path": str(Path(__file__).with_name("summary_results.txt")),
    }

    summary_path = Path(config["summary_path"])
    if summary_path.exists():
        existing = summary_path.read_text()
        if existing and not existing.endswith("\n\n"):
            summary_path.write_text(existing + "\n")
    append_summary(config["summary_path"], f"RUN {Path(__file__).name} | mode={args.mode} | L={args.L} | time={run_ts}")

    params_grid = b4_param_grid()

    for params in params_grid:
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        x_parity, y_parity = generate_parity(params.n + config["n_test"], params.d, args.seed + 1)
        datasets["parity"] = (x_parity, y_parity)
        x_gmm, y_gmm = generate_gaussian_mixture(params.n + config["n_test"], params.d, args.seed + 2)
        datasets["gaussian"] = (x_gmm, y_gmm)

        x_gt = np.random.normal(size=(params.n + config["n_test"], params.d)).astype(np.float32)
        y_gt = generate_ground_truth_labels(x_gt, params.m_star, args.seed + 5)
        datasets["ground_truth"] = (x_gt, y_gt)

        run_ffn_experiments(datasets, params, config, P_per_layer=P_per_layer)


if __name__ == "__main__":
    main()
