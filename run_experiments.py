import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_sign_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    y_signed = np.where(y > 0, 1.0, -1.0)
    return y_signed.astype(np.float32)


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


def generate_checkerboard(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if d != 2:
        raise ValueError("checkerboard is defined for d=2.")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, d)).astype(np.float32)
    grid = np.floor(x + 2.0).astype(int)
    parity = (grid[:, 0] + grid[:, 1]) % 2
    y = parity * 2 - 1
    return x, y.astype(np.float32)


def generate_two_spirals(n: int, seed: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = np.sqrt(rng.uniform(0, 1, n_half)) * 2 * np.pi
    r = 2 * theta + np.pi
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
    x = np.concatenate([x1, x2], axis=0).astype(np.float32)
    x += rng.normal(scale=noise, size=x.shape).astype(np.float32)
    y = np.concatenate([np.ones(n_half), -np.ones(n_half)], axis=0).astype(np.float32)
    return x, y


def generate_two_step_xor(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
    x2 = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
    predicate1 = (x1 @ np.array([1.0, -1.0]) >= 0).astype(int)
    predicate2 = (x2 @ np.array([1.0, -1.0]) >= 0).astype(int)
    y = (predicate1 ^ predicate2) * 2 - 1
    return x1, x2, y.astype(np.float32)


def generate_moving_gaussian_blobs(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers_t1 = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    shift = rng.normal(scale=0.2, size=centers_t1.shape).astype(np.float32)
    centers_t2 = centers_t1 + shift
    labels = rng.integers(0, 2, size=n)
    x1 = rng.normal(size=(n, 2)).astype(np.float32) + centers_t1[labels]
    x2 = rng.normal(size=(n, 2)).astype(np.float32) + centers_t2[labels]
    y = labels * 2 - 1
    return x1, x2, y.astype(np.float32)


def generate_rotated_mnist_pairs(n: int, seed: int, angle_deg: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from torchvision import datasets, transforms

    rng = np.random.default_rng(seed)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    indices = rng.choice(len(mnist), size=n, replace=False)
    images = []
    labels = []
    for idx in indices:
        img, label = mnist[idx]
        images.append(img.squeeze(0).numpy())
        labels.append(label)
    x1 = np.stack(images, axis=0).astype(np.float32)
    rot = transforms.RandomRotation((angle_deg, angle_deg))
    x2 = np.stack([rot(torch.tensor(img).unsqueeze(0)).squeeze(0).numpy() for img in x1], axis=0)
    y = np.array(labels, dtype=np.int64)
    y = (y > (y.mean())).astype(np.float32) * 2 - 1
    x1 = x1.reshape(n, -1)
    x2 = x2.reshape(n, -1)
    return x1, x2, y.astype(np.float32)


def build_representation_matrix(x: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(x.shape[1], m)).astype(np.float32)
    x_tilde = (x @ h >= 0).astype(np.float32)
    return x_tilde


def sample_hyperplane_arrangements(z: np.ndarray, num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = z.shape
    u = rng.normal(size=(p, num_samples)).astype(np.float32)
    patterns = (z @ u >= 0).astype(np.int32)
    uniq = {}
    for idx in range(patterns.shape[1]):
        key = tuple(patterns[:, idx].tolist())
        if key not in uniq:
            uniq[key] = u[:, idx]
    d = np.stack(list(uniq.keys()), axis=1).astype(np.float32)
    u_unique = np.stack(list(uniq.values()), axis=1).astype(np.float32)
    return d, u_unique


@dataclass
class FFNArrangementSpec:
    u1: np.ndarray
    subset_indices: List[np.ndarray]
    u2: List[np.ndarray]


def build_ffn_arrangements(
    x: np.ndarray,
    m1: int,
    num_arr_samples: int,
    num_subset_samples: int,
    seed: int,
) -> Tuple[np.ndarray, FFNArrangementSpec]:
    d1, u1 = sample_hyperplane_arrangements(x, num_arr_samples, seed)
    rng = np.random.default_rng(seed + 1)
    if d1.shape[1] < m1:
        raise ValueError("Not enough arrangements in D1 to sample subsets.")
    subset_indices = []
    u2 = []
    d2_columns = []
    for _ in range(num_subset_samples):
        subset = rng.choice(d1.shape[1], size=m1, replace=False)
        d1_subset = d1[:, subset]
        d2_subset, u2_subset = sample_hyperplane_arrangements(d1_subset, num_arr_samples, rng.integers(1, 1_000_000))
        for col_idx in range(d2_subset.shape[1]):
            d2_columns.append(d2_subset[:, col_idx])
            subset_indices.append(subset)
            u2.append(u2_subset[:, col_idx])
    d2 = np.stack(d2_columns, axis=1).astype(np.float32)
    spec = FFNArrangementSpec(u1=u1, subset_indices=subset_indices, u2=u2)
    return d2, spec


def transform_ffn_arrangements(x: np.ndarray, spec: FFNArrangementSpec) -> np.ndarray:
    d1 = (x @ spec.u1 >= 0).astype(np.float32)
    d2_columns = []
    for subset, u2 in zip(spec.subset_indices, spec.u2):
        d2_columns.append((d1[:, subset] @ u2 >= 0).astype(np.float32))
    d2 = np.stack(d2_columns, axis=1).astype(np.float32)
    return d2


@dataclass
class RNNArrangementSpec:
    u11: np.ndarray
    u21: np.ndarray
    u12: np.ndarray
    u22: np.ndarray


def build_rnn_arrangements(
    x1: np.ndarray,
    x2: np.ndarray,
    num_arr_samples: int,
    seed: int,
) -> Tuple[np.ndarray, RNNArrangementSpec]:
    d10 = np.zeros((x1.shape[0], 1), dtype=np.float32)
    d20 = np.zeros((x1.shape[0], 1), dtype=np.float32)
    d11_input = np.concatenate([x1, d10], axis=1)
    d11, u11 = sample_hyperplane_arrangements(d11_input, num_arr_samples, seed + 1)
    d21_input = np.concatenate([d11, d20], axis=1)
    d21, u21 = sample_hyperplane_arrangements(d21_input, num_arr_samples, seed + 2)
    d12_input = np.concatenate([x2, d11], axis=1)
    d12, u12 = sample_hyperplane_arrangements(d12_input, num_arr_samples, seed + 3)
    d22_input = np.concatenate([d12, d21], axis=1)
    d22, u22 = sample_hyperplane_arrangements(d22_input, num_arr_samples, seed + 4)
    spec = RNNArrangementSpec(u11=u11, u21=u21, u12=u12, u22=u22)
    return d22.astype(np.float32), spec


def transform_rnn_arrangements(x1: np.ndarray, x2: np.ndarray, spec: RNNArrangementSpec) -> np.ndarray:
    d10 = np.zeros((x1.shape[0], 1), dtype=np.float32)
    d20 = np.zeros((x1.shape[0], 1), dtype=np.float32)
    d11_input = np.concatenate([x1, d10], axis=1)
    d11 = (d11_input @ spec.u11 >= 0).astype(np.float32)
    d21_input = np.concatenate([d11, d20], axis=1)
    d21 = (d21_input @ spec.u21 >= 0).astype(np.float32)
    d12_input = np.concatenate([x2, d11], axis=1)
    d12 = (d12_input @ spec.u12 >= 0).astype(np.float32)
    d22_input = np.concatenate([d12, d21], axis=1)
    d22 = (d22_input @ spec.u22 >= 0).astype(np.float32)
    return d22.astype(np.float32)


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


class STERNN(nn.Module):
    def __init__(self, input_dim: int, m1: int, m2: int):
        super().__init__()
        self.p1 = nn.Linear(input_dim, m1, bias=False)
        self.p2_in = nn.Linear(m1, m2, bias=False)
        self.p2_rec = nn.Linear(m2, m2, bias=False)
        self.p3 = nn.Linear(m2, 1, bias=False)

    def forward(self, x1, x2):
        h1_1 = ThresholdSTE.apply(self.p1(x1))
        h2_0 = torch.zeros(h1_1.shape[0], self.p2_rec.in_features, device=h1_1.device, dtype=h1_1.dtype)
        h2_1 = ThresholdSTE.apply(self.p2_in(h1_1) + self.p2_rec(h2_0))
        h1_2 = ThresholdSTE.apply(self.p1(x2))
        h2_2 = ThresholdSTE.apply(self.p2_in(h1_2) + self.p2_rec(h2_1))
        out = self.p3(h2_2)
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
    train_acc = (train_pred == y_train_t).float().mean().item()
    test_acc = (test_pred == y_test_t).float().mean().item()
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "w1": model.w1.weight.detach().cpu().numpy(),
        "w2": model.w2.weight.detach().cpu().numpy(),
        "w3": model.w3.weight.detach().cpu().numpy(),
    }


def train_ste_rnn(
    x1_train: np.ndarray,
    x2_train: np.ndarray,
    y_train: np.ndarray,
    x1_test: np.ndarray,
    x2_test: np.ndarray,
    y_test: np.ndarray,
    m1: int,
    m2: int,
    beta: float,
    lr: float,
    epochs: int,
) -> Dict[str, float]:
    model = STERNN(x1_train.shape[1], m1, m2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x1_train_t = torch.from_numpy(x1_train)
    x2_train_t = torch.from_numpy(x2_train)
    y_train_t = torch.from_numpy(y_train)
    x1_test_t = torch.from_numpy(x1_test)
    x2_test_t = torch.from_numpy(x2_test)
    y_test_t = torch.from_numpy(y_test)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x1_train_t, x2_train_t)
        hinge = torch.relu(1 - y_train_t * logits).mean()
        loss = hinge + beta * model.p3.weight.abs().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_pred = model(x1_train_t, x2_train_t).sign()
        test_pred = model(x1_test_t, x2_test_t).sign()
    train_acc = (train_pred == y_train_t).float().mean().item()
    test_acc = (test_pred == y_test_t).float().mean().item()
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "p1": model.p1.weight.detach().cpu().numpy(),
        "p2_in": model.p2_in.weight.detach().cpu().numpy(),
        "p2_rec": model.p2_rec.weight.detach().cpu().numpy(),
        "p3": model.p3.weight.detach().cpu().numpy(),
    }


def solve_convex_lasso_squared(d_mat: np.ndarray, y: np.ndarray, beta: float, solver: str) -> Dict[str, float]:
    # Primal (squared loss): min_w 0.5||Dw - y||_2^2 + beta||w||_1
    w = cp.Variable(d_mat.shape[1])
    objective = 0.5 * cp.sum_squares(d_mat @ w - y) + beta * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver)
    w_val = w.value
    primal = problem.value
    # Dual (Draft_1 / arXiv formulation):
    # max_lambda -0.5||lambda||_2^2 - y^T lambda  s.t. ||D^T lambda||_inf <= beta
    lam = cp.Variable(d_mat.shape[0])
    dual_constraints = [cp.norm_inf(d_mat.T @ lam) <= beta]
    dual_objective = cp.Maximize(-0.5 * cp.sum_squares(lam) - lam @ y)
    dual_problem = cp.Problem(dual_objective, dual_constraints)
    dual_problem.solve(solver=solver)
    dual = dual_problem.value
    gap = primal - dual
    return {"w": w_val, "primal": primal, "dual": dual, "gap": gap}


def solve_convex_lasso_hinge(
    d_mat: np.ndarray,
    y: np.ndarray,
    beta: float,
    m2: int,
    solver: str,
) -> Dict[str, float]:
    # Primal (hinge loss): min_w (1/n) sum_r max(0, 1 - y_r (Dw)_r) + beta_hat ||w||_1
    # with beta_hat = beta / sqrt(m2) per Draft_1 (5.14) scaling.
    beta_hat = beta / np.sqrt(m2)
    w = cp.Variable(d_mat.shape[1])
    loss = cp.sum(cp.pos(1 - cp.multiply(y, d_mat @ w))) / y.shape[0]
    objective = loss + beta_hat * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver)
    w_val = w.value
    primal = problem.value
    # Dual (hinge loss with L1 regularizer):
    # max_lambda -sum_r y_r lambda_r
    # s.t. y_r * lambda_r in [-1/n, 0] and ||D^T lambda||_inf <= beta_hat
    lam = cp.Variable(d_mat.shape[0])
    constraints = [
        cp.multiply(y, lam) <= 0,
        cp.multiply(y, lam) >= -1.0 / y.shape[0],
        cp.norm_inf(d_mat.T @ lam) <= beta_hat,
    ]
    dual_objective = cp.Maximize(-cp.sum(cp.multiply(y, lam)))
    dual_problem = cp.Problem(dual_objective, constraints)
    dual_problem.solve(solver=solver)
    dual = dual_problem.value
    gap = primal - dual
    return {"w": w_val, "primal": primal, "dual": dual, "gap": gap}


def compute_accuracy_from_arrangements(d_mat: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    preds = np.sign(d_mat @ w)
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
    plt.bar(x + 0.5 * width, cvx_train, width, label="Convex train")
    plt.bar(x + 1.5 * width, cvx_test, width, label="Convex test")
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
        B4Params(n=20, d=100, m1=1000, m2=20, m_star=20, m_repr=1000, beta=1e-3),
        B4Params(n=50, d=50, m1=1000, m2=50, m_star=20, m_repr=1000, beta=1e-3),
        B4Params(n=100, d=20, m1=1000, m2=100, m_star=20, m_repr=1000, beta=1e-3),
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


def run_ffn_experiments(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    params: B4Params,
    config: Dict[str, float],
) -> None:
    records = []
    zero_gap_tol = 1e-4
    summary_path = config["summary_path"]
    for name, (x, y) in datasets.items():
        x_train = x[: params.n]
        y_train = y[: params.n]
        x_test = x[params.n : params.n + config["n_test"]]
        y_test = y[params.n : params.n + config["n_test"]]
        x_train_tilde = build_representation_matrix(x_train, params.m_repr, config["seed"])
        x_test_tilde = build_representation_matrix(x_test, params.m_repr, config["seed"])
        ste_metrics = train_ste_ffn(
            x_train_tilde,
            y_train,
            x_test_tilde,
            y_test,
            params.m1,
            params.m2,
            params.beta,
            config["lr"],
            config["epochs"],
        )
        print(f"[FFN:{name}] STE weights w1={ste_metrics['w1']}")
        print(f"[FFN:{name}] STE weights w2={ste_metrics['w2']}")
        print(f"[FFN:{name}] STE weights w3={ste_metrics['w3']}")
        arr_samples = max(config["arr_samples"], params.m1 * 4)
        d2_train, spec = build_ffn_arrangements(
            x_train_tilde,
            params.m1,
            arr_samples,
            config["subset_samples"],
            config["seed"],
        )
        d2_test = transform_ffn_arrangements(x_test_tilde, spec)
        cvx = solve_convex_lasso_squared(d2_train, y_train, params.beta, config["solver"])
        print(f"[FFN:{name}] Convex-LASSO weights w={cvx['w']}")
        train_acc = compute_accuracy_from_arrangements(d2_train, cvx["w"], y_train)
        test_acc = compute_accuracy_from_arrangements(d2_test, cvx["w"], y_test)
        print(f"[FFN:{name}] STE train acc={ste_metrics['train_acc']:.3f} test acc={ste_metrics['test_acc']:.3f}")
        print(f"[FFN:{name}] Convex train acc={train_acc:.3f} test acc={test_acc:.3f}")
        is_zero_gap = abs(cvx["gap"]) <= zero_gap_tol
        print(f"[FFN:{name}] Primal={cvx['primal']:.6f} Dual={cvx['dual']:.6f}")
        print(f"[FFN:{name}] Duality gap={cvx['gap']:.6f} zero_gap={is_zero_gap}")
        append_summary(
            summary_path,
            f"FFN:{name} | STE train={ste_metrics['train_acc']:.3f} test={ste_metrics['test_acc']:.3f} | "
            f"Convex train={train_acc:.3f} test={test_acc:.3f} | "
            f"Primal={cvx['primal']:.6f} Dual={cvx['dual']:.6f} Gap={cvx['gap']:.6f} zero_gap={is_zero_gap}",
        )
        records.append(
            {
                "label": name,
                "ste_train": ste_metrics["train_acc"],
                "ste_test": ste_metrics["test_acc"],
                "cvx_train": train_acc,
                "cvx_test": test_acc,
            }
        )
    plot_accuracies(records, "FFN Accuracies", config["ffn_plot_path"])


def run_rnn_experiments(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    params: B4Params,
    config: Dict[str, float],
) -> None:
    records = []
    zero_gap_tol = 1e-4
    summary_path = config["summary_path"]
    for name, data in datasets.items():
        x1_train, x2_train, y_train, x1_test, x2_test, y_test = data
        ste_metrics = train_ste_rnn(
            x1_train,
            x2_train,
            y_train,
            x1_test,
            x2_test,
            y_test,
            params.m1,
            params.m2,
            params.beta,
            config["lr"],
            config["epochs"],
        )
        print(f"[RNN:{name}] STE weights p1={ste_metrics['p1']}")
        print(f"[RNN:{name}] STE weights p2_in={ste_metrics['p2_in']}")
        print(f"[RNN:{name}] STE weights p2_rec={ste_metrics['p2_rec']}")
        print(f"[RNN:{name}] STE weights p3={ste_metrics['p3']}")
        d22_train, spec = build_rnn_arrangements(
            x1_train,
            x2_train,
            config["arr_samples"],
            config["seed"],
        )
        d22_test = transform_rnn_arrangements(x1_test, x2_test, spec)
        cvx = solve_convex_lasso_hinge(d22_train, y_train, params.beta, params.m2, config["solver"])
        print(f"[RNN:{name}] Convex-LASSO weights w={cvx['w']}")
        train_acc = compute_accuracy_from_arrangements(d22_train, cvx["w"], y_train)
        test_acc = compute_accuracy_from_arrangements(d22_test, cvx["w"], y_test)
        print(f"[RNN:{name}] STE train acc={ste_metrics['train_acc']:.3f} test acc={ste_metrics['test_acc']:.3f}")
        print(f"[RNN:{name}] Convex train acc={train_acc:.3f} test acc={test_acc:.3f}")
        is_zero_gap = abs(cvx["gap"]) <= zero_gap_tol
        print(f"[RNN:{name}] Primal={cvx['primal']:.6f} Dual={cvx['dual']:.6f}")
        print(f"[RNN:{name}] Duality gap={cvx['gap']:.6f} zero_gap={is_zero_gap}")
        append_summary(
            summary_path,
            f"RNN:{name} | STE train={ste_metrics['train_acc']:.3f} test={ste_metrics['test_acc']:.3f} | "
            f"Convex train={train_acc:.3f} test={test_acc:.3f} | "
            f"Primal={cvx['primal']:.6f} Dual={cvx['dual']:.6f} Gap={cvx['gap']:.6f} zero_gap={is_zero_gap}",
        )
        records.append(
            {
                "label": name,
                "ste_train": ste_metrics["train_acc"],
                "ste_test": ste_metrics["test_acc"],
                "cvx_train": train_acc,
                "cvx_test": test_acc,
            }
        )
    plot_accuracies(records, "RNN Accuracies", config["rnn_plot_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ffn", "rnn", "all"], required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    np.set_printoptions(threshold=np.inf, linewidth=200)
    params_grid = b4_param_grid()
    config = {
        "seed": args.seed,
        "arr_samples": 200 if args.debug else 2000,
        "subset_samples": 5 if args.debug else 50,
        "epochs": 50 if args.debug else 500,
        "lr": 1e-2,
        "solver": args.solver,
        "n_test": 200 if args.debug else 3000,
        "ffn_plot_path": "/Users/hima_3114/Desktop/Paper_1/experiments/ffn_accuracies.png",
        "rnn_plot_path": "/Users/hima_3114/Desktop/Paper_1/experiments/rnn_accuracies.png",
        "summary_path": "/Users/hima_3114/Desktop/Paper_1/experiments/summary_results.txt",
    }
    Path(config["summary_path"]).write_text("")

    for params in params_grid:
        if args.mode in ["ffn", "all"]:
            datasets = {}
            x_parity, y_parity = generate_parity(params.n + config["n_test"], params.d, args.seed + 1)
            datasets["parity"] = (x_parity, y_parity)
            x_gmm, y_gmm = generate_gaussian_mixture(params.n + config["n_test"], params.d, args.seed + 2)
            datasets["gaussian"] = (x_gmm, y_gmm)
            if params.d == 2:
                x_chk, y_chk = generate_checkerboard(params.n + config["n_test"], params.d, args.seed + 3)
                datasets["checkerboard"] = (x_chk, y_chk)
                x_sp, y_sp = generate_two_spirals(params.n + config["n_test"], args.seed + 4)
                datasets["two_spirals"] = (x_sp, y_sp)
            x_gt = np.random.normal(size=(params.n + config["n_test"], params.d)).astype(np.float32)
            y_gt = generate_ground_truth_labels(x_gt, params.m_star, args.seed + 5)
            datasets["ground_truth"] = (x_gt, y_gt)
            run_ffn_experiments(datasets, params, config)
        if args.mode in ["rnn", "all"]:
            datasets = {}
            x1, x2, y = generate_two_step_xor(params.n + config["n_test"], args.seed + 6)
            datasets["two_step_xor"] = (
                x1[: params.n],
                x2[: params.n],
                y[: params.n],
                x1[params.n : params.n + config["n_test"]],
                x2[params.n : params.n + config["n_test"]],
                y[params.n : params.n + config["n_test"]],
            )
            x1, x2, y = generate_moving_gaussian_blobs(params.n + config["n_test"], args.seed + 7)
            datasets["moving_gaussian"] = (
                x1[: params.n],
                x2[: params.n],
                y[: params.n],
                x1[params.n : params.n + config["n_test"]],
                x2[params.n : params.n + config["n_test"]],
                y[params.n : params.n + config["n_test"]],
            )
            if not args.debug:
                x1, x2, y = generate_rotated_mnist_pairs(params.n + config["n_test"], args.seed + 8)
                datasets["rotated_mnist"] = (
                    x1[: params.n],
                    x2[: params.n],
                    y[: params.n],
                    x1[params.n : params.n + config["n_test"]],
                    x2[params.n : params.n + config["n_test"]],
                    y[params.n : params.n + config["n_test"]],
                )
            run_rnn_experiments(datasets, params, config)


if __name__ == "__main__":
    main()
