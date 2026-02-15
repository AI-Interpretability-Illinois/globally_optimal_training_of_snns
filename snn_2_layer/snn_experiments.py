import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_sign_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    y_signed = np.where(y > 0, 1.0, -1.0)
    return y_signed.astype(np.float32)


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


def create_representation_weights(d: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(d, m)).astype(np.float32)
    return h


def apply_representation(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    x_tilde = (x @ h >= 0).astype(np.float32)
    return x_tilde


def sample_hyperplane_arrangements(
    z: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, p = z.shape
    u = rng.normal(size=(p, num_samples)).astype(np.float32)
    patterns = (z @ u >= 0).astype(np.int32)
    uniq = {}
    for idx in range(patterns.shape[1]):
        key = tuple(patterns[:, idx].tolist())
        if key not in uniq:
            uniq[key] = patterns[:, idx].astype(np.float32)
    if not uniq:
        raise ValueError("A_SNN sampling produced no spike patterns from input.")
    return np.stack(list(uniq.values()), axis=1)


def a_snn_operator(
    z: np.ndarray,
    sampled: bool,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """
    A_SNN(·): enumerate spike patterns 1{Z u >= 0} over u ∈ R^p.

    This follows the hyperplane arrangement logic (Section 3.1):
    - p = 0: only the all-ones pattern is possible.
    - p = 1: sign is determined by the single column.
    - p >= 2: either sample random rays (approximate), or enumerate candidate
      rays from intersections of (p - 1) rows (full combinatorial).
    """
    n, p = z.shape

    if sampled:
        return sample_hyperplane_arrangements(z, num_samples, seed)

    # Degenerate: no features
    if p == 0:
        return np.ones((n, 1), dtype=np.float32)

    # Single feature: only two possible sign patterns (u and -u)
    if p == 1:
        # Use direction u = [1]; sign(Z u) = sign(column 0)
        u = np.array([[1.0]], dtype=np.float32)  # shape (1,1)
        s = (z @ u >= 0).astype(np.float32).reshape(-1)
        s_neg = (z @ (-u) >= 0).astype(np.float32).reshape(-1)
        uniq = {tuple(s.tolist()): s, tuple(s_neg.tolist()): s_neg}
        return np.stack(list(uniq.values()), axis=1)

    # General case: p >= 2
    # Full combinatorial method: enumerate all subsets of size (p - 1).
    if n < p - 1:
        raise ValueError("A_SNN requires n >= p-1 for full combinatorial enumeration.")
    subset_size = p - 1
    uniq = {}

    for subset in itertools.combinations(range(n), subset_size):
        zs = z[list(subset), :]          # shape (subset_size, p)
        # SVD to find a direction u in the nullspace of these rows.
        _, _, vt = np.linalg.svd(zs)
        u = vt[-1]                       # shape (p,)
        if np.allclose(u, 0):
            continue

        s = (z @ u >= 0).astype(np.float32).reshape(-1)
        s_neg = (z @ (-u) >= 0).astype(np.float32).reshape(-1)

        key = tuple(s.tolist())
        if key not in uniq:
            uniq[key] = s
        key_neg = tuple(s_neg.tolist())
        if key_neg not in uniq:
            uniq[key_neg] = s_neg

    if not uniq:
        raise ValueError("A_SNN produced no spike patterns from input.")

    return np.stack(list(uniq.values()), axis=1)


def a_snn_from_blocks(
    block1: np.ndarray,
    block2: np.ndarray,
    op: str,
    bias: np.ndarray,
    sampled: bool,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Combine two blocks and a bias into a single design matrix Z, then apply A_SNN.

    block1: first block (e.g. X^t or D^{i-1,t})
    block2: second block (e.g. D^{i,t-1} or a previous D)
    op    : '+' or '-' controlling the sign of block2
    bias  : n×1 column of ones (for the affine term)
    """
    if op not in {"+", "-"}:
        raise ValueError(f"Unsupported op '{op}', expected '+' or '-'.")

    sign = 1.0 if op == "+" else -1.0
    z = np.concatenate([block1, sign * block2, bias], axis=1)
    return a_snn_operator(z, sampled, num_samples, seed)


def build_snn_arrangements(
    x1: np.ndarray,
    x2: np.ndarray,
    seed: int,
    num_samples: int,
    sampled: bool,
) -> Dict[str, np.ndarray]:
    """
    HyperplaneArrangementSNN for a 3-layer, T=2 SNN toy model.

    n: number of samples
    x1: X^1 ∈ R^{n×d}
    x2: X^2 ∈ R^{n×d}

    Returns:
      D11 ≈ D^{(1,1)}, D12 ≈ D^{(1,2)},
      D21 ≈ D^{(2,1)}, D22 ≈ D^{(2,2)}.
    """
    n = x1.shape[0]

    # U^1[0] and U^2[0] are taken as 0 here (no pre-input membrane),
    # S[0] = 0, consistent with the theoretical setup.
    x0 = np.zeros((n, 1), dtype=np.float32)  # acts as "U2[0]" placeholder
    h0 = np.zeros((n, 1), dtype=np.float32)  # acts as "U1[0]" placeholder

    ones = np.ones((n, 1), dtype=np.float32)

    # D(1,1) = A_SNN([X1, +U1[0], 1])  with U1[0] = 0
    d11 = a_snn_from_blocks(x1, h0, "+", ones, sampled, num_samples, seed + 1)

    # D(1,2) = A_SNN([X2, -S1[1], 1])  with S1[1] patterns coming from D(1,1)
    d12 = a_snn_from_blocks(x2, d11, "-", ones, sampled, num_samples, seed + 2)

    # D(2,1) = A_SNN([D(1,1), +U2[0], 1])  with U2[0] = 0
    d21 = a_snn_from_blocks(d11, x0, "+", ones, sampled, num_samples, seed + 3)

    # D(2,2) = A_SNN([D(1,2), -S2[1], 1])  with S2[1] patterns coming from D(2,1)
    d22 = a_snn_from_blocks(d12, d21, "-", ones, sampled, num_samples, seed + 4)

    return {
        "D11": d11.astype(np.float32),
        "D12": d12.astype(np.float32),
        "D21": d21.astype(np.float32),
        "D22": d22.astype(np.float32),
    }


def solve_primal_convex_snn(
    d22: np.ndarray,
    y: np.ndarray,
    beta: float,
    m2: int,
    solver: str,
    solver_opts: Dict[str, float],
) -> Dict[str, float]:
    # Algorithm 3: SolvePrimalConvexSNN
    beta_hat = beta / np.sqrt(m2)
    w = cp.Variable(d22.shape[1])
    objective = 0.5 * cp.sum_squares(d22 @ w - y) + beta_hat * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver, **solver_opts)
    w_val = w.value
    primal = problem.value
    preds = np.sign(d22 @ w_val)
    acc = float(np.mean(preds == y))
    return {"primal": primal, "w": w_val, "acc": acc}


def solve_dual_convex_snn(
    d22: np.ndarray,
    y: np.ndarray,
    beta: float,
    m2: int,
    solver: str,
    solver_opts: Dict[str, float],
) -> Dict[str, float]:
    # Algorithm 4: SolveDualConvexSNN
    beta_hat = beta / np.sqrt(m2)
    lam = cp.Variable(d22.shape[0])
    constraints = [
        d22.T @ lam <= beta_hat,
        d22.T @ lam >= -beta_hat,
    ]
    objective = cp.Maximize(-0.5 * cp.sum_squares(lam) - lam @ y)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **solver_opts)
    return {"dual": problem.value, "lambda": lam.value}


def train_leaky_snn(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    m1: int,
    m2: int,
    lr: float,
    epochs: int,
) -> Dict[str, float]:
    # Algorithm 5: TrainLeakySnnTorch (snnTorch Leaky neuron)
    fc1 = nn.Linear(x1.shape[1], m1, bias=False)
    lif1 = snn.Leaky(beta=0.9)
    fc2 = nn.Linear(m1, m2, bias=False)
    lif2 = snn.Leaky(beta=0.9)
    fc3 = nn.Linear(m2, 1, bias=False)
    params = list(fc1.parameters()) + list(fc2.parameters()) + list(fc3.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    x1_t = torch.from_numpy(x1)
    x2_t = torch.from_numpy(x2)
    y_t = torch.from_numpy(y)
    for _ in range(epochs):
        optimizer.zero_grad()
        mem1 = lif1.init_leaky()
        mem2 = lif2.init_leaky()
        cur1 = fc1(x1_t)
        spk1, mem1 = lif1(cur1, mem1)
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)

        cur1 = fc1(x2_t)
        spk1, mem1 = lif1(cur1, mem1)
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)
        logits = fc3(spk2).squeeze(-1)
        loss = torch.mean((logits - y_t) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mem1 = lif1.init_leaky()
        mem2 = lif2.init_leaky()
        spk1, mem1 = lif1(fc1(x1_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        spk1, mem1 = lif1(fc1(x2_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        preds = fc3(spk2).squeeze(-1).sign()
    acc = float((preds == y_t).float().mean().item())
    return {"acc": acc, "layers": (fc1, lif1, fc2, lif2, fc3)}


def eval_leaky_snn(layers: Tuple[nn.Linear, snn.Leaky, nn.Linear, snn.Leaky, nn.Linear], x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    fc1, lif1, fc2, lif2, fc3 = layers
    x1_t = torch.from_numpy(x1)
    x2_t = torch.from_numpy(x2)
    with torch.no_grad():
        mem1 = lif1.init_leaky()
        mem2 = lif2.init_leaky()
        spk1, mem1 = lif1(fc1(x1_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        spk1, mem1 = lif1(fc1(x2_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        preds = fc3(spk2).squeeze(-1).sign().numpy()
    return preds


def solve_svm_hyperplane(
    x: np.ndarray,
    y: np.ndarray,
    C: float,
    solver: str,
    solver_opts: Dict[str, float],
) -> np.ndarray:
    # Algorithm 1 reconstruction uses SVMs (5.12) and (5.13).
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    w = cp.Variable(x.shape[1])
    xi = cp.Variable(x.shape[0])
    objective = 0.5 * cp.sum_squares(w) + C * cp.sum(xi)
    constraints = [cp.multiply(y, x @ w) >= 1 - xi, xi >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, **solver_opts)
    return w.value


def select_activation_patterns(
    d11: np.ndarray,
    d12: np.ndarray,
    d21: np.ndarray,
    d22: np.ndarray,
    w: np.ndarray,
    m1: int,
    m2: int,
    allow_repeats: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Use active patterns from the convex solution to pick h1/h2 activations.
    if d11.shape[1] < m1 or d12.shape[1] < m1:
        if not allow_repeats:
            raise ValueError("Not enough layer-1 patterns to reconstruct m1 neurons.")
        reps = int(np.ceil(m1 / d11.shape[1]))
        d11 = np.tile(d11, (1, reps))
        d12 = np.tile(d12, (1, reps))
    if d22.shape[1] < m2:
        if not allow_repeats:
            raise ValueError("Not enough layer-2 patterns to reconstruct m2 neurons.")
        reps = int(np.ceil(m2 / d22.shape[1]))
        d22 = np.tile(d22, (1, reps))
    h1_1 = d11[:, :m1]
    h1_2 = d12[:, :m1]
    top_idx = np.argsort(-np.abs(w))[:m2]
    h2_2 = d22[:, top_idx]
    if h2_2.shape[1] < m2:
        if not allow_repeats:
            raise ValueError("Not enough layer-2 patterns after selection to reconstruct m2 neurons.")
        reps = int(np.ceil(m2 / h2_2.shape[1]))
        h2_2 = np.tile(h2_2, (1, reps))[:, :m2]
    # For each neuron, pick the closest pattern in D21 to enforce temporal consistency.
    h2_1 = np.zeros((d21.shape[0], m2), dtype=np.float32)
    for j in range(m2):
        diff = np.sum(np.abs(d21 - h2_2[:, [j]]), axis=0)
        h2_1[:, j] = d21[:, np.argmin(diff)]
    return h1_1, h1_2, h2_1, h2_2


def reconstruct_weights(
    x1: np.ndarray,
    x2: np.ndarray,
    h1_1: np.ndarray,
    h1_2: np.ndarray,
    h2_1: np.ndarray,
    h2_2: np.ndarray,
    C: float,
    solver: str,
    solver_opts: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Reconstruction with P1_rec and P2_rec using the provided timestep equations.
    n = x1.shape[0]
    m1 = h1_1.shape[1]
    m2 = h2_2.shape[1]

    zeros_rec1 = np.zeros((n, m1 + 1), dtype=np.float32)
    rec_feat1 = np.hstack([-h1_1, np.ones((n, 1), dtype=np.float32)])
    x_t1 = np.hstack([x1, zeros_rec1])
    x_t2 = np.hstack([x2, rec_feat1])
    x_layer1 = np.vstack([x_t1, x_t2])
    P1_in = np.zeros((x1.shape[1], m1), dtype=np.float32)
    P1_rec = np.zeros((m1 + 1, m1), dtype=np.float32)
    for j in range(m1):
        y_stack = np.concatenate([2 * h1_1[:, j] - 1, 2 * h1_2[:, j] - 1])
        w = solve_svm_hyperplane(x_layer1, y_stack, C, solver, solver_opts)
        P1_in[:, j] = w[: x1.shape[1]]
        P1_rec[:, j] = w[x1.shape[1] :]

    zeros_rec2 = np.zeros((n, m2 + 1), dtype=np.float32)
    rec_feat2 = np.hstack([-h2_1, np.ones((n, 1), dtype=np.float32)])
    x2_t1 = np.hstack([h1_1, zeros_rec2])
    x2_t2 = np.hstack([h1_2, rec_feat2])
    x_layer2 = np.vstack([x2_t1, x2_t2])
    P2_in = np.zeros((m1, m2), dtype=np.float32)
    P2_rec = np.zeros((m2 + 1, m2), dtype=np.float32)
    for j in range(m2):
        y_stack = np.concatenate([2 * h2_1[:, j] - 1, 2 * h2_2[:, j] - 1])
        w = solve_svm_hyperplane(x_layer2, y_stack, C, solver, solver_opts)
        P2_in[:, j] = w[:m1]
        P2_rec[:, j] = w[m1:]
    return P1_in, P1_rec, P2_in, P2_rec


def forward_reconstructed(
    x1: np.ndarray,
    x2: np.ndarray,
    P1_in: np.ndarray,
    P1_rec: np.ndarray,
    P2_in: np.ndarray,
    P2_rec: np.ndarray,
) -> np.ndarray:
    # Threshold activations for the reconstructed SNN weights (T=2).
    h1_1 = (x1 @ P1_in >= 0).astype(np.float32)
    h1_rec = np.hstack([-h1_1, np.ones((x1.shape[0], 1), dtype=np.float32)])
    h1_2 = (x2 @ P1_in + h1_rec @ P1_rec >= 0).astype(np.float32)

    h2_0 = np.zeros((x1.shape[0], P2_rec.shape[0]), dtype=np.float32)
    h2_1 = (h1_1 @ P2_in + h2_0 @ P2_rec >= 0).astype(np.float32)
    h2_rec = np.hstack([-h2_1, np.ones((x1.shape[0], 1), dtype=np.float32)])
    h2_2 = (h1_2 @ P2_in + h2_rec @ P2_rec >= 0).astype(np.float32)
    return h2_2


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
        B4Params(n=20, d=100, m1=20, m2=20, m_star=20, m_repr=1000, beta=1e-3),
        B4Params(n=50, d=50, m1=100, m2=70, m_star=20, m_repr=1000, beta=1e-3),
        B4Params(n=100, d=20, m1=100, m2=120, m_star=20, m_repr=1000, beta=1e-3),
    ]


def append_summary(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_snn_pipeline(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: B4Params,
    config: Dict[str, float],
) -> None:
    zero_gap_tol = 1e-4
    summary_path = config["summary_path"]
    for name, (x1, x2, y) in datasets.items():
        h_repr = create_representation_weights(x1.shape[1], params.m_repr, config["seed"] + 11)
        x1_train = x1[: params.n]
        x2_train = x2[: params.n]
        y_train = y[: params.n]
        x1_test = x1[params.n : params.n + config["n_total"]]
        x2_test = x2[params.n : params.n + config["n_total"]]
        y_test = y[params.n : params.n + config["n_total"]]
        x1_train_tilde = apply_representation(x1_train, h_repr)
        x2_train_tilde = apply_representation(x2_train, h_repr)
        x1_test_tilde = apply_representation(x1_test, h_repr)
        x2_test_tilde = apply_representation(x2_test, h_repr)

        arr_samples = max(
            config["arr_samples"],
            params.m2 * config["arr_sample_factor"],
        )
        d_arr = build_snn_arrangements(
            x1_train_tilde,
            x2_train_tilde,
            config["seed"],
            arr_samples,
            config["sampled_arrangements"],
        )
        d22 = d_arr["D22"]
        primal = solve_primal_convex_snn(
            d22,
            y_train,
            params.beta,
            params.m2,
            config["solver"],
            config["solver_opts"],
        )
        dual = solve_dual_convex_snn(
            d22,
            y_train,
            params.beta,
            params.m2,
            config["solver"],
            config["solver_opts"],
        )
        gap = primal["primal"] - dual["dual"]
        is_zero_gap = abs(gap) <= zero_gap_tol
        if config["sampled_arrangements"]:
            print(
                f"[SNN:{name}] D11 columns={d_arr['D11'].shape[1]} m1={params.m1} "
                f"D22 columns={d_arr['D22'].shape[1]} m2={params.m2}"
            )
        m1_eff = min(params.m1, d_arr["D11"].shape[1], d_arr["D12"].shape[1])
        if m1_eff < params.m1:
            print(f"[SNN:{name}] Reducing m1 from {params.m1} to {m1_eff} due to D11 size.")
        active_idx = np.where(np.abs(primal["w"]) > 1e-8)[0]
        active_hyperplanes = int(active_idx.size)
        if active_hyperplanes == 0:
            active_mean = 0.0
        else:
            active_mean = float(d22[:, active_idx].sum(axis=1).mean())
        convex_train_acc = float(np.mean(np.sign(d22 @ primal["w"]) == y_train))

        h1_1, h1_2, h2_1, h2_2 = select_activation_patterns(
            d_arr["D11"],
            d_arr["D12"],
            d_arr["D21"],
            d_arr["D22"],
            primal["w"],
            m1_eff,
            params.m2,
            config["allow_pattern_repeats"],
        )
        P1_in, P1_rec, P2_in, P2_rec = reconstruct_weights(
            x1_train_tilde,
            x2_train_tilde,
            h1_1,
            h1_2,
            h2_1,
            h2_2,
            config["svm_C"],
            config["solver"],
            config["solver_opts"],
        )
        h2_train_hat = forward_reconstructed(
            x1_train_tilde,
            x2_train_tilde,
            P1_in,
            P1_rec,
            P2_in,
            P2_rec,
        )
        h2_test_hat = forward_reconstructed(
            x1_test_tilde,
            x2_test_tilde,
            P1_in,
            P1_rec,
            P2_in,
            P2_rec,
        )
        v = solve_svm_hyperplane(
            h2_train_hat,
            y_train,
            config["svm_C"],
            config["solver"],
            config["solver_opts"],
        )
        recon_test_preds = np.sign(h2_test_hat @ v)
        recon_test_acc = float(np.mean(recon_test_preds == y_test))

        lif = train_leaky_snn(
            x1_train_tilde,
            x2_train_tilde,
            y_train,
            m1_eff,
            params.m2,
            config["lr"],
            config["epochs"],
        )
        lif_test_preds = eval_leaky_snn(lif["layers"], x1_test_tilde, x2_test_tilde)
        lif_test_acc = float(np.mean(lif_test_preds == y_test))

        append_summary(
            summary_path,
            f"SNN:{name} | Convex train={convex_train_acc:.3f} | "
            f"Reconstructed test={recon_test_acc:.3f} | "
            f"LIF train={lif['acc']:.3f} test={lif_test_acc:.3f} | "
            f"m1_eff={m1_eff} | "
            f"Active hyperplanes={active_hyperplanes} mean_active={active_mean:.3f} | "
            f"Primal={primal['primal']:.6f} Dual={dual['dual']:.6f} Gap={gap:.6f} zero_gap={is_zero_gap}",
        )
        if config["allow_pattern_repeats"]:
            if d_arr["D11"].shape[1] < params.m1 or d_arr["D22"].shape[1] < params.m2:
                append_summary(
                    summary_path,
                    f"SNN:{name} | WARNING pattern_repeats_used for reconstruction",
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--solver", type=str, default="ECOS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_pattern_repeats", action="store_true")
    parser.add_argument("--sampled_arrangements", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    run_ts = int(time.time())
    mode = "debug" if args.debug else "full"
    run_id = f"{mode}_{run_ts}"
    params_grid = b4_param_grid()
    solver_name = args.solver
    if solver_name.upper() == "SCS":
        solver_opts = {"eps": 1e-9, "max_iters": 200000}
    else:
        solver_opts = {"abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9, "max_iters": 50000}

    config = {
        "seed": args.seed,
        "epochs": 50 if args.debug else 500,
        "lr": 1e-2,
        "solver": solver_name,
        "n_total": 200 if args.debug else 3000,
        "svm_C": 1000.0,
        "solver_opts": solver_opts,
        "summary_path": "/Users/hima_3114/Desktop/Paper_1/experiments/summary_results_snn.txt",
        "allow_pattern_repeats": args.allow_pattern_repeats,
        "sampled_arrangements": args.sampled_arrangements,
        "arr_samples": 200 if args.debug else 2000,
        "arr_sample_factor": 20 if args.debug else 5,
    }
    summary_path = Path(config["summary_path"])
    if summary_path.exists():
        existing = summary_path.read_text()
        if existing and not existing.endswith("\n\n"):
            summary_path.write_text(existing + "\n")
    else:
        summary_path.write_text("")
    append_summary(
        config["summary_path"],
        f"RUN {Path(__file__).name} | mode={mode} | time={run_ts}",
    )

    for params in params_grid:
        datasets = {}
        x1, x2, y = generate_two_step_xor(params.n + config["n_total"], args.seed + 1)
        datasets["two_step_xor"] = (x1, x2, y)
        x1, x2, y = generate_moving_gaussian_blobs(params.n + config["n_total"], args.seed + 2)
        datasets["moving_gaussian"] = (x1, x2, y)
        x1, x2, y = generate_rotated_mnist_pairs(params.n + config["n_total"], args.seed + 3)
        datasets["rotated_mnist"] = (x1, x2, y)
        run_snn_pipeline(datasets, params, config)


if __name__ == "__main__":
    main()
