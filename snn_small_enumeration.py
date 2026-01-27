import time
from dataclasses import dataclass
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


def generate_b1_synthetic(
    seed: int,
    n_train: int = 5,
    n_test: int = 50,
    shift_mode: str = "fixed",
    shift_value: float = 0.5,
    shift_std: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Section B.1 (Figure 5) base: X = [-2, -1, 0, 1, 2]^T.

    For T=2, we define x1 = X and x2 = X + shift. Labels
    come from a random T=2 threshold SNN with m1=m2=2.
    """
    rng = np.random.default_rng(seed)
    x_base = np.arange(-2, 3, dtype=np.float32).reshape(-1, 1)
    if x_base.shape[0] != n_train:
        raise ValueError("n_train must be 5 to match Section B.1.")
    x1_train = x_base.copy()
    x1_test = rng.uniform(-2.5, 2.5, size=(n_test, 1)).astype(np.float32)
    if shift_mode == "fixed":
        shift_train = np.full((n_train, 1), shift_value, dtype=np.float32)
        shift_test = np.full((n_test, 1), shift_value, dtype=np.float32)
    elif shift_mode == "gaussian":
        shift_train = rng.normal(loc=shift_value, scale=shift_std, size=(n_train, 1)).astype(np.float32)
        shift_test = rng.normal(loc=shift_value, scale=shift_std, size=(n_test, 1)).astype(np.float32)
    else:
        raise ValueError("shift_mode must be 'fixed' or 'gaussian'.")
    x2_train = x_base + shift_train
    x2_test = x1_test + shift_test

    # Ground-truth T=2 threshold SNN with m1=m2=2
    m1_star = 2
    m2_star = 2
    p1 = rng.normal(size=(1, m1_star)).astype(np.float32)
    p2_in = rng.normal(size=(m1_star, m2_star)).astype(np.float32)
    p2_rec = rng.normal(size=(m2_star, m2_star)).astype(np.float32)
    v = rng.normal(size=(m2_star,)).astype(np.float32)

    def forward_t2(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        h1_1 = (x1 @ p1 >= 0).astype(np.float32)
        h2_0 = np.zeros((x1.shape[0], m2_star), dtype=np.float32)
        h2_1 = (h1_1 @ p2_in + h2_0 @ p2_rec >= 0).astype(np.float32)
        h1_2 = (x2 @ p1 >= 0).astype(np.float32)
        h2_2 = (h1_2 @ p2_in + h2_1 @ p2_rec >= 0).astype(np.float32)
        return h2_2

    h2_train = forward_t2(x1_train, x2_train)
    h2_test = forward_t2(x1_test, x2_test)
    y_train = np.where(h2_train @ v >= 0, 1.0, -1.0).astype(np.float32)
    y_test = np.where(h2_test @ v >= 0, 1.0, -1.0).astype(np.float32)
    return x1_train, x2_train, x1_test, x2_test, y_train, y_test


def generate_two_step_xor_small(
    seed: int,
    n_train: int = 16,
    n_test: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1_train = rng.integers(0, 2, size=(n_train, 2)).astype(np.float32)
    x2_train = rng.integers(0, 2, size=(n_train, 2)).astype(np.float32)
    x1_test = rng.integers(0, 2, size=(n_test, 2)).astype(np.float32)
    x2_test = rng.integers(0, 2, size=(n_test, 2)).astype(np.float32)
    predicate1 = (x1_train @ np.array([1.0, -1.0]) >= 0).astype(int)
    predicate2 = (x2_train @ np.array([1.0, -1.0]) >= 0).astype(int)
    y_train = (predicate1 ^ predicate2) * 2 - 1
    predicate1_t = (x1_test @ np.array([1.0, -1.0]) >= 0).astype(int)
    predicate2_t = (x2_test @ np.array([1.0, -1.0]) >= 0).astype(int)
    y_test = (predicate1_t ^ predicate2_t) * 2 - 1
    return x1_train, x2_train, x1_test, x2_test, y_train.astype(np.float32), y_test.astype(np.float32)


def apply_hyperplane_weights(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    if z.shape[1] != u.shape[0]:
        raise ValueError("apply_hyperplane_weights: z and u shapes are incompatible.")
    return (z @ u >= 0).astype(np.float32)


def enumerate_arrangements_exact(
    z: np.ndarray,
    solver: str,
    solver_opts: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all hyperplane arrangements 1{Z u >= 0} exactly by checking
    strict separability for every pattern in {0,1}^n and fixing the scale
    via u[j] = ±1 for some coordinate j.
    """
    n, p = z.shape
    patterns = []
    weights = []
    for bits in itertools.product([0, 1], repeat=n):
        b = np.array(bits, dtype=np.float32)
        if np.all(b == 1):
            u_val = np.zeros((p,), dtype=np.float32)
            s = (z @ u_val >= 0).astype(np.float32)
            patterns.append(s)
            weights.append(u_val)
            continue
        u = cp.Variable(p)
        t = cp.Variable()
        constraints = [cp.norm(u, 2) <= 1, t >= 0]
        for i in range(n):
            zi = z[i, :]
            if b[i] == 1:
                constraints.append(zi @ u >= 0)
            else:
                constraints.append(zi @ u <= -t)
        problem = cp.Problem(cp.Maximize(t), constraints)
        problem.solve(solver=solver, **solver_opts)
        if t.value is None:
            continue
        if float(t.value) > 0.0:
            u_val = u.value.astype(np.float32)
            s = (z @ u_val >= 0).astype(np.float32)
            patterns.append(s)
            weights.append(u_val)
    if not patterns:
        raise ValueError("Exact enumeration produced no patterns.")
    return np.stack(patterns, axis=1), np.stack(weights, axis=1)


def a_snn_from_blocks_full(
    block1: np.ndarray,
    block2: np.ndarray,
    op: str,
    bias: np.ndarray,
    solver: str,
    solver_opts: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    if op not in {"+", "-"}:
        raise ValueError(f"Unsupported op '{op}', expected '+' or '-'.")
    sign = 1.0 if op == "+" else -1.0
    z = np.concatenate([block1, sign * block2, bias], axis=1)
    return enumerate_arrangements_exact(z, solver, solver_opts)


def build_snn_arrangements_full(
    x1: np.ndarray,
    x2: np.ndarray,
    solver: str,
    solver_opts: Dict[str, float],
) -> Dict[str, np.ndarray]:
    n = x1.shape[0]
    x0 = np.zeros((n, 1), dtype=np.float32)
    h0 = np.zeros((n, 1), dtype=np.float32)
    ones = np.ones((n, 1), dtype=np.float32)

    d11, u11 = a_snn_from_blocks_full(x1, h0, "+", ones, solver, solver_opts)
    d12, u12 = a_snn_from_blocks_full(x2, d11, "-", ones, solver, solver_opts)
    d21, u21 = a_snn_from_blocks_full(d11, x0, "+", ones, solver, solver_opts)
    d22, u22 = a_snn_from_blocks_full(d12, d21, "-", ones, solver, solver_opts)

    return {
        "D11": d11.astype(np.float32),
        "D12": d12.astype(np.float32),
        "D21": d21.astype(np.float32),
        "D22": d22.astype(np.float32),
        "U11": u11.astype(np.float32),
        "U12": u12.astype(np.float32),
        "U21": u21.astype(np.float32),
        "U22": u22.astype(np.float32),
    }


def build_snn_arrangements_from_weights(
    x1: np.ndarray,
    x2: np.ndarray,
    u_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    n = x1.shape[0]
    x0 = np.zeros((n, 1), dtype=np.float32)
    h0 = np.zeros((n, 1), dtype=np.float32)
    ones = np.ones((n, 1), dtype=np.float32)
    z11 = np.concatenate([x1, h0, ones], axis=1)
    d11 = apply_hyperplane_weights(z11, u_dict["U11"])
    z12 = np.concatenate([x2, -d11, ones], axis=1)
    d12 = apply_hyperplane_weights(z12, u_dict["U12"])
    z21 = np.concatenate([d11, x0, ones], axis=1)
    d21 = apply_hyperplane_weights(z21, u_dict["U21"])
    z22 = np.concatenate([d12, -d21, ones], axis=1)
    d22 = apply_hyperplane_weights(z22, u_dict["U22"])
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
    solver: str,
    solver_opts: Dict[str, float],
) -> Dict[str, float]:
    w = cp.Variable(d22.shape[1])
    objective = 0.5 * cp.sum_squares(d22 @ w - y) + beta * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=solver, **solver_opts)
    w_val = w.value
    preds = np.sign(d22 @ w_val)
    acc = float(np.mean(preds == y))
    return {"primal": problem.value, "w": w_val, "acc": acc}


def solve_dual_convex_snn(
    d22: np.ndarray,
    y: np.ndarray,
    beta: float,
    solver: str,
    solver_opts: Dict[str, float],
) -> Dict[str, float]:
    lam = cp.Variable(d22.shape[0])
    constraints = [d22.T @ lam <= beta, d22.T @ lam >= -beta]
    objective = cp.Maximize(-0.5 * cp.sum_squares(lam) - lam @ y)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **solver_opts)
    return {"dual": problem.value, "lambda": lam.value}


def solve_svm_hyperplane(
    x: np.ndarray,
    y: np.ndarray,
    C: float,
    solver: str,
    solver_opts: Dict[str, float],
) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if d11.shape[1] < m1 or d12.shape[1] < m1:
        raise ValueError("Not enough layer-1 patterns for exact reconstruction.")
    if d22.shape[1] < m2:
        raise ValueError("Not enough layer-2 patterns for exact reconstruction.")
    h1_1 = d11[:, :m1]
    h1_2 = d12[:, :m1]
    top_idx = np.argsort(-np.abs(w))[:m2]
    h2_2 = d22[:, top_idx]
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x1.shape[0]
    m1 = h1_1.shape[1]
    m2 = h2_2.shape[1]
    P1 = np.zeros((x1.shape[1], m1), dtype=np.float32)
    for j in range(m1):
        w = solve_svm_hyperplane(x1, 2 * h1_1[:, j] - 1, C, solver, solver_opts)
        P1[:, j] = w
    h2_0 = np.zeros((n, m2), dtype=np.float32)
    x_t1 = np.hstack([h1_1, h2_0])
    x_t2 = np.hstack([h1_2, h2_1])
    x_layer2 = np.vstack([x_t1, x_t2])
    P2_in = np.zeros((m1, m2), dtype=np.float32)
    P2_rec = np.zeros((m2, m2), dtype=np.float32)
    for j in range(m2):
        y_stack = np.concatenate([2 * h2_1[:, j] - 1, 2 * h2_2[:, j] - 1])
        w = solve_svm_hyperplane(x_layer2, y_stack, C, solver, solver_opts)
        P2_in[:, j] = w[:m1]
        P2_rec[:, j] = w[m1:]
    return P1, P2_in, P2_rec


def forward_reconstructed(
    x1: np.ndarray,
    x2: np.ndarray,
    P1: np.ndarray,
    P2_in: np.ndarray,
    P2_rec: np.ndarray,
) -> np.ndarray:
    h1_1 = (x1 @ P1 >= 0).astype(np.float32)
    h2_0 = np.zeros((x1.shape[0], P2_rec.shape[0]), dtype=np.float32)
    h2_1 = (h1_1 @ P2_in + h2_0 @ P2_rec >= 0).astype(np.float32)
    h1_2 = (x2 @ P1 >= 0).astype(np.float32)
    h2_2 = (h1_2 @ P2_in + h2_1 @ P2_rec >= 0).astype(np.float32)
    return h2_2


def train_leaky_snn(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    m1: int,
    m2: int,
    lr: float,
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
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
        spk1, mem1 = lif1(fc1(x1_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        spk1, mem1 = lif1(fc1(x2_t), mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
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


def eval_leaky_snn(
    layers: Tuple[nn.Linear, snn.Leaky, nn.Linear, snn.Leaky, nn.Linear],
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
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


@dataclass
class SmallConfig:
    m1_max: int = 20
    m2_max: int = 20
    beta: float = 1.0
    svm_C: float = 1e6
    solver: str = "SCS"
    solver_opts: Dict[str, float] = None
    lif_lr: float = 1e-2
    lif_epochs: int = 500


def main() -> None:
    set_seed(0)
    config = SmallConfig()
    if config.solver_opts is None:
        config.solver_opts = {"eps": 1e-9, "max_iters": 200000}
    def run_case(label: str, data_fn, data_kwargs: Dict[str, float], run_idx: int) -> None:
        x1_train, x2_train, x1_test, x2_test, y_train, y_test = data_fn(**data_kwargs)
        beta_case = config.beta
        if data_kwargs.get("shift_mode") == "gaussian":
            beta_case = config.beta * 2.0
        start = time.time()
        d_arr = build_snn_arrangements_full(
            x1_train,
            x2_train,
            config.solver,
            config.solver_opts,
        )
        d_arr_test = build_snn_arrangements_from_weights(
            x1_test,
            x2_test,
            {
                "U11": d_arr["U11"],
                "U12": d_arr["U12"],
                "U21": d_arr["U21"],
                "U22": d_arr["U22"],
            },
        )
        d22 = d_arr["D22"]
        primal = solve_primal_convex_snn(d22, y_train, beta_case, config.solver, config.solver_opts)
        dual = solve_dual_convex_snn(d22, y_train, beta_case, config.solver, config.solver_opts)
        gap = primal["primal"] - dual["dual"]
        convex_train_acc = float(np.mean(np.sign(d22 @ primal["w"]) == y_train))
        active_idx = np.where(np.abs(primal["w"]) > 1e-10)[0]
        m2 = min(config.m2_max, d_arr["D22"].shape[1])
        m1 = min(config.m1_max, d_arr["D11"].shape[1], d_arr["D12"].shape[1])
        if m1 == 0:
            raise ValueError("No layer-1 patterns available for reconstruction.")
        h1_1, h1_2, h2_1, h2_2 = select_activation_patterns(
            d_arr["D11"],
            d_arr["D12"],
            d_arr["D21"],
            d_arr["D22"],
            primal["w"],
            m1,
            m2,
        )
        P1, P2_in, P2_rec = reconstruct_weights(
            x1_train,
            x2_train,
            h1_1,
            h1_2,
            h2_1,
            h2_2,
            config.svm_C,
            config.solver,
            config.solver_opts,
        )
        h2_train_hat = forward_reconstructed(x1_train, x2_train, P1, P2_in, P2_rec)
        h2_test_hat = forward_reconstructed(x1_test, x2_test, P1, P2_in, P2_rec)
        v = solve_svm_hyperplane(
            h2_train_hat,
            y_train,
            config.svm_C,
            config.solver,
            config.solver_opts,
        )
        recon_train_acc = float(np.mean(np.sign(h2_train_hat @ v) == y_train))
        recon_test_acc = float(np.mean(np.sign(h2_test_hat @ v) == y_test))
        lif = train_leaky_snn(
            x1_train,
            x2_train,
            y_train,
            m1,
            m2,
            config.lif_lr,
            config.lif_epochs,
            seed=1000 + run_idx,
        )
        lif_test_preds = eval_leaky_snn(lif["layers"], x1_test, x2_test)
        lif_test_acc = float(np.mean(lif_test_preds == y_test))
        test_acc_delta = recon_test_acc - lif_test_acc
        elapsed = time.time() - start
        print(f"== Small SNN (T=2) full enumeration | {label} | run={run_idx} ==")
        print(f"D11={d_arr['D11'].shape[1]} D12={d_arr['D12'].shape[1]} "
              f"D21={d_arr['D21'].shape[1]} D22={d_arr['D22'].shape[1]}")
        print(
            f"Convex train={convex_train_acc:.3f} beta={beta_case:.3f} "
            f"active_hyperplanes={active_idx.size}"
        )
        print(f"Recon train={recon_train_acc:.3f} Recon test={recon_test_acc:.3f}")
        print(f"LIF train={lif['acc']:.3f} LIF test={lif_test_acc:.3f}")
        print(f"test-acc-delta={test_acc_delta:.3f}")
        print(f"Primal={primal['primal']:.6f} Dual={dual['dual']:.6f} Gap={gap:.6f}")
        print(f"Elapsed={elapsed:.3f}s")

    datasets = [
        ("fixed_shift", generate_b1_synthetic, {"seed": 0, "n_train": 5, "n_test": 50, "shift_mode": "fixed"}),
        ("gaussian_shift", generate_b1_synthetic, {"seed": 0, "n_train": 5, "n_test": 50, "shift_mode": "gaussian"}),
        ("two_step_xor_small", generate_two_step_xor_small, {"seed": 0, "n_train": 16, "n_test": 64}),
    ]
    for run_idx in range(5):
        for label, data_fn, data_kwargs in datasets:
            data_kwargs["seed"] = run_idx
            run_case(label, data_fn, data_kwargs, run_idx)


if __name__ == "__main__":
    main()
