#!/usr/bin/env python3
"""
Convex-Lasso vs STE on MNIST (threshold-only), fixed to be consistent with 1,2,3,4:

(1) Enforce P_last_unique >= n_train by sampling last-layer hyperplanes until enough unique patterns.
    We use target_last = max(n_train, P_list[-1]) (NOT forcing equality).
(2) CVX always uses L1 regularization on last-layer weights: CE + beta*||W||_1.
(3) Beta chosen by validation accuracy (never test).
(4) LR tuned + StepLR decay.

IMPORTANT FIX: We generate patterns ONCE per seed and reuse them for all (beta, lr) grid points.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


# ============================================================
# Device + seed
# ============================================================
def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available; using CPU.")
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Reference-style dataset wrappers
# ============================================================
class PrepareData(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class PrepareData3D(Dataset):
    """Stores cached sign patterns z for training."""
    def __init__(self, X: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.z = torch.from_numpy(z.astype(np.uint8))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.z[idx]


# ============================================================
# Pattern generation (unique-only, enforce last >= n)
# ============================================================
def _col_normalize_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0, keepdims=True) + eps
    return U / norms


def _dedupe_bool_cols_keep_first(D_bool: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seen = set()
    keep = []
    D_u8 = D_bool.astype(np.uint8)
    for j in range(D_u8.shape[1]):
        key = D_u8[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    keep = np.asarray(keep, dtype=np.int64)
    return D_bool[:, keep], U[:, keep]


def generate_sign_patterns_Llayers_enforce_last_ge_n(
    A: np.ndarray,
    P_list: List[int],
    *,
    n_train: int,
    seed: int,
    normalize: bool = True,
    verbose: bool = False,
    target_last: Optional[int] = None,
    chunk_mult: int = 4,
    max_rounds: int = 200,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns:
      z_last_bool: (n_train, target_last) bool
      u_unique_list: list of U_l_unique; last U has shape (in_dim, target_last)
    """
    if len(P_list) < 1:
        raise ValueError("P_list must have length >= 1.")

    if target_last is None:
        target_last = max(n_train, int(P_list[-1]))
    if target_last < n_train:
        raise ValueError("target_last must be >= n_train.")

    rng = np.random.default_rng(seed)
    D_float = A.astype(np.float32)
    u_unique_list: List[np.ndarray] = []
    L = len(P_list)

    for l, P in enumerate(P_list):
        in_dim = D_float.shape[1]
        is_last = (l == L - 1)

        if not is_last:
            U = rng.normal(size=(in_dim, P)).astype(np.float32)
            if normalize:
                U = _col_normalize_np(U)
            D_bool = (D_float @ U >= 0)
            D_u, U_u = _dedupe_bool_cols_keep_first(D_bool, U)
            if verbose:
                print(f"[cvx patterns] layer {l+1}: P={P} -> P_unique={U_u.shape[1]}")
            u_unique_list.append(U_u)
            D_float = D_u.astype(np.float32)
            continue

        # last layer: enforce >= target_last unique
        uniq: Dict[bytes, np.ndarray] = {}
        rounds = 0
        chunk_P = max(P, chunk_mult * target_last)

        while len(uniq) < target_last and rounds < max_rounds:
            rounds += 1
            U = rng.normal(size=(in_dim, chunk_P)).astype(np.float32)
            if normalize:
                U = _col_normalize_np(U)
            D_bool = (D_float @ U >= 0)
            D_u8 = D_bool.astype(np.uint8)

            for j in range(D_u8.shape[1]):
                key = D_u8[:, j].tobytes()
                if key not in uniq:
                    uniq[key] = U[:, j].copy()
                    if len(uniq) >= target_last:
                        break

            if verbose:
                print(f"[cvx patterns] layer {l+1} enforce: round={rounds}/{max_rounds} "
                      f"uniques={len(uniq)}/{target_last}")

        if len(uniq) < target_last:
            raise RuntimeError(
                f"Could not collect target_last={target_last} unique last-layer patterns "
                f"(got {len(uniq)}). Increase chunk_mult/max_rounds or widen earlier layers."
            )

        keys = list(uniq.keys())[:target_last]
        D_last = np.stack([np.frombuffer(k, dtype=np.uint8) for k in keys], axis=1)
        z_last_bool = (D_last.astype(np.float32) >= 0.5)
        U_last = np.stack([uniq[k] for k in keys], axis=1).astype(np.float32)

        if verbose:
            print(f"[cvx patterns] layer {l+1}: forced P_last_unique={target_last} (>= n_train={n_train})")

        u_unique_list.append(U_last)
        return z_last_bool, u_unique_list

    raise RuntimeError("Internal error.")


def forward_patterns_Llayers_torch(
    x: torch.Tensor,
    u_unique_list: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    D = x.view(x.shape[0], -1).float()
    for U in u_unique_list:
        U_t = torch.from_numpy(U).float().to(device)
        D = (D @ U_t >= 0).float()
    return (D >= 0.5)


# ============================================================
# CVX last layer (trainable W) + L1 regularization
# ============================================================
class CvxLastLayer(nn.Module):
    def __init__(self, P_last: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(P_last, num_classes), requires_grad=True)

    def forward(self, sign_patterns: torch.Tensor) -> torch.Tensor:
        return sign_patterns.float() @ self.W


def cvx_loss_ce_l1(logits: torch.Tensor, y: torch.Tensor, model: CvxLastLayer, beta: float) -> torch.Tensor:
    return F.cross_entropy(logits, y) + beta * model.W.abs().sum()


@torch.no_grad()
def cvx_eval_acc_cached_z(model: CvxLastLayer, loader3d: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for _x, _y, _z in loader3d:
        _y = _y.to(device)
        _z = _z.to(device)
        pred = torch.argmax(model(_z), dim=1)
        correct += (pred == _y).sum().item()
        total += _y.numel()
    return correct / total


@torch.no_grad()
def cvx_eval_acc_recompute_z(
    model: CvxLastLayer,
    loader2d: DataLoader,
    u_unique_list: List[np.ndarray],
    device: torch.device,
) -> float:
    model.eval()
    correct, total = 0, 0
    for _x, _y in loader2d:
        _x = _x.to(device)
        _y = _y.to(device)
        _z = forward_patterns_Llayers_torch(_x, u_unique_list, device)
        pred = torch.argmax(model(_z), dim=1)
        correct += (pred == _y).sum().item()
        total += _y.numel()
    return correct / total


def train_cvx_head(
    train_loader3d: DataLoader,
    val_loader2d: DataLoader,
    u_unique_list: List[np.ndarray],
    *,
    P_last: int,
    beta: float,
    lr: float,
    epochs: int,
    device: torch.device,
    optimizer_name: str,
    step_size: int,
    gamma: float,
) -> Tuple[CvxLastLayer, float]:
    model = CvxLastLayer(P_last, 10).to(device)

    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for _x, _y, _z in train_loader3d:
            _y = _y.to(device)
            _z = _z.to(device)
            logits = model(_z)
            loss = cvx_loss_ce_l1(logits, _y, model, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_acc = cvx_eval_acc_recompute_z(model, val_loader2d, u_unique_list, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


# ============================================================
# STE model + path regularization (unchanged)
# ============================================================
class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STE_Threshold_FCN(nn.Module):
    def __init__(self, in_dim: int, widths: List[int], num_classes: int = 10):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev = in_dim
        for w in widths:
            self.hidden.append(nn.Linear(prev, w, bias=False))
            prev = w
        self.out = nn.Linear(prev, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for fc in self.hidden:
            h = ThresholdSTE.apply(fc(h))
        return self.out(h)


def path_reg_squared(model: STE_Threshold_FCN) -> torch.Tensor:
    device = next(model.parameters()).device
    d0 = model.hidden[0].weight.shape[1]
    s = torch.ones(d0, device=device)
    for fc in model.hidden:
        s = fc.weight.pow(2).matmul(s)
    return torch.sum(model.out.weight.pow(2).matmul(s))


@torch.no_grad()
def ste_eval_acc(model: STE_Threshold_FCN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device).view(xb.size(0), -1)
        yb = yb.to(device)
        pred = torch.argmax(model(xb), dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def train_ste(
    widths: List[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    epochs: int,
    beta_path: float,
    device: torch.device,
    step_size: int,
    gamma: float,
) -> Tuple[STE_Threshold_FCN, float]:
    model = STE_Threshold_FCN(28 * 28, widths, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).view(xb.size(0), -1)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + beta_path * path_reg_squared(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_acc = ste_eval_acc(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


# ============================================================
# MNIST + split
# ============================================================
def load_mnist_normalized(root: str = "data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def take_first_n_as_numpy(ds: Dataset, n: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(n):
        x, y = ds[i]
        xs.append(x.view(-1).numpy())
        ys.append(int(y))
    return np.stack(xs, 0).astype(np.float32), np.array(ys, dtype=np.int64)


def split_train_val(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# ============================================================
# MAIN (patterns generated once per seed, grid search reuses them)
# ============================================================
@dataclass
class GridSpec:
    betas: List[float]
    lrs: List[float]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train_total", type=int, default=1250)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--P_list", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)

    ap.add_argument("--cvx_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--cvx_step_size", type=int, default=30)
    ap.add_argument("--cvx_gamma", type=float, default=0.5)
    ap.add_argument("--beta_grid", type=float, nargs="+", default=[1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0])
    ap.add_argument("--lr_grid", type=float, nargs="+", default=[1e-2, 5e-3, 1e-3, 1e-1])

    ap.add_argument("--ste_lr", type=float, default=1e-3)
    ap.add_argument("--ste_beta_path", type=float, default=1e-4)
    ap.add_argument("--ste_step_size", type=int, default=30)
    ap.add_argument("--ste_gamma", type=float, default=0.5)

    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--verbose_patterns", action="store_true")

    args = ap.parse_args()
    device = get_device(args.device)

    print(f"[info] device={device} epochs={args.epochs} batch={args.batch}")
    n_train = int(round(args.n_train_total * (1.0 - args.val_frac)))
    print(f"[info] n_train_total={args.n_train_total} val_frac={args.val_frac} => n_train={n_train} n_test={args.n_test}")
    print(f"[info] P_list={args.P_list} (last-layer target_last = max(n_train, P_list[-1]))")
    print(f"[info] CVX grid betas={args.beta_grid} lrs={args.lr_grid}")

    train_ds, test_ds = load_mnist_normalized()

    cvx_tests, ste_tests = [], []

    for s in args.seeds:
        set_seed(s)

        X_total, y_total = take_first_n_as_numpy(train_ds, args.n_train_total)
        X_test, y_test = take_first_n_as_numpy(test_ds, args.n_test)
        X_train, y_train, X_val, y_val = split_train_val(X_total, y_total, args.val_frac, seed=s)

        # ---- Generate patterns ONCE per seed (critical fix) ----
        target_last = max(X_train.shape[0], int(args.P_list[-1]))
        z_train_bool, u_unique_list = generate_sign_patterns_Llayers_enforce_last_ge_n(
            X_train,
            list(args.P_list),
            n_train=X_train.shape[0],
            seed=s,
            normalize=True,
            verbose=args.verbose_patterns,
            target_last=target_last,
        )
        z_train = z_train_bool.astype(np.uint8)
        P_last = z_train.shape[1]

        train_ds3d = PrepareData3D(X_train, y_train, z_train)
        val_ds2d = PrepareData(X_val, y_val)
        test_ds2d = PrepareData(X_test, y_test)

        train_loader3d = DataLoader(train_ds3d, batch_size=args.batch, shuffle=True)
        val_loader2d = DataLoader(val_ds2d, batch_size=512, shuffle=False)
        test_loader2d = DataLoader(test_ds2d, batch_size=512, shuffle=False)

        # ---- Grid search over SAME features ----
        best_val, best_beta, best_lr, best_model = -1.0, None, None, None
        for beta in args.beta_grid:
            for lr in args.lr_grid:
                model, val_acc = train_cvx_head(
                    train_loader3d, val_loader2d, u_unique_list,
                    P_last=P_last, beta=beta, lr=lr, epochs=args.epochs,
                    device=device, optimizer_name=args.cvx_optimizer,
                    step_size=args.cvx_step_size, gamma=args.cvx_gamma,
                )
                if val_acc > best_val:
                    best_val, best_beta, best_lr, best_model = val_acc, beta, lr, model

        cvx_test = cvx_eval_acc_recompute_z(best_model, test_loader2d, u_unique_list, device)

        # ---- STE widths match realized unique widths (per layer) ----
        ste_widths = [U.shape[1] for U in u_unique_list]
        Xtr_t = torch.from_numpy(X_train).float().view(-1, 1, 28, 28)
        ytr_t = torch.from_numpy(y_train).long()
        Xva_t = torch.from_numpy(X_val).float().view(-1, 1, 28, 28)
        yva_t = torch.from_numpy(y_val).long()
        Xte_t = torch.from_numpy(X_test).float().view(-1, 1, 28, 28)
        yte_t = torch.from_numpy(y_test).long()

        ste_train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=args.batch, shuffle=True)
        ste_val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=512, shuffle=False)
        ste_test_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=512, shuffle=False)

        ste_model, ste_val = train_ste(
            ste_widths, ste_train_loader, ste_val_loader,
            lr=args.ste_lr, epochs=args.epochs, beta_path=args.ste_beta_path,
            device=device, step_size=args.ste_step_size, gamma=args.ste_gamma,
        )
        ste_test = ste_eval_acc(ste_model, ste_test_loader, device)

        cvx_tests.append(float(cvx_test))
        ste_tests.append(float(ste_test))

        print(f"[seed {s}] CVX test={cvx_test:.4f} (val={best_val:.4f}, beta={best_beta}, lr={best_lr}, P_last={P_last}) "
              f"| STE test={ste_test:.4f} (val={ste_val:.4f})")

    cvx_mean, cvx_std = float(np.mean(cvx_tests)), float(np.std(cvx_tests))
    ste_mean, ste_std = float(np.mean(ste_tests)), float(np.std(ste_tests))

    print("\n=== FINAL (mean ± std) ===")
    print(f"CVX test_acc = {cvx_mean:.4f} ± {cvx_std:.4f}")
    print(f"STE test_acc = {ste_mean:.4f} ± {ste_std:.4f}")


if __name__ == "__main__":
    main()
