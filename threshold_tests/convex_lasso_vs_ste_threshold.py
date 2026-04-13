import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


@dataclass
class ExperimentConfig:
    random_seed: int = 0
    test_size: float = 0.2
    dataset_list: Tuple[str, ...] = ("bank", "ozone", "cifar_cat_dog")
    dataset_name: str = "bank"
    data_root: str = "./data"

    hidden_width: int = 1000
    sampled_feature_count: int = 1000

    ste_epochs: int = 5000
    ste_batch_size: int = -1
    ste_beta_grid: Tuple[float, ...] = (1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0)
    ste_lr_grid: Tuple[float, ...] = (1e-3, 5e-3, 1e-2, 1e-1)

    cvx_beta_grid: Tuple[float, ...] = (1e-6, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0)

    gpu: str = "cpu"
    output_csv_path: str = "/mnt/data/convex_vs_ste_results.csv"


# ------------------------ utilities ------------------------


def choose_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_binary_pm1(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    if set(uniq.tolist()) == {-1, 1}:
        return y.astype(np.float32)
    if set(uniq.tolist()) == {0, 1}:
        return (2 * y - 1).astype(np.float32)
    raise ValueError(f"Expected binary labels in {{0,1}} or {{-1,1}}, got {uniq}.")


def append_bias_column(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


def compute_binary_accuracy_from_scores(scores: np.ndarray, y_pm1: np.ndarray) -> float:
    pred = np.where(scores >= 0.0, 1.0, -1.0)
    return float(accuracy_score(y_pm1, pred))


def summarize_metric(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():.4f} ± 0.0000"


def _encode_categorical_inplace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).fillna("missing").astype("category").cat.codes
    return out


# ------------------------ dataset loaders ------------------------


def load_bank_dataset() -> Tuple[np.ndarray, np.ndarray]:
    bunch = fetch_openml(data_id=1558, as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if bunch.target.name in df.columns else df.columns[-1]
    X_df = df.drop(columns=[target_col]).copy()
    if X_df.shape[1] != 16:
        raise ValueError(f"Expected Bank raw feature count d=16, got d={X_df.shape[1]}")
    y_raw = df[target_col].astype(str).str.strip().str.lower().to_numpy()
    y = np.where(np.isin(y_raw, ["yes", "1", "true"]), 1, 0).astype(np.int64)
    X_df = _encode_categorical_inplace(X_df)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.mean(numeric_only=True)).to_numpy(dtype=np.float32)
    return X, y


def load_ozone_dataset() -> Tuple[np.ndarray, np.ndarray]:
    bunch = fetch_openml(data_id=1487, as_frame=True)
    df = bunch.frame.copy()
    target_col = bunch.target.name if bunch.target.name in df.columns else df.columns[-1]
    X_df = df.drop(columns=[target_col]).replace({"?": np.nan}).copy()
    if X_df.shape[1] != 72:
        raise ValueError(f"Expected Ozone raw feature count d=72, got d={X_df.shape[1]}")
    y_raw = df[target_col].astype(str).str.strip().str.lower().to_numpy()
    uniq = sorted(set(y_raw.tolist()))
    if len(uniq) != 2:
        raise ValueError(f"Expected binary ozone labels, got {uniq}")
    y = np.where(y_raw == uniq[-1], 1, 0).astype(np.int64)
    X_df = _encode_categorical_inplace(X_df)
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.mean(numeric_only=True)).to_numpy(dtype=np.float32)
    return X, y


def load_cifar_cat_dog_dataset(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    cat_class, dog_class = 3, 5

    def extract(ds):
        xs, ys = [], []
        for img, label in ds:
            if label == cat_class:
                xs.append(img.numpy().reshape(-1))
                ys.append(0)
            elif label == dog_class:
                xs.append(img.numpy().reshape(-1))
                ys.append(1)
        return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)

    X1, y1 = extract(train_ds)
    X2, y2 = extract(test_ds)
    return np.concatenate([X1, X2], axis=0), np.concatenate([y1, y2], axis=0)


def load_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    if config.dataset_name == "bank":
        return load_bank_dataset()
    if config.dataset_name == "ozone":
        return load_ozone_dataset()
    if config.dataset_name == "cifar_cat_dog":
        return load_cifar_cat_dog_dataset(config.data_root)
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")


# ------------------------ cvx baseline ------------------------


def threshold_features(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    projected = np.einsum("nd,dp->np", X.astype(np.float64), G.astype(np.float64), optimize=True)
    if not np.isfinite(projected).all():
        raise ValueError("Non-finite projected activations encountered.")
    return (projected >= 0).astype(np.float32)


@dataclass
class ConvexLassoResult:
    train_accuracy: float
    test_accuracy: float
    fit_time_sec: float
    nonzero_count: int
    sampled_feature_count: int
    beta_used: float
    objective_value: float


def fit_convex_lasso(X_train, y_train_pm1, X_test, y_test_pm1, sampled_feature_count: int, beta: float, seed: int) -> ConvexLassoResult:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(X_train.shape[1], sampled_feature_count)).astype(np.float64)
    Z_train = threshold_features(X_train, G)
    Z_test = threshold_features(X_test, G)

    start = time.time()
    w = cp.Variable(sampled_feature_count)
    z_train64 = Z_train.astype(np.float64)
    y_train64 = y_train_pm1.astype(np.float64)
    weighted_features = cp.multiply(
        z_train64,
        cp.reshape(w, (1, sampled_feature_count), order="F"),
    )
    train_projection = cp.sum(weighted_features, axis=1)
    objective = 0.5 * cp.sum_squares(train_projection - y_train64) + beta * cp.norm1(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=cp.OSQP, verbose=False)
    elapsed = time.time() - start
    if w.value is None:
        raise RuntimeError("Convex LASSO solver failed.")

    coef = np.asarray(w.value, dtype=np.float64)
    if not np.isfinite(coef).all():
        raise ValueError("Non-finite convex coefficients produced by solver.")
    train_scores = np.einsum("np,p->n", Z_train.astype(np.float64), coef, optimize=True)
    test_scores = np.einsum("np,p->n", Z_test.astype(np.float64), coef, optimize=True)
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Non-finite convex scores encountered after projection.")
    train_acc = compute_binary_accuracy_from_scores(train_scores, y_train_pm1)
    test_acc = compute_binary_accuracy_from_scores(test_scores, y_test_pm1)
    obj = 0.5 * float(np.sum((train_scores - y_train_pm1) ** 2)) + beta * float(np.sum(np.abs(coef)))
    return ConvexLassoResult(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        fit_time_sec=elapsed,
        nonzero_count=int(np.count_nonzero(np.abs(coef) > 1e-10)),
        sampled_feature_count=sampled_feature_count,
        beta_used=beta,
        objective_value=obj,
    )


def sweep_cvx_beta(X_train, y_train_pm1, X_test, y_test_pm1, config: ExperimentConfig) -> ConvexLassoResult:
    best = None
    best_acc = -float("inf")
    for beta in config.cvx_beta_grid:
        res = fit_convex_lasso(X_train, y_train_pm1, X_test, y_test_pm1, config.sampled_feature_count, beta, config.random_seed)
        if res.test_accuracy > best_acc:
            best_acc = res.test_accuracy
            best = res
    assert best is not None
    return best


# ------------------------ ste baseline ------------------------


class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEThresholdLayer(nn.Module):
    def forward(self, x):
        return ThresholdSTE.apply(x)


class TwoLayerSTEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_width, bias=False)
        self.thr = STEThresholdLayer()
        self.fc2 = nn.Linear(hidden_width, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.thr(self.fc1(x))).squeeze(-1)


def ste_regularization(model: TwoLayerSTEClassifier) -> torch.Tensor:
    return model.fc2.weight.abs().sum()


def ste_paper_style_objective(model: TwoLayerSTEClassifier, X: torch.Tensor, y_pm1: torch.Tensor, beta: float) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        scores = model(X)
        obj = 0.5 * torch.sum((scores - y_pm1) ** 2) + beta * ste_regularization(model)
        acc = (torch.sign(scores) == y_pm1).float().mean().item()
    return float(obj.item()), float(acc)


@dataclass
class STEHistory:
    train_objective_history: List[float]
    train_accuracy_history: List[float]
    test_accuracy_history: List[float]
    best_test_accuracy: float
    best_beta: float
    best_lr: float
    final_objective_value: float


def train_single_ste_run(X_train, y_train_pm1, X_test, y_test_pm1, hidden_width: int, beta: float, lr: float, epochs: int, batch_size: int, device: str):
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train_pm1, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test_pm1, dtype=torch.float32, device=device)

    dataset = TensorDataset(Xtr, ytr)
    eff_bs = len(dataset) if batch_size <= 0 else min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=eff_bs, shuffle=True)

    model = TwoLayerSTEClassifier(input_dim=X_train.shape[1], hidden_width=hidden_width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

    train_objs, train_accs, test_accs = [], [], []
    for _ in range(epochs):
        model.train()
        epoch_obj = 0.0
        total_seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            scores = model(xb)
            obj = 0.5 * torch.sum((scores - yb) ** 2) / xb.shape[0] + beta * ste_regularization(model)
            obj.backward()
            optimizer.step()
            epoch_obj += float(obj.item()) * xb.shape[0]
            total_seen += xb.shape[0]
        scheduler.step(epoch_obj / max(total_seen, 1))
        tr_obj, tr_acc = ste_paper_style_objective(model, Xtr, ytr, beta)
        _, te_acc = ste_paper_style_objective(model, Xte, yte, beta)
        train_objs.append(tr_obj)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
    return model, train_objs, train_accs, test_accs


def sweep_ste(X_train, y_train_pm1, X_test, y_test_pm1, config: ExperimentConfig) -> STEHistory:
    best_acc = -float("inf")
    best_beta, best_lr = config.ste_beta_grid[0], config.ste_lr_grid[0]
    best_train_objs, best_train_accs, best_test_accs = [], [], []
    for beta in config.ste_beta_grid:
        for lr in config.ste_lr_grid:
            _, train_objs, train_accs, test_accs = train_single_ste_run(
                X_train, y_train_pm1, X_test, y_test_pm1,
                config.hidden_width, beta, lr, config.ste_epochs, config.ste_batch_size, config.gpu,
            )
            candidate = max(test_accs)
            if candidate > best_acc:
                best_acc = candidate
                best_beta, best_lr = beta, lr
                best_train_objs, best_train_accs, best_test_accs = train_objs, train_accs, test_accs
    return STEHistory(best_train_objs, best_train_accs, best_test_accs, best_acc, best_beta, best_lr, best_train_objs[-1])


# ------------------------ experiment loop ------------------------


def preprocess_for_training(X: np.ndarray, dataset_name: str, seed: int, test_size: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X[0], X[1], test_size=test_size, random_state=seed, stratify=X[1]
    )


def evaluate_best_hparams_over_runs(X_train, y_train_pm1, X_test, y_test_pm1, config: ExperimentConfig, best_ste_beta: float, best_ste_lr: float, best_cvx_beta: float, num_runs: int = 5):
    ste_train, ste_test, cvx_train, cvx_test = [], [], [], []
    for i in range(num_runs):
        seed = config.random_seed + i
        set_seed(seed)
        _, _, st_tr_hist, st_te_hist = train_single_ste_run(
            X_train, y_train_pm1, X_test, y_test_pm1,
            config.hidden_width, best_ste_beta, best_ste_lr, config.ste_epochs, config.ste_batch_size, config.gpu,
        )
        ste_train.append(float(st_tr_hist[-1]))
        ste_test.append(float(max(st_te_hist)))
        cvx_res = fit_convex_lasso(X_train, y_train_pm1, X_test, y_test_pm1, config.sampled_feature_count, best_cvx_beta, seed)
        cvx_train.append(cvx_res.train_accuracy)
        cvx_test.append(cvx_res.test_accuracy)
    return ste_train, ste_test, cvx_train, cvx_test


def run_experiment(config: ExperimentConfig) -> Dict[str, float]:
    config.gpu = choose_best_device()
    set_seed(config.random_seed)

    X, y = load_dataset(config)
    y_pm1 = ensure_binary_pm1(y)
    dataset_name = config.dataset_name.lower()
    raw_d = X.shape[1]
    if dataset_name == "bank" and raw_d != 16:
        raise ValueError(f"Bank raw d must be 16, got {raw_d}")
    if dataset_name == "ozone" and raw_d != 72:
        raise ValueError(f"Ozone raw d must be 72, got {raw_d}")

    X_train, X_test, y_train_pm1, y_test_pm1 = train_test_split(
        X, y_pm1, test_size=config.test_size, random_state=config.random_seed, stratify=y_pm1
    )

    if dataset_name != "cifar_cat_dog":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = append_bias_column(X_train).astype(np.float32)
    X_test = append_bias_column(X_test).astype(np.float32)

    print("=" * 80)
    print("Data summary")
    print(f"Dataset         : {config.dataset_name}")
    print(f"Raw feature dim : {raw_d}")
    print(f"Train shape     : {X_train.shape}")
    print(f"Test shape      : {X_test.shape}")
    print(f"Device          : {config.gpu}")
    print("=" * 80)

    cvx = sweep_cvx_beta(X_train, y_train_pm1, X_test, y_test_pm1, config)
    print("\n[Convex-LASSO]")
    print(f"train_acc         = {cvx.train_accuracy:.4f}")
    print(f"test_acc          = {cvx.test_accuracy:.4f}")
    print(f"fit_time_sec      = {cvx.fit_time_sec:.2f}")
    print(f"nonzero_count     = {cvx.nonzero_count}")
    print(f"sampled_features  = {cvx.sampled_feature_count}")
    print(f"beta_used         = {cvx.beta_used:.6e}")
    print(f"objective_value   = {cvx.objective_value:.6f}")

    ste = sweep_ste(X_train, y_train_pm1, X_test, y_test_pm1, config)
    print("\n[STE baseline: best sweep result]")
    print(f"best_beta         = {ste.best_beta}")
    print(f"best_lr           = {ste.best_lr}")
    print(f"best_test_acc     = {ste.best_test_accuracy:.4f}")
    print(f"final_train_obj   = {ste.train_objective_history[-1]:.6f}")
    print(f"final_train_acc   = {ste.train_accuracy_history[-1]:.4f}")
    print(f"final_test_acc    = {ste.test_accuracy_history[-1]:.4f}")
    print(f"objective_value   = {ste.final_objective_value:.6f}")

    ste_train_runs, ste_test_runs, cvx_train_runs, cvx_test_runs = evaluate_best_hparams_over_runs(
        X_train, y_train_pm1, X_test, y_test_pm1, config, ste.best_beta, ste.best_lr, cvx.beta_used, num_runs=5
    )
    print("\n[5-run summary]")
    print(f"STE train_acc     = {summarize_metric(ste_train_runs)}")
    print(f"STE test_acc      = {summarize_metric(ste_test_runs)}")
    print(f"CVX train_acc     = {summarize_metric(cvx_train_runs)}")
    print(f"CVX test_acc      = {summarize_metric(cvx_test_runs)}")

    return {
        "dataset": config.dataset_name,
        "cvx_train_acc": float(np.mean(cvx_train_runs)),
        "cvx_test_acc": float(np.mean(cvx_test_runs)),
        "ste_train_acc": float(np.mean(ste_train_runs)),
        "ste_test_acc": float(np.mean(ste_test_runs)),
    }


def run_all_datasets(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for dataset_name in config.dataset_list:
        print("\n" + "#" * 100)
        print(f"Running dataset: {dataset_name}")
        print("#" * 100)
        dataset_cfg = ExperimentConfig(**{**asdict(config), "dataset_name": dataset_name})
        rows.append(run_experiment(dataset_cfg))
    df = pd.DataFrame(rows, columns=["dataset", "cvx_train_acc", "cvx_test_acc", "ste_train_acc", "ste_test_acc"])
    print("\nFinal result table")
    print(df.to_string(index=False))
    Path(config.output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_csv_path, index=False)
    print(f"\nSaved CSV to: {config.output_csv_path}")
    return df


if __name__ == "__main__":
    cfg = ExperimentConfig(ste_epochs=500)
    run_all_datasets(cfg)
