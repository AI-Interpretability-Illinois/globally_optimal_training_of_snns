import numpy as np
import torch
from torch.utils.data import Dataset
from snntorch import datasets


def bin_timesteps(spk: torch.Tensor, target_steps: int) -> torch.Tensor:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if spk.dim() < 2:
        raise ValueError("spk tensor must include a time dimension.")
    t = spk.shape[0]
    if target_steps == t:
        return spk
    if target_steps > t:
        reps = int(np.ceil(target_steps / t))
        return spk.repeat_interleave(reps, dim=0)[:target_steps]
    bin_size = int(np.ceil(t / target_steps))
    bins = []
    for idx in range(target_steps):
        start = idx * bin_size
        end = min((idx + 1) * bin_size, t)
        if start >= t:
            break
        chunk = spk[start:end]
        bins.append((chunk.sum(dim=0) > 0).float())
    return torch.stack(bins, dim=0)


def ensure_time_first(spk: torch.Tensor) -> torch.Tensor:
    if spk.dim() < 2:
        raise ValueError("spk tensor must be at least 2D.")
    if spk.shape[0] <= 8 and spk.shape[1] > 8:
        return spk
    return spk.permute(1, 0, *range(2, spk.dim()))


def load_dataset(name: str, root: str, train: bool) -> Dataset:
    name = name.lower()
    if name == "nmnist":
        return datasets.NMNIST(root=root, train=train, download=True)
    if name == "shd":
        return datasets.SHD(root=root, train=train, download=True)
    if name == "ssc":
        return datasets.SSC(root=root, train=train, download=True)
    if name == "dvsgesture":
        return datasets.DVSGesture(root=root, train=train, download=True)
    raise ValueError(f"Unsupported dataset '{name}'.")
