"""Save / load frozen-weight checkpoints so CVX-family jobs can run on another machine.

Layout (one cell):
    <ckpt_dir>/seed{S}_T{T}_L{L}_K{K}_{tag}.npz          # w0, w1, ...
    <ckpt_dir>/seed{S}_T{T}_L{L}_K{K}_{tag}.meta.json    # architecture + knobs

``tag`` is ``sg`` for the STE/SG snapshot used by SG-CVX, or ``lsm`` for the
criticality-tuned reservoir used by R-CVX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def cell_stem(*, seed: int, T: int, L: int, K: int) -> str:
    return f"seed{int(seed)}_T{int(T)}_L{int(L)}_K{int(K)}"


def ckpt_path(ckpt_dir: Path, *, seed: int, T: int, L: int, K: int, tag: str) -> Path:
    return Path(ckpt_dir) / f"{cell_stem(seed=seed, T=T, L=L, K=K)}_{tag}.npz"


def save_weight_list(
    path: Path,
    weights: List[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    path = Path(path)
    if len(weights) == 0:
        raise ValueError(f"Refusing to write empty weight list to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)})
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[ckpt] wrote {path} ({len(weights)} tensors) + {meta_path.name}", flush=True)


def load_weight_list(path: Path) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CVX-side checkpoint missing: {path}. "
            "Run the non-CVX side first (--side non_cvx) and rsync that ckpt_dir onto this machine."
        )
    payload = np.load(path)
    keys = sorted(
        (k for k in payload.files if k.startswith("w") and k[1:].isdigit()),
        key=lambda s: int(s[1:]),
    )
    if len(keys) == 0:
        raise ValueError(f"No weight arrays (w0, w1, ...) in {path}. files={list(payload.files)}")
    weights = [np.asarray(payload[k]) for k in keys]
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta json for checkpoint: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise TypeError(f"Checkpoint meta must be a dict, got {type(meta)} from {meta_path}.")
    return weights, meta
