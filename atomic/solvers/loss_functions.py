from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    name: str
    value: torch.Tensor


class LossFunction:
    """Centralized loss definitions used across STE/CVX pipelines."""

    @staticmethod
    def ce(y: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        if f_x.ndim == 3:
            bsz, steps, num_classes = f_x.shape
            if y.ndim == 1:
                logits = f_x[:, -1, :]
                return F.cross_entropy(logits, y)
            if y.ndim == 2:
                return F.cross_entropy(f_x.reshape(bsz * steps, num_classes), y.reshape(bsz * steps))
            raise ValueError(f"CE expects y to have ndim 1 or 2 when f_x is 3D, got y.ndim={y.ndim}.")
        if f_x.ndim == 2:
            return F.cross_entropy(f_x, y)
        raise ValueError(f"CE expects f_x with ndim 2 or 3, got ndim={f_x.ndim}.")

    @staticmethod
    def hinge(y: torch.Tensor, f_x: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        if f_x.ndim > 1 and f_x.shape[-1] == 1:
            f_x = f_x.squeeze(-1)
        if y.ndim != f_x.ndim:
            raise ValueError(f"Hinge expects y and f_x with matching ndim, got {y.ndim} vs {f_x.ndim}.")
        if torch.equal(torch.unique(y), torch.tensor([0, 1], device=y.device, dtype=y.dtype)):
            y_pm1 = y * 2 - 1
        else:
            y_pm1 = y
        y_pm1 = y_pm1.to(dtype=f_x.dtype)
        return torch.relu(margin - y_pm1 * f_x).mean()

    @staticmethod
    def hinge_ovr(y: torch.Tensor, f_x: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        if f_x.ndim == 2:
            num_samples, num_classes = f_x.shape
            y_flat = y.reshape(num_samples)
            targets = -torch.ones_like(f_x)
            targets.scatter_(1, y_flat.unsqueeze(1), 1.0)
            return torch.relu(margin - targets * f_x).mean()
        if f_x.ndim == 3:
            bsz, steps, num_classes = f_x.shape
            if y.ndim == 1:
                scores = f_x[:, -1, :]
                y_flat = y
            elif y.ndim == 2:
                scores = f_x.reshape(bsz * steps, num_classes)
                y_flat = y.reshape(bsz * steps)
            else:
                raise ValueError(f"hinge_ovr expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
            targets = -torch.ones_like(scores)
            targets.scatter_(1, y_flat.unsqueeze(1), 1.0)
            return torch.relu(margin - targets * scores).mean()
        raise ValueError(f"hinge_ovr expects f_x with ndim 2 or 3, got {f_x.ndim}.")

    @staticmethod
    def squared(y: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        y_cast = y.to(dtype=f_x.dtype)
        if y_cast.shape != f_x.shape:
            raise ValueError(f"Squared loss expects y and f_x shapes to match exactly, got {y_cast.shape} vs {f_x.shape}.")
        return torch.mean((f_x - y_cast) ** 2)

    @classmethod
    def compute(cls, name: str, y: torch.Tensor, f_x: torch.Tensor) -> LossOutput:
        if name == "ce":
            return LossOutput(name=name, value=cls.ce(y=y, f_x=f_x))
        if name == "hinge":
            return LossOutput(name=name, value=cls.hinge(y=y, f_x=f_x))
        if name == "hinge_ovr":
            return LossOutput(name=name, value=cls.hinge_ovr(y=y, f_x=f_x))
        if name == "squared":
            return LossOutput(name=name, value=cls.squared(y=y, f_x=f_x))
        raise ValueError(f"Unknown loss function: {name}. Expected one of: ce, hinge, hinge_ovr, squared.")
