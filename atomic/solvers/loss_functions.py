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
                # Non-arithmetic sequence tasks: supervise last timestep only.
                logits = f_x[:, -1, :]
                return F.cross_entropy(logits, y)
            if y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
                return F.cross_entropy(f_x.reshape(bsz * steps, num_classes), y.reshape(bsz * steps))
            raise ValueError(f"CE expects y to have ndim 1 or 2 when f_x is 3D, got y.ndim={y.ndim}.")
        if f_x.ndim == 2:
            return F.cross_entropy(f_x, y)
        raise ValueError(f"CE expects f_x with ndim 2 or 3, got ndim={f_x.ndim}.")

    @staticmethod
    def hinge(y: torch.Tensor, f_x: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        # snn_p2-style binary hinge:
        # - single-label sequence classification (y.ndim==1): use last timestep logits
        # - sequence labels (y.ndim==2): supervise all timesteps
        def _binary_margin_from_logits(z: torch.Tensor) -> torch.Tensor:
            # Accept either a single binary logit channel or 2-class logits.
            if z.ndim >= 1 and z.shape[-1] == 1:
                return z.squeeze(-1)
            if z.ndim >= 1 and z.shape[-1] == 2:
                return z[..., 1] - z[..., 0]
            raise ValueError(f"Binary hinge expects final channel size 1 or 2, got shape={z.shape}.")

        if f_x.ndim == 3:
            margins = _binary_margin_from_logits(f_x)
            if y.ndim == 1:
                # Non-arithmetic sequence tasks: supervise last timestep only.
                f_x = margins[:, -1]
            elif y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
                bsz, steps = margins.shape
                f_x = margins.reshape(bsz * steps)
                y = y.reshape(bsz * steps)
            else:
                raise ValueError(f"Hinge expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
        elif f_x.ndim == 2:
            f_x = _binary_margin_from_logits(f_x)
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
                # Non-arithmetic sequence tasks: supervise last timestep only.
                scores = f_x[:, -1, :]
                y_flat = y
            elif y.ndim == 2:
                # Arithmetic-style token supervision: supervise all timesteps.
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
        # Keep the same sequence supervision convention as other losses:
        # y.ndim==1 -> last timestep only, y.ndim==2 -> all timesteps.
        if f_x.ndim == 3:
            bsz, steps, num_outputs = f_x.shape
            if y.ndim == 1:
                preds = f_x[:, -1, :]
                if num_outputs == 1:
                    target = y.to(dtype=preds.dtype).reshape(-1, 1)
                    return torch.mean((preds - target) ** 2)
                if num_outputs > 1:
                    target = F.one_hot(y.to(torch.long), num_classes=num_outputs).to(dtype=preds.dtype)
                    return torch.mean((preds - target) ** 2)
                raise ValueError(f"Invalid num_outputs={num_outputs} for squared loss.")
            if y.ndim == 2:
                preds = f_x.reshape(bsz * steps, num_outputs)
                y_flat = y.reshape(bsz * steps)
                if num_outputs == 1:
                    target = y_flat.to(dtype=preds.dtype).reshape(-1, 1)
                    return torch.mean((preds - target) ** 2)
                target = F.one_hot(y_flat.to(torch.long), num_classes=num_outputs).to(dtype=preds.dtype)
                return torch.mean((preds - target) ** 2)
            raise ValueError(f"Squared expects y with ndim 1 or 2 when f_x is 3D, got {y.ndim}.")
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
