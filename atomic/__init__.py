"""Atomic, self-contained research modules for SNN/CVX experiments."""

from __future__ import annotations

from typing import Any

__all__ = ["LossFunction"]


def __getattr__(name: str) -> Any:
    if name == "LossFunction":
        from .solvers.loss_functions import LossFunction

        return LossFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
