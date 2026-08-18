from __future__ import annotations

from typing import Final, Sequence

# Shared search grids used across atomic fine-tune, layer-wise, and bench entry points.
BETA_GRID_DEFAULT: Final[tuple[float, ...]] = (1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0)
LR_GRID_DEFAULT: Final[tuple[float, ...]] = (1e-3, 5e-3, 1e-2, 1e-1)
BIAS_GRID_DEFAULT: Final[tuple[float, ...]] = (0.0,)


def cvx_lr_sweep_values(cvx_method: str, lr_grid: Sequence[float]) -> tuple[float, ...]:
    """
    For SolveConfig.method == \"cvx\", the convex solver ignores learning rate; sweep beta × bias only.
    For method == \"sgd\", include the full lr grid (beta × lr × bias).
    """
    if cvx_method in ("cvx", "cvx_lite"):
        return (0.0,)
    return tuple(float(x) for x in lr_grid)
