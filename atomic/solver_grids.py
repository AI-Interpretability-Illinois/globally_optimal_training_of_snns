from __future__ import annotations

from typing import Final

# Shared search grids used across atomic fine-tune, layer-wise, and bench entry points.
BETA_GRID_DEFAULT: Final[tuple[float, ...]] = (1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0)
LR_GRID_DEFAULT: Final[tuple[float, ...]] = (1e-3, 5e-3, 1e-2, 1e-1)
BIAS_GRID_DEFAULT: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0)
