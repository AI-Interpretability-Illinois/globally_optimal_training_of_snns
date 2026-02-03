import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from snn_experiments import (
    a_snn_from_blocks,
    generate_moving_gaussian_blobs,
    set_seed,
)


@dataclass
class TenLayerConfig:
    seed: int
    layers: int
    n: int
    d: int
    num_samples: int
    debug: bool


def build_config(args: argparse.Namespace) -> TenLayerConfig:
    if args.debug:
        return TenLayerConfig(
            seed=args.seed,
            layers=10,
            n=20,
            d=5,
            num_samples=80,
            debug=True,
        )
    return TenLayerConfig(
        seed=args.seed,
        layers=10,
        n=100,
        d=20,
        num_samples=1000,
        debug=False,
    )


def build_timestep2_arrangements(
    x1: np.ndarray,
    x2: np.ndarray,
    layers: int,
    seed: int,
    num_samples: int,
    sampled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n = x1.shape[0]
    ones = np.ones((n, 1), dtype=np.float32)
    zero_block = np.zeros((n, 1), dtype=np.float32)

    d1_1 = a_snn_from_blocks(x1, zero_block, "+", ones, sampled, num_samples, seed + 1)
    d1_2 = a_snn_from_blocks(x2, d1_1, "-", ones, sampled, num_samples, seed + 2)

    d_prev_1 = d1_1
    d_prev_2 = d1_2
    for layer in range(2, layers + 1):
        d_curr_1 = a_snn_from_blocks(d_prev_1, zero_block, "+", ones, sampled, num_samples, seed + 10 * layer)
        d_curr_2 = a_snn_from_blocks(d_prev_2, d_curr_1, "-", ones, sampled, num_samples, seed + 10 * layer + 1)
        d_prev_1 = d_curr_1
        d_prev_2 = d_curr_2
    return d_prev_1, d_prev_2


def main() -> None:
    parser = argparse.ArgumentParser(description="D_{10,2} arrangement size (sampled vs full).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sampled-only", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)

    x1, x2, _ = generate_moving_gaussian_blobs(config.n, config.seed + 1)

    d10_1_sampled, d10_2_sampled = build_timestep2_arrangements(
        x1,
        x2,
        config.layers,
        config.seed,
        config.num_samples,
        sampled=True,
    )
    print(f"[sampled] D_({config.layers},2) shape = {d10_2_sampled.shape}")

    if args.sampled_only:
        return

    d10_1_full, d10_2_full = build_timestep2_arrangements(
        x1,
        x2,
        config.layers,
        config.seed,
        config.num_samples,
        sampled=False,
    )
    print(f"[full] D_({config.layers},2) shape = {d10_2_full.shape}")


if __name__ == "__main__":
    main()
