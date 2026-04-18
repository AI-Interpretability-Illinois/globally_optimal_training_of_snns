# Parallelization Helper

This folder provides a cluster-friendly launcher that submits all top-level `run_*.py` sweep scripts with side control.

## Resource Mapping (from current partition snapshot)

- **CVX side (`--side cvx_only`)**
  - Prefer CPU partitions: `cpu-preempt` or `cpu`.
  - Immediately idle CPU node in your snapshot: `cn137`.
  - Helper defaults:
    - `--cpu_partition cpu-preempt`
    - `--cpu_nodelist cn137`

- **SNN side (`--side ste_only`)**
  - Prefer GPU partitions: `gpuA100x4-interactive` (A100) or `gpuA40x4-interactive` (A40).
  - Immediately idle GPU nodes in your snapshot: `gpua001`, `gpub002`.
  - Helper defaults:
    - `--gpu_partition gpuA100x4-interactive`
    - `--gpu_nodelist gpua001`
    - `--gpu_gres gpu:1`

You can override all defaults from CLI.

## Script

- `parallelization_helper.py`

It auto-discovers top-level run scripts in `atomic/`:
- `run_*.py`

And launches each with the chosen side mode.

## Examples

From `atomic/`:

```bash
python3 parallelization/parallelization_helper.py --mode both --execution slurm
```

CVX only (CPU), dry-run:

```bash
python3 parallelization/parallelization_helper.py \
  --mode cvx_only \
  --execution slurm \
  --dry_run
```

SNN only on A40 idle node:

```bash
python3 parallelization/parallelization_helper.py \
  --mode ste_only \
  --execution slurm \
  --gpu_partition gpuA40x4-interactive \
  --gpu_nodelist gpub002
```

Run locally with limited parallelism:

```bash
python3 parallelization/parallelization_helper.py \
  --mode both \
  --execution local \
  --max_local_workers 2
```
