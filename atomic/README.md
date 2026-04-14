# Atomic package

This folder is intentionally self-contained:

- Modules in `atomic/` must not import project code from sibling folders.
- No `sys.path` patching or dynamic file-path imports from outside `atomic/`.
- Shared code inside this package should be imported using relative imports.

Run the guard check:

```bash
python3 -m atomic.self_check
```

If the check fails, it reports which file introduced outside-folder coupling.

Run sweeps from the repository root (`experiments/`) using module mode:

```bash
# Noisy benchmark sweep (defaults: dataset=mnist_seq, pipeline_mode=both, cvx_method=cvx)
python3 -m atomic.noisy_test_bench

# Noisy sweep with explicit options
python3 -m atomic.noisy_test_bench \
  --dataset mnist_seq \
  --pipeline_mode both \
  --cvx_method cvx \
  --csv_path noisy_data_results.csv

# Deterministic benchmark sweep (defaults: tasks=addition xor parity, pipeline_mode=both, init_mode=both, cvx_method=cvx)
python3 -m atomic.deterministic_test_bench

# Deterministic sweep with explicit options
python3 -m atomic.deterministic_test_bench \
  --tasks addition xor parity \
  --pipeline_mode both \
  --init_mode both \
  --cvx_method cvx \
  --seed 0 \
  --csv_path deterministic_data_results.csv
```
