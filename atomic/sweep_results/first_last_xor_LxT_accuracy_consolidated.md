# first_last_xor — consolidated test accuracy (.md summaries + L=3,T=11 JSON)

Grid: **L** (readout/hidden width) ∈ [3, 5, 10, 15], **T** (sequence length) ∈ [3, 6, 8, 11, 14].
Each cell is **mean ± sample std** over all `.md` runs in `sweep_results/first_last_xor_{L}_{T}/` when present; `(n=…)` is the number of runs.
**L=3, T=11:** no `.md` in this repo — values from `*.json` using `test_last_step_acc`: STE = mean ± std over all `ste_only` files; **CVX = best** accuracy among all `cvx_only` files (`metric_aggregate.json` skipped).

**Note:** No `first_last_xor_*_3` dirs and no L=3,T=8 `.md` here — those cells stay empty.

## STE — test accuracy

| L \ T | 3 | 6 | 8 | 11 | 14 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 3 | — | 0.7462 ± 0.0000 (n=1) | — | 0.6223 ± 0.0523 (n=4) | 0.6405 ± 0.0124 (n=3) |
| 5 | — | 0.7435 ± 0.0388 (n=4) | 0.6868 ± 0.0145 (n=3) | 0.6048 ± 0.0200 (n=3) | 0.5188 ± 0.0194 (n=3) |
| 10 | — | 0.5008 ± 0.0025 (n=2) | 0.4963 ± 0.0000 (n=1) | 0.4990 ± 0.0000 (n=1) | 0.5035 ± 0.0000 (n=1) |
| 15 | — | 0.5037 ± 0.0024 (n=3) | 0.4990 ± 0.0032 (n=3) | 0.5006 ± 0.0021 (n=3) | 0.4970 ± 0.0028 (n=3) |

## CVX — test accuracy

| L \ T | 3 | 6 | 8 | 11 | 14 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 3 | — | 1.0000 ± 0.0000 (n=1) | — | 0.8643 +- 0.0000 | 0.6890 ± 0.0213 (n=3) |
| 5 | — | 1.0000 ± 0.0000 (n=4) | 1.0000 ± 0.0000 (n=3) | 0.5010 ± 0.0008 (n=3) | 0.5007 ± 0.0045 (n=3) |
| 10 | — | 0.9732 ± 0.0378 (n=2) | 0.8578 ± 0.0000 (n=1) | 0.5010 ± 0.0000 (n=1) | 0.5035 ± 0.0000 (n=1) |
| 15 | — | 0.8410 ± 0.0584 (n=3) | 0.6835 ± 0.0515 (n=3) | 0.5021 ± 0.0030 (n=3) | 0.5064 ± 0.0069 (n=3) |

## Per-layer aggregate over T ∈ {3, 6, 8, 11, 14}

Pooled **all** run-level accuracies that feed each cell (for L=3,T=11 CVX, only the **single best** CVX value). Mean ± sample std over the pooled lists.

| L | STE | CVX | pooled n (STE / CVX) |
|---:|---:|---:|---:|
| 3 | 0.6446 ± 0.0546 (n=8) | 0.7863 ± 0.1423 (n=5) | 8 / 5 |
| 5 | 0.6465 ± 0.0927 (n=13) | 0.7696 ± 0.2590 (n=13) | 13 / 13 |
| 10 | 0.5001 ± 0.0029 (n=5) | 0.7618 ± 0.2423 (n=5) | 5 / 5 |
| 15 | 0.5001 ± 0.0034 (n=12) | 0.6333 ± 0.1505 (n=12) | 12 / 12 |
