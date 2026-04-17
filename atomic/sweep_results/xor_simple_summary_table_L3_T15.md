# XOR Simple Run Summary (L=3, T=15)

| Hyperparameter | Value |
|---|---:|
| P_rec | 8000 |
| P_last | 7000 |
| n_train | 6000 |
| n_val | 2000 |
| n_test | 5000 |

| Model | Source | Selected lr | Selected beta | Selected bias | Train loss | Val loss | Test loss | Test acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STE | `atomic/sweep_results/temp_text_v2.txt` | 0.001 | 0.01 | - | 0.848776 | 0.888625 | 0.897865 | 0.5802 |
| CVX (gaussian, sgd) | `terminals/37.txt` | 0.005 | 0.01 | 0 | 0.470859 | 0.537249 | 0.545406 | 0.7800 |
