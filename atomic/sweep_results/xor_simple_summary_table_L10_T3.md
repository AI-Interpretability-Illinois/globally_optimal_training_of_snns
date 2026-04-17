# XOR Simple Run Summary (L=10, T=3)

| Hyperparameter | Value |
|---|---:|
| P_rec | 8000 |
| P_last | 7000 |
| n_train | 6000 |
| n_test | 5000 |

| Model | Selected lr | Selected beta | Selected bias | Train loss | Val loss | Test loss | Test acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| STE | 0.001 | 0.01 | - | 0.762867 | 0.775296 | 0.756482 | 0.6314 |
| CVX (gaussian, sgd) | 0.01 | 0.01 | 1 | 0.098912 | 0.104541 | 0.122247 | 0.9582 |
