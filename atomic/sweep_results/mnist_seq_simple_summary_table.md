# MNIST-Seq Simple Run Summary (L=3, T=28)

| Hyperparameter | Value |
|---|---:|
| P_rec | 8000 |
| P_last | 7000 |
| n_train | 6000 |
| n_val | 2000 |
| n_test | 5000 |

| Model | Selected lr | Selected beta | Selected bias | Train loss | Val loss | Test loss | Test acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| STE | 0.001 | 0.01 | - | 0.672277 | 0.807158 | 0.809327 | 0.7630 |
| CVX (gaussian, sgd) | 0.005 | 0.01 | -1.0 | 0.258956 | 0.528499 | 0.551863 | 0.8308 |
