# DFA token metrics (`last_step_acc`) — `q3_a2_seed50081`

- **dfa_spec**: `random_3_2_seed50081` | **T_train**: 9 | **n_test** (ID): 1024 | **n_test** (OOD): 1024

- **Stages**: ste_pretrain, cvx_from_ste_pretrain, ste_finetune_from_ste_pretrain_new_train, cvx_pretrain_gaussian, ste_finetune_from_cvx_pretrain_new_train

- **Metric**: `last_step_acc` = fraction of sequences with correct **last-step** prediction (ID test split and full-length OOD sequences).

## `ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7539 | 0.5059 | 0.5107 | 0.5283 |
| 1 | 0.7979 | 0.5371 | 0.6191 | 0.5186 |
| 2 | 0.7969 | 0.5146 | 0.5547 | 0.5684 |
| **mean ± std** | 0.7829 ± 0.0251 | 0.5192 ± 0.0161 | 0.5615 ± 0.0545 | 0.5384 ± 0.0264 |

## `cvx_from_ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.8506 | 0.5537 | 0.5088 | 0.4941 |
| 1 | 0.8398 | 0.5361 | 0.5264 | 0.5020 |
| 2 | 0.8223 | 0.5615 | 0.5449 | 0.5479 |
| **mean ± std** | 0.8376 ± 0.0143 | 0.5505 ± 0.0130 | 0.5267 ± 0.0181 | 0.5146 ± 0.0290 |

## `ste_finetune_from_ste_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.8594 | 0.5469 | 0.4980 | 0.5156 |
| 1 | 0.8389 | 0.5518 | 0.5420 | 0.5107 |
| 2 | 0.8164 | 0.5410 | 0.5498 | 0.5234 |
| **mean ± std** | 0.8382 ± 0.0215 | 0.5465 ± 0.0054 | 0.5299 ± 0.0279 | 0.5166 ± 0.0064 |

## `cvx_pretrain_gaussian`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7773 | 0.5537 | 0.5439 | 0.5137 |
| 1 | 0.7676 | 0.5225 | 0.4980 | 0.5010 |
| 2 | 0.7090 | 0.5049 | 0.5117 | 0.4912 |
| **mean ± std** | 0.7513 ± 0.0370 | 0.5270 ± 0.0247 | 0.5179 ± 0.0236 | 0.5020 ± 0.0113 |

## `ste_finetune_from_cvx_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7676 | 0.5918 | 0.5254 | 0.5195 |
| 1 | 0.7686 | 0.5752 | 0.5576 | 0.5811 |
| 2 | 0.7227 | 0.5752 | 0.4902 | 0.5039 |
| **mean ± std** | 0.7529 ± 0.0262 | 0.5807 ± 0.0096 | 0.5244 ± 0.0337 | 0.5348 ± 0.0408 |

