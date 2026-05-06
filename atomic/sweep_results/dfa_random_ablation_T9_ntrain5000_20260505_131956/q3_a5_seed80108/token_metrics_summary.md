# DFA token metrics (`last_step_acc`) — `q3_a5_seed80108`

- **dfa_spec**: `random_3_5_seed80108` | **T_train**: 9 | **n_test** (ID): 1024 | **n_test** (OOD): 1024

- **Stages**: ste_pretrain, cvx_from_ste_pretrain, ste_finetune_from_ste_pretrain_new_train, cvx_pretrain_gaussian, ste_finetune_from_cvx_pretrain_new_train

- **Metric**: `last_step_acc` = fraction of sequences with correct **last-step** prediction (ID test split and full-length OOD sequences).

## `ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.6846 | 0.6562 | 0.6719 | 0.6592 |
| 1 | 0.6416 | 0.6221 | 0.6406 | 0.6279 |
| 2 | 0.6553 | 0.6514 | 0.6299 | 0.6553 |
| **mean ± std** | 0.6605 ± 0.0220 | 0.6432 ± 0.0185 | 0.6475 ± 0.0218 | 0.6475 ± 0.0170 |

## `cvx_from_ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.6572 | 0.6494 | 0.6357 | 0.6309 |
| 1 | 0.6250 | 0.6289 | 0.6494 | 0.5918 |
| 2 | 0.6396 | 0.6484 | 0.6611 | 0.6484 |
| **mean ± std** | 0.6406 ± 0.0161 | 0.6423 ± 0.0116 | 0.6488 ± 0.0127 | 0.6237 ± 0.0290 |

## `ste_finetune_from_ste_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.6787 | 0.6553 | 0.6318 | 0.6465 |
| 1 | 0.6562 | 0.6240 | 0.6221 | 0.6289 |
| 2 | 0.6602 | 0.6426 | 0.6406 | 0.6367 |
| **mean ± std** | 0.6650 ± 0.0120 | 0.6406 ± 0.0157 | 0.6315 ± 0.0093 | 0.6374 ± 0.0088 |

## `cvx_pretrain_gaussian`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.5420 | 0.5459 | 0.5566 | 0.5586 |
| 1 | 0.5439 | 0.5039 | 0.5146 | 0.5195 |
| 2 | 0.5693 | 0.5420 | 0.5352 | 0.5459 |
| **mean ± std** | 0.5518 ± 0.0153 | 0.5306 ± 0.0232 | 0.5355 ± 0.0210 | 0.5413 ± 0.0199 |

## `ste_finetune_from_cvx_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.6875 | 0.6650 | 0.6494 | 0.6396 |
| 1 | 0.6426 | 0.6328 | 0.6279 | 0.5918 |
| 2 | 0.5879 | 0.5781 | 0.6035 | 0.6055 |
| **mean ± std** | 0.6393 ± 0.0499 | 0.6253 ± 0.0439 | 0.6270 ± 0.0230 | 0.6123 ± 0.0246 |

