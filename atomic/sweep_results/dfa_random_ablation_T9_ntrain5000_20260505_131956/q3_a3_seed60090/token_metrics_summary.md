# DFA token metrics (`last_step_acc`) — `q3_a3_seed60090`

- **dfa_spec**: `random_3_3_seed60090` | **T_train**: 9 | **n_test** (ID): 1024 | **n_test** (OOD): 1024

- **Stages**: ste_pretrain, cvx_from_ste_pretrain, ste_finetune_from_ste_pretrain_new_train, cvx_pretrain_gaussian, ste_finetune_from_cvx_pretrain_new_train

- **Metric**: `last_step_acc` = fraction of sequences with correct **last-step** prediction (ID test split and full-length OOD sequences).

## `ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7734 | 0.6045 | 0.6172 | 0.6475 |
| 1 | 0.7578 | 0.6660 | 0.6416 | 0.6211 |
| 2 | 0.7988 | 0.6143 | 0.6250 | 0.6094 |
| **mean ± std** | 0.7767 ± 0.0207 | 0.6283 ± 0.0331 | 0.6279 ± 0.0125 | 0.6260 ± 0.0195 |

## `cvx_from_ste_pretrain`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7686 | 0.5977 | 0.6270 | 0.6162 |
| 1 | 0.7490 | 0.6553 | 0.6611 | 0.6562 |
| 2 | 0.7773 | 0.6123 | 0.6455 | 0.6025 |
| **mean ± std** | 0.7650 ± 0.0145 | 0.6217 ± 0.0299 | 0.6445 ± 0.0171 | 0.6250 ± 0.0279 |

## `ste_finetune_from_ste_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7891 | 0.6035 | 0.6182 | 0.6572 |
| 1 | 0.7588 | 0.6641 | 0.6338 | 0.6426 |
| 2 | 0.7988 | 0.6074 | 0.6387 | 0.5947 |
| **mean ± std** | 0.7822 ± 0.0209 | 0.6250 ± 0.0339 | 0.6302 ± 0.0107 | 0.6315 ± 0.0327 |

## `cvx_pretrain_gaussian`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.6406 | 0.6035 | 0.5742 | 0.5664 |
| 1 | 0.6211 | 0.5898 | 0.5518 | 0.5322 |
| 2 | 0.6016 | 0.6006 | 0.5889 | 0.5576 |
| **mean ± std** | 0.6211 ± 0.0195 | 0.5980 ± 0.0072 | 0.5716 ± 0.0187 | 0.5521 ± 0.0177 |

## `ste_finetune_from_cvx_pretrain_new_train`

| seed | ID last_step_acc | OOD T=18 (train T=9, ×2) | OOD T=45 (train T=9, ×5) | OOD T=90 (train T=9, ×10) |
|---:|---:|---:|---:|---:|
| 0 | 0.7744 | 0.6357 | 0.6328 | 0.6436 |
| 1 | 0.7510 | 0.6309 | 0.6182 | 0.5908 |
| 2 | 0.7803 | 0.6660 | 0.6846 | 0.6807 |
| **mean ± std** | 0.7686 ± 0.0155 | 0.6442 ± 0.0190 | 0.6452 ± 0.0349 | 0.6383 ± 0.0451 |

