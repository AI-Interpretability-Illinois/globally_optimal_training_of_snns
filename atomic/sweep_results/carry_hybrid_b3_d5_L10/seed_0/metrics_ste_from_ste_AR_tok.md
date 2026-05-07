# Carry hybrid b3 d5 L10 — AR joint token accuracies

- **Source:** `metrics_ste_from_ste.json` (carry hybrid b3 d5 L10)
- **finetune_variant:** `ste_from_ste`

| λ_carry | phase | AR joint (ID) | AR joint OOD d=10 | AR joint OOD d=25 | AR joint OOD d=50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.125 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.125 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.25 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.25 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.5 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.5 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.75 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 0.75 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1.25 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1.25 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1.5 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 1.5 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 2 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 2 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 4 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 4 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 6 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 6 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 8 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 8 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 10 | pretrain_eval (loaded init) | 0.2389 | 0.2523 | 0.2679 | 0.2690 |
| 10 | ste_finetune | 0.2389 | 0.2523 | 0.2679 | 0.2690 |

*Mean over (sequence, timestep) in each split; OOD digit counts in `ood_eval`.*
