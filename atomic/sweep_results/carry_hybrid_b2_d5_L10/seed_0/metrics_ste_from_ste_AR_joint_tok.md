# Carry hybrid b2 d5 L10 — AR joint token accuracy

- **Source:** `metrics_ste_from_ste.json` (carry hybrid b2 d5 L10)
- **finetune_variant:** `ste_from_ste`

| λ_carry | phase | AR joint (ID) | AR joint OOD d=10 | AR joint OOD d=25 | AR joint OOD d=50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.125 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.125 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.25 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.25 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.5 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.5 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.75 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 0.75 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1.25 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1.25 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1.5 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1.5 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 2 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 2 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 4 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 4 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 6 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 6 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 8 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 8 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 10 | pretrain_eval (loaded init) | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 10 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |

*ID / OOD: mean joint token accuracy over (sequence, timestep) in evaluated splits; OOD digit counts from `ood_eval`.*
