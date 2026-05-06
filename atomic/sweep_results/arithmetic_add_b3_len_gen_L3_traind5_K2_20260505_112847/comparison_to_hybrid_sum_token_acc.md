# arithmetic_add_b3_len_gen_L3_traind5_K2_20260505_112847 vs hybrid carry bench (sum_token_acc)

- Length-gen run: `arithmetic_add_b3_len_gen_L3_traind5_K2_20260505_112847` (arith_base=3, L=3, K_parallel=2, n_digits_train=5, n_train=8192, loss_type=hinge_ovr, cvx_method=cvx, variants=full5, mask_carry_in=True, seeds=[0, 1, 2])

- Hybrid run:    `carry_hybrid_b3_d5_20260424_031252` (arith_base=3, L=3, K_parallel=2, n_digits=5, ste_last_layer_readout=spike, cvx_last_layer_readout=spike, lambda_carry_grid=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0], seeds=[0, 1, 2])

- Metric: **sum_token_acc** in the hybrid bench averages ``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where ``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the MSD carry-out at the final step. The matching length-gen quantity is its raw ``token_acc`` (single-head, same target sequence, same averaging) — both metrics are mathematically identical.

- Hybrid eval: **AR** (autoregressive rollout — model feeds its own predicted carry into the next step). Teacher-forcing (TF) hybrid numbers are not listed here.

- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid uses its own seed list (they are independent — see headers above).

## Length-gen token_acc (== hybrid sum_token_acc; full T positions)

| length-gen stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| len-gen STE pretrain | 0.4135 ± 0.0127 | 0.3615 ± 0.0121 | 0.3489 ± 0.0089 | - | 0.3451 ± 0.0126 |
| len-gen CVX from STE init | 0.4552 ± 0.0085 | 0.3850 ± 0.0081 | 0.3648 ± 0.0032 | - | 0.3538 ± 0.0103 |
| len-gen STE finetune (from STE) | 0.4193 ± 0.0116 | 0.3680 ± 0.0098 | 0.3493 ± 0.0024 | - | 0.3417 ± 0.0037 |
| len-gen CVX pretrain (Gaussian) | 0.5359 ± 0.0062 | 0.4490 ± 0.0025 | 0.3970 ± 0.0041 | - | 0.3640 ± 0.0034 |
| len-gen STE finetune (from CVX) | 0.3871 ± 0.0302 | 0.3630 ± 0.0111 | 0.3492 ± 0.0054 | - | 0.3436 ± 0.0022 |

## Hybrid sum_token_acc at λ_carry = 1.0 (AR eval)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | 0.4249 ± 0.0198 | 0.3864 ± 0.0092 | - | 0.3571 ± 0.0042 | 0.3485 ± 0.0005 |
| CVX from STE init | 0.4583 ± 0.0279 | 0.3913 ± 0.0071 | - | 0.3537 ± 0.0078 | 0.3448 ± 0.0017 |
| STE finetune (from STE, split B) | 0.4437 ± 0.0178 | 0.4077 ± 0.0157 | - | 0.3752 ± 0.0115 | 0.3692 ± 0.0100 |
| CVX pretrain (Gaussian) | 0.4209 ± 0.0117 | 0.3752 ± 0.0046 | - | 0.3468 ± 0.0057 | 0.3397 ± 0.0007 |
| STE finetune (from CVX, split B) | 0.4830 ± 0.0057 | 0.4200 ± 0.0014 | - | 0.3709 ± 0.0005 | 0.3534 ± 0.0014 |

## Hybrid sum_token_acc — best λ_carry per cell (AR eval; selection on max mean)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | 0.4384 ± 0.0069 (λ=1.25) | 0.3971 ± 0.0040 (λ=6) | - | 0.3705 ± 0.0173 (λ=2) | 0.3616 ± 0.0071 (λ=6) |
| CVX from STE init | 0.4628 ± 0.0285 (λ=4) | 0.3962 ± 0.0120 (λ=4) | - | 0.3629 ± 0.0071 (λ=4) | 0.3523 ± 0.0076 (λ=4) |
| STE finetune (from STE, split B) | 0.4723 ± 0.0075 (λ=8) | 0.4197 ± 0.0052 (λ=6) | - | 0.3882 ± 0.0087 (λ=6) | 0.3766 ± 0.0092 (λ=2) |
| CVX pretrain (Gaussian) | 0.4227 ± 0.0073 (λ=2) | 0.3768 ± 0.0067 (λ=2) | - | 0.3477 ± 0.0059 (λ=0.75) | 0.3402 ± 0.0016 (λ=10) |
| STE finetune (from CVX, split B) | 0.4921 ± 0.0141 (λ=1.5) | 0.4245 ± 0.0083 (λ=1.25) | - | 0.3737 ± 0.0053 (λ=1.25) | 0.3554 ± 0.0045 (λ=2) |

