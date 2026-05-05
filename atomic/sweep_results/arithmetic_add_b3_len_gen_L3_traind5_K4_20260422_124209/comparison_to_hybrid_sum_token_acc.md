# arithmetic_add_b3_len_gen_L3_traind5_K4_20260422_124209 vs hybrid carry bench (sum_token_acc)

- Length-gen run: `arithmetic_add_b3_len_gen_L3_traind5_K4_20260422_124209` (arith_base=3, L=3, K_parallel=4, n_digits_train=5, n_train=7308, loss_type=hinge_ovr, cvx_method=cvx, variants=minimal, mask_carry_in=False, seeds=[0, 1, 2])

- Hybrid run:    `carry_hybrid_b3_d5_20260424_031252` (arith_base=3, L=3, K_parallel=2, n_digits=5, ste_last_layer_readout=spike, cvx_last_layer_readout=spike, lambda_carry_grid=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0], seeds=[0, 1, 2])

- Metric: **sum_token_acc** in the hybrid bench averages ``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where ``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the MSD carry-out at the final step. The matching length-gen quantity is its raw ``token_acc`` (single-head, same target sequence, same averaging) — both metrics are mathematically identical.

- Hybrid eval modes: **TF** = teacher-forced (true carry-in input every step, matching the length-gen regime); **AR** = autoregressive rollout (model uses its own predicted carry as the next input).

- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid uses its own seed list (they are independent — see headers above).

## Length-gen token_acc (== hybrid sum_token_acc; full T positions)

| length-gen stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| len-gen STE pretrain | 0.4114 ± 0.0144 | 0.3590 ± 0.0096 | 0.3450 ± 0.0054 | - | 0.3386 ± 0.0009 |
| len-gen CVX pretrain (Gaussian) | 0.6759 ± 0.0058 | 0.5239 ± 0.0045 | 0.4363 ± 0.0048 | - | 0.3810 ± 0.0041 |

## Hybrid sum_token_acc at λ_carry = 1.0

| hybrid stage | mode | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | TF | 0.4922 ± 0.0162 | 0.4149 ± 0.0075 | - | 0.3648 ± 0.0049 | 0.3513 ± 0.0025 |
| STE pretrain (split A) | AR | 0.4249 ± 0.0198 | 0.3864 ± 0.0092 | - | 0.3571 ± 0.0042 | 0.3485 ± 0.0005 |
| CVX from STE init | TF | 0.4851 ± 0.0118 | 0.4105 ± 0.0059 | - | 0.3630 ± 0.0041 | 0.3495 ± 0.0007 |
| CVX from STE init | AR | 0.4583 ± 0.0279 | 0.3913 ± 0.0071 | - | 0.3537 ± 0.0078 | 0.3448 ± 0.0017 |
| STE finetune (from STE, split B) | TF | 0.5269 ± 0.0201 | 0.4425 ± 0.0107 | - | 0.3834 ± 0.0126 | 0.3637 ± 0.0120 |
| STE finetune (from STE, split B) | AR | 0.4437 ± 0.0178 | 0.4077 ± 0.0157 | - | 0.3752 ± 0.0115 | 0.3692 ± 0.0100 |
| CVX pretrain (Gaussian) | TF | 0.4343 ± 0.0052 | 0.3826 ± 0.0056 | - | 0.3541 ± 0.0068 | 0.3454 ± 0.0035 |
| CVX pretrain (Gaussian) | AR | 0.4209 ± 0.0117 | 0.3752 ± 0.0046 | - | 0.3468 ± 0.0057 | 0.3397 ± 0.0007 |
| STE finetune (from CVX, split B) | TF | 0.5524 ± 0.0072 | 0.4521 ± 0.0098 | - | 0.3816 ± 0.0048 | 0.3585 ± 0.0025 |
| STE finetune (from CVX, split B) | AR | 0.4830 ± 0.0057 | 0.4200 ± 0.0014 | - | 0.3709 ± 0.0005 | 0.3534 ± 0.0014 |

## Hybrid sum_token_acc — best λ_carry per cell (selection on max mean)

| hybrid stage | mode | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | TF | 0.4982 ± 0.0050 (λ=1.25) | 0.4251 ± 0.0091 (λ=4) | - | 0.3800 ± 0.0177 (λ=6) | 0.3677 ± 0.0180 (λ=6) |
| STE pretrain (split A) | AR | 0.4384 ± 0.0069 (λ=1.25) | 0.3971 ± 0.0040 (λ=6) | - | 0.3705 ± 0.0173 (λ=2) | 0.3616 ± 0.0071 (λ=6) |
| CVX from STE init | TF | 0.4851 ± 0.0118 (λ=1) | 0.4176 ± 0.0043 (λ=4) | - | 0.3698 ± 0.0053 (λ=4) | 0.3587 ± 0.0100 (λ=10) |
| CVX from STE init | AR | 0.4628 ± 0.0285 (λ=4) | 0.3962 ± 0.0120 (λ=4) | - | 0.3629 ± 0.0071 (λ=4) | 0.3523 ± 0.0076 (λ=4) |
| STE finetune (from STE, split B) | TF | 0.5479 ± 0.0173 (λ=2) | 0.4600 ± 0.0114 (λ=2) | - | 0.3973 ± 0.0111 (λ=8) | 0.3801 ± 0.0088 (λ=2) |
| STE finetune (from STE, split B) | AR | 0.4723 ± 0.0075 (λ=8) | 0.4197 ± 0.0052 (λ=6) | - | 0.3882 ± 0.0087 (λ=6) | 0.3766 ± 0.0092 (λ=2) |
| CVX pretrain (Gaussian) | TF | 0.4343 ± 0.0052 (λ=0.125) | 0.3826 ± 0.0056 (λ=0.125) | - | 0.3541 ± 0.0068 (λ=0.125) | 0.3454 ± 0.0035 (λ=0.125) |
| CVX pretrain (Gaussian) | AR | 0.4227 ± 0.0073 (λ=2) | 0.3768 ± 0.0067 (λ=2) | - | 0.3477 ± 0.0059 (λ=0.75) | 0.3402 ± 0.0016 (λ=10) |
| STE finetune (from CVX, split B) | TF | 0.5684 ± 0.0227 (λ=1.5) | 0.4604 ± 0.0141 (λ=1.5) | - | 0.3883 ± 0.0089 (λ=2) | 0.3643 ± 0.0057 (λ=2) |
| STE finetune (from CVX, split B) | AR | 0.4921 ± 0.0141 (λ=1.5) | 0.4245 ± 0.0083 (λ=1.25) | - | 0.3737 ± 0.0053 (λ=1.25) | 0.3554 ± 0.0045 (λ=2) |

