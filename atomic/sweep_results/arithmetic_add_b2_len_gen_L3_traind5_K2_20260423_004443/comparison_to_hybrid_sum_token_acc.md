# arithmetic_add_b2_len_gen_L3_traind5_K2_20260423_004443 vs hybrid carry bench (sum_token_acc)

- Length-gen run: `arithmetic_add_b2_len_gen_L3_traind5_K2_20260423_004443` (arith_base=2, L=3, K_parallel=2, n_digits_train=5, n_train=8192, loss_type=hinge_ovr, cvx_method=cvx, variants=minimal, mask_carry_in=False, seeds=[0, 1, 2])

- Hybrid run:    `carry_hybrid_b2_d5_20260423_022747` (arith_base=2, L=3, K_parallel=2, n_digits=5, ste_last_layer_readout=spike, cvx_last_layer_readout=spike, lambda_carry_grid=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0], seeds=[0, 1, 2])

- Metric: **sum_token_acc** in the hybrid bench averages ``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where ``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the MSD carry-out at the final step. The matching length-gen quantity is its raw ``token_acc`` (single-head, same target sequence, same averaging) — both metrics are mathematically identical.

- Hybrid eval modes: **TF** = teacher-forced (true carry-in input every step, matching the length-gen regime); **AR** = autoregressive rollout (model uses its own predicted carry as the next input).

- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid uses its own seed list (they are independent — see headers above).

## Length-gen token_acc (== hybrid sum_token_acc; full T positions)

| length-gen stage | n=10 | n=15 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| len-gen STE pretrain | 0.5874 ± 0.0117 | 0.5970 ± 0.0046 | - | 0.6003 ± 0.0097 | - |
| len-gen CVX pretrain (Gaussian) | 0.7402 ± 0.0038 | 0.6815 ± 0.0037 | - | 0.6237 ± 0.0045 | - |

## Hybrid sum_token_acc at λ_carry = 1.0

| hybrid stage | mode | n=10 | n=15 | n=20 | n=25 | n=50 |
|:---|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | TF | 0.6723 ± 0.0225 | - | 0.6530 ± 0.0247 | - | 0.6381 ± 0.0219 |
| STE pretrain (split A) | AR | 0.5767 ± 0.0336 | - | 0.5769 ± 0.0296 | - | 0.5708 ± 0.0258 |
| CVX from STE init | TF | 0.6923 ± 0.0196 | - | 0.6716 ± 0.0188 | - | 0.6520 ± 0.0179 |
| CVX from STE init | AR | 0.5928 ± 0.0140 | - | 0.5872 ± 0.0193 | - | 0.5791 ± 0.0243 |
| STE finetune (from STE, split B) | TF | 0.7441 ± 0.0318 | - | 0.7174 ± 0.0340 | - | 0.6989 ± 0.0324 |
| STE finetune (from STE, split B) | AR | 0.6481 ± 0.0597 | - | 0.6310 ± 0.0529 | - | 0.6202 ± 0.0460 |
| CVX pretrain (Gaussian) | TF | 0.6599 ± 0.0075 | - | 0.5955 ± 0.0104 | - | 0.5532 ± 0.0074 |
| CVX pretrain (Gaussian) | AR | 0.6265 ± 0.0008 | - | 0.5709 ± 0.0071 | - | 0.5380 ± 0.0048 |
| STE finetune (from CVX, split B) | TF | 0.7843 ± 0.0027 | - | 0.7046 ± 0.0048 | - | 0.6445 ± 0.0065 |
| STE finetune (from CVX, split B) | AR | 0.7452 ± 0.0023 | - | 0.6641 ± 0.0109 | - | 0.6011 ± 0.0015 |

## Hybrid sum_token_acc — best λ_carry per cell (selection on max mean)

| hybrid stage | mode | n=10 | n=15 | n=20 | n=25 | n=50 |
|:---|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | TF | 0.6867 ± 0.0048 (λ=0.5) | - | 0.6641 ± 0.0228 (λ=0.75) | - | 0.6491 ± 0.0195 (λ=0.75) |
| STE pretrain (split A) | AR | 0.6066 ± 0.0188 (λ=0.75) | - | 0.5964 ± 0.0186 (λ=0.75) | - | 0.5871 ± 0.0194 (λ=0.75) |
| CVX from STE init | TF | 0.6935 ± 0.0060 (λ=0.75) | - | 0.6750 ± 0.0186 (λ=0.75) | - | 0.6522 ± 0.0221 (λ=0.75) |
| CVX from STE init | AR | 0.6036 ± 0.0442 (λ=1.5) | - | 0.5884 ± 0.0238 (λ=0.75) | - | 0.5818 ± 0.0264 (λ=0.75) |
| STE finetune (from STE, split B) | TF | 0.7441 ± 0.0318 (λ=1) | - | 0.7174 ± 0.0340 (λ=1) | - | 0.6989 ± 0.0324 (λ=1) |
| STE finetune (from STE, split B) | AR | 0.6710 ± 0.0303 (λ=1.5) | - | 0.6420 ± 0.0533 (λ=1.25) | - | 0.6295 ± 0.0430 (λ=1.25) |
| CVX pretrain (Gaussian) | TF | 0.6599 ± 0.0075 (λ=0.125) | - | 0.5955 ± 0.0104 (λ=0.125) | - | 0.5532 ± 0.0074 (λ=0.125) |
| CVX pretrain (Gaussian) | AR | 0.6295 ± 0.0037 (λ=10) | - | 0.5724 ± 0.0065 (λ=2) | - | 0.5383 ± 0.0065 (λ=10) |
| STE finetune (from CVX, split B) | TF | 0.7908 ± 0.0054 (λ=0.5) | - | 0.7144 ± 0.0148 (λ=0.5) | - | 0.6486 ± 0.0060 (λ=0.125) |
| STE finetune (from CVX, split B) | AR | 0.7507 ± 0.0075 (λ=0.5) | - | 0.6686 ± 0.0167 (λ=0.5) | - | 0.6011 ± 0.0015 (λ=1) |

