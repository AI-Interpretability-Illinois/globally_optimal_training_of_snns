# arithmetic_add_b2_len_gen_L3_traind5_K2_20260505_111902 vs hybrid carry bench (sum_token_acc)

- Length-gen run: `arithmetic_add_b2_len_gen_L3_traind5_K2_20260505_111902` (arith_base=2, L=3, K_parallel=2, n_digits_train=5, n_train=8192, loss_type=hinge_ovr, cvx_method=cvx, variants=full5, mask_carry_in=True, seeds=[0, 1, 2])

- Hybrid run:    `carry_hybrid_b2_d5_20260423_022747` (arith_base=2, L=3, K_parallel=2, n_digits=5, ste_last_layer_readout=spike, cvx_last_layer_readout=spike, lambda_carry_grid=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0], seeds=[0, 1, 2])

- Metric: **sum_token_acc** in the hybrid bench averages ``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where ``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the MSD carry-out at the final step. The matching length-gen quantity is its raw ``token_acc`` (single-head, same target sequence, same averaging) — both metrics are mathematically identical.

- Hybrid eval: **AR** (autoregressive rollout — model feeds its own predicted carry into the next step). Teacher-forcing (TF) hybrid numbers are not listed here.

- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid uses its own seed list (they are independent — see headers above).

## Length-gen token_acc (== hybrid sum_token_acc; full T positions)

| length-gen stage | ID (n=5) | n=10 | n=20 | n=50 |
|:---|---:|---:|---:|---:|
| len-gen STE pretrain | 0.6174 ± 0.0093 | 0.5613 ± 0.0150 | 0.5299 ± 0.0145 | 0.5135 ± 0.0144 |
| len-gen CVX from STE init | 0.6757 ± 0.0105 | 0.6040 ± 0.0250 | 0.5545 ± 0.0166 | 0.5268 ± 0.0095 |
| len-gen STE finetune (from STE) | 0.6392 ± 0.0039 | 0.5813 ± 0.0025 | 0.5496 ± 0.0027 | 0.5271 ± 0.0030 |
| len-gen CVX pretrain (Gaussian) | 0.7757 ± 0.0026 | 0.6607 ± 0.0077 | 0.5963 ± 0.0055 | 0.5518 ± 0.0057 |
| len-gen STE finetune (from CVX) | 0.6266 ± 0.0297 | 0.5607 ± 0.0179 | 0.5332 ± 0.0092 | 0.5131 ± 0.0056 |

## Hybrid sum_token_acc at λ_carry = 1.0 (AR eval)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=50 |
|:---|---:|---:|---:|---:|
| STE pretrain (split A) | 0.5865 ± 0.0272 | 0.5767 ± 0.0336 | 0.5769 ± 0.0296 | 0.5708 ± 0.0258 |
| CVX from STE init | 0.6047 ± 0.0049 | 0.5928 ± 0.0140 | 0.5872 ± 0.0193 | 0.5791 ± 0.0243 |
| STE finetune (from STE, split B) | 0.6671 ± 0.0754 | 0.6481 ± 0.0597 | 0.6310 ± 0.0529 | 0.6202 ± 0.0460 |
| CVX pretrain (Gaussian) | 0.7109 ± 0.0126 | 0.6265 ± 0.0008 | 0.5709 ± 0.0071 | 0.5380 ± 0.0048 |
| STE finetune (from CVX, split B) | 0.8538 ± 0.0032 | 0.7452 ± 0.0023 | 0.6641 ± 0.0109 | 0.6011 ± 0.0015 |

## Hybrid sum_token_acc — best λ_carry per cell (AR eval; selection on max mean)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=50 |
|:---|---:|---:|---:|---:|
| STE pretrain (split A) | 0.6235 ± 0.0178 (λ=0.5) | 0.6066 ± 0.0188 (λ=0.75) | 0.5964 ± 0.0186 (λ=0.75) | 0.5871 ± 0.0194 (λ=0.75) |
| CVX from STE init | 0.6241 ± 0.0516 (λ=1.5) | 0.6036 ± 0.0442 (λ=1.5) | 0.5884 ± 0.0238 (λ=0.75) | 0.5818 ± 0.0264 (λ=0.75) |
| STE finetune (from STE, split B) | 0.7257 ± 0.0412 (λ=1.5) | 0.6710 ± 0.0303 (λ=1.5) | 0.6420 ± 0.0533 (λ=1.25) | 0.6295 ± 0.0430 (λ=1.25) |
| CVX pretrain (Gaussian) | 0.7158 ± 0.0136 (λ=10) | 0.6295 ± 0.0037 (λ=10) | 0.5724 ± 0.0065 (λ=2) | 0.5383 ± 0.0065 (λ=10) |
| STE finetune (from CVX, split B) | 0.8538 ± 0.0032 (λ=1) | 0.7507 ± 0.0075 (λ=0.5) | 0.6686 ± 0.0167 (λ=0.5) | 0.6011 ± 0.0015 (λ=1) |

