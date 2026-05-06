# arithmetic_add_b5_len_gen_L3_traind5_K2_20260505_123229 vs hybrid carry bench (sum_token_acc)

- Length-gen run: `arithmetic_add_b5_len_gen_L3_traind5_K2_20260505_123229` (arith_base=5, L=3, K_parallel=2, n_digits_train=5, n_train=8192, loss_type=hinge_ovr, cvx_method=cvx, variants=full5, mask_carry_in=True, seeds=[0, 1, 2])

- Hybrid run:    `carry_hybrid_b5_d5_20260426_201340` (arith_base=5, L=3, K_parallel=2, n_digits=5, ste_last_layer_readout=spike, cvx_last_layer_readout=spike, lambda_carry_grid=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0], seeds=[0, 1])

- Metric: **sum_token_acc** in the hybrid bench averages ``(sum_pred == y_sum).mean()`` over all ``T = n_digits + 1`` timesteps, where ``y_sum = target_tokens`` covers ``n_digits`` sum-digit positions plus the MSD carry-out at the final step. The matching length-gen quantity is its raw ``token_acc`` (single-head, same target sequence, same averaging) — both metrics are mathematically identical.

- Hybrid eval: **AR** (autoregressive rollout — model feeds its own predicted carry into the next step). Teacher-forcing (TF) hybrid numbers are not listed here.

- Cells render `mean ± std` across seeds. Length-gen uses its own seed list; hybrid uses its own seed list (they are independent — see headers above).

## Length-gen token_acc (== hybrid sum_token_acc; full T positions)

| length-gen stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| len-gen STE pretrain | 0.2812 ± 0.0020 | 0.2342 ± 0.0040 | 0.2096 ± 0.0041 | - | 0.2021 ± 0.0026 |
| len-gen CVX from STE init | 0.3011 ± 0.0070 | 0.2429 ± 0.0077 | 0.2175 ± 0.0100 | - | 0.2059 ± 0.0053 |
| len-gen STE finetune (from STE) | 0.2883 ± 0.0057 | 0.2380 ± 0.0051 | 0.2153 ± 0.0044 | - | 0.2041 ± 0.0043 |
| len-gen CVX pretrain (Gaussian) | 0.3302 ± 0.0090 | 0.2699 ± 0.0070 | 0.2325 ± 0.0027 | - | 0.2156 ± 0.0020 |
| len-gen STE finetune (from CVX) | 0.2611 ± 0.0232 | 0.2284 ± 0.0212 | 0.2101 ± 0.0183 | - | 0.2033 ± 0.0096 |

## Hybrid sum_token_acc at λ_carry = 1.0 (AR eval)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | 0.3018 ± 0.0000 | 0.2379 ± 0.0000 | - | 0.2091 ± 0.0000 | 0.2052 ± 0.0000 |
| CVX from STE init | 0.2873 ± 0.0000 | 0.2322 ± 0.0000 | - | 0.2062 ± 0.0000 | 0.2027 ± 0.0000 |
| STE finetune (from STE, split B) | 0.3203 ± 0.0000 | 0.2555 ± 0.0000 | - | 0.2185 ± 0.0000 | 0.2101 ± 0.0000 |
| CVX pretrain (Gaussian) | 0.2777 ± 0.0000 | 0.2345 ± 0.0000 | - | 0.2119 ± 0.0000 | 0.2068 ± 0.0000 |
| STE finetune (from CVX, split B) | 0.3029 ± 0.0000 | 0.2462 ± 0.0000 | - | 0.2169 ± 0.0000 | 0.2100 ± 0.0000 |

## Hybrid sum_token_acc — best λ_carry per cell (AR eval; selection on max mean)

| hybrid stage | ID (n=5) | n=10 | n=20 | n=25 | n=50 |
|:---|---:|---:|---:|---:|---:|
| STE pretrain (split A) | 0.3517 ± 0.0000 (λ=4) | 0.2672 ± 0.0000 (λ=4) | - | 0.2254 ± 0.0000 (λ=4) | 0.2136 ± 0.0000 (λ=4) |
| CVX from STE init | 0.3234 ± 0.0000 (λ=8) | 0.2551 ± 0.0000 (λ=8) | - | 0.2156 ± 0.0000 (λ=8) | 0.2073 ± 0.0000 (λ=8) |
| STE finetune (from STE, split B) | 0.3631 ± 0.0000 (λ=10) | 0.2845 ± 0.0000 (λ=4) | - | 0.2432 ± 0.0000 (λ=2) | 0.2283 ± 0.0000 (λ=4) |
| CVX pretrain (Gaussian) | 0.2796 ± 0.0000 (λ=1.5) | 0.2372 ± 0.0000 (λ=0.125) | - | 0.2125 ± 0.0000 (λ=0.25) | 0.2069 ± 0.0000 (λ=0.125) |
| STE finetune (from CVX, split B) | 0.3175 ± 0.0000 (λ=6) | 0.2670 ± 0.0000 (λ=6) | - | 0.2233 ± 0.0000 (λ=6) | 0.2134 ± 0.0000 (λ=6) |

