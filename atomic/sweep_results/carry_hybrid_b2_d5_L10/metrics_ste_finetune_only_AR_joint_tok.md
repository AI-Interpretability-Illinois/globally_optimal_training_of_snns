# Carry hybrid b2 d5 L10 — AR joint token accuracy

- **Source:** `sweep_results/carry_hybrid_b2_d5_L10/metrics_ste_finetune_only.json`
- **finetune_variant (file):** `ste_from_cvx`
- **loaded_pretrain:** `cvx`

| λ_carry | seed | phase | AR joint (ID) | AR joint OOD d=10 | AR joint OOD d=25 | AR joint OOD d=50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.125 | 0 | pretrain_eval (loaded init) | 0.3223 | 0.2894 | 0.2809 | 0.2751 |
| 0.125 | 0 | ste_finetune | 0.3431 | 0.3548 | 0.3594 | 0.3624 |
| 0.25 | 0 | pretrain_eval (loaded init) | 0.3223 | 0.2894 | 0.2809 | 0.2751 |
| 0.25 | 0 | ste_finetune | 0.3397 | 0.3433 | 0.3454 | 0.3486 |
| 0.5 | 0 | pretrain_eval (loaded init) | 0.3433 | 0.3042 | 0.2864 | 0.2826 |
| 0.5 | 0 | ste_finetune | 0.3446 | 0.3517 | 0.3595 | 0.3577 |
| 0.75 | 0 | pretrain_eval (loaded init) | 0.4538 | 0.3739 | 0.3164 | 0.3027 |
| 0.75 | 0 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1 | 0 | pretrain_eval (loaded init) | 0.4762 | 0.3935 | 0.3242 | 0.3054 |
| 1 | 0 | ste_finetune | 0.1025 | 0.1145 | 0.1178 | 0.1254 |
| 1.25 | 0 | pretrain_eval (loaded init) | 0.4798 | 0.3929 | 0.3247 | 0.3058 |
| 1.25 | 0 | ste_finetune | 0.2067 | 0.2656 | 0.3053 | 0.3195 |
| 1.5 | 0 | pretrain_eval (loaded init) | 0.4803 | 0.3938 | 0.3253 | 0.3041 |
| 1.5 | 0 | ste_finetune | 0.2100 | 0.2760 | 0.3306 | 0.3508 |
| 2 | 0 | pretrain_eval (loaded init) | 0.4801 | 0.3937 | 0.3238 | 0.3063 |
| 2 | 0 | ste_finetune | 0.2303 | 0.2895 | 0.3333 | 0.3516 |
| 4 | 0 | pretrain_eval (loaded init) | 0.5024 | 0.4074 | 0.3321 | 0.3140 |
| 4 | 0 | ste_finetune | 0.2894 | 0.3226 | 0.3524 | 0.3608 |
| 6 | 0 | pretrain_eval (loaded init) | 0.4995 | 0.4083 | 0.3326 | 0.3156 |
| 6 | 0 | ste_finetune | 0.2926 | 0.3224 | 0.3410 | 0.3469 |
| 8 | 0 | pretrain_eval (loaded init) | 0.5042 | 0.4091 | 0.3306 | 0.3131 |
| 8 | 0 | ste_finetune | 0.2868 | 0.3236 | 0.3477 | 0.3556 |
| 10 | 0 | pretrain_eval (loaded init) | 0.5039 | 0.4085 | 0.3314 | 0.3146 |
| 10 | 0 | ste_finetune | 0.2889 | 0.3232 | 0.3523 | 0.3594 |
| 0.125 | 1 | pretrain_eval (loaded init) | 0.3361 | 0.3003 | 0.2866 | 0.2887 |
| 0.125 | 1 | ste_finetune | 0.2243 | 0.2471 | 0.2977 | 0.3222 |
| 0.25 | 1 | pretrain_eval (loaded init) | 0.4067 | 0.3334 | 0.3030 | 0.2898 |
| 0.25 | 1 | ste_finetune | 0.2126 | 0.2529 | 0.2974 | 0.3247 |
| 0.5 | 1 | pretrain_eval (loaded init) | 0.4406 | 0.3527 | 0.3083 | 0.2960 |
| 0.5 | 1 | ste_finetune | 0.3154 | 0.3046 | 0.3085 | 0.3219 |
| 0.75 | 1 | pretrain_eval (loaded init) | 0.4401 | 0.3527 | 0.3080 | 0.2963 |
| 0.75 | 1 | ste_finetune | 0.2799 | 0.2836 | 0.2601 | 0.2541 |
| 1 | 1 | pretrain_eval (loaded init) | 0.4525 | 0.3572 | 0.3093 | 0.2955 |
| 1 | 1 | ste_finetune | 0.2983 | 0.2982 | 0.3086 | 0.3170 |
| 1.25 | 1 | pretrain_eval (loaded init) | 0.4725 | 0.3690 | 0.3142 | 0.2982 |
| 1.25 | 1 | ste_finetune | 0.3335 | 0.3051 | 0.2986 | 0.3131 |
| 1.5 | 1 | pretrain_eval (loaded init) | 0.4803 | 0.3723 | 0.3165 | 0.3001 |
| 1.5 | 1 | ste_finetune | 0.2575 | 0.2773 | 0.3130 | 0.3328 |
| 2 | 1 | pretrain_eval (loaded init) | 0.4914 | 0.3821 | 0.3169 | 0.2997 |
| 2 | 1 | ste_finetune | 0.3104 | 0.3028 | 0.2832 | 0.2881 |
| 4 | 1 | pretrain_eval (loaded init) | 0.5041 | 0.3888 | 0.3207 | 0.3015 |
| 4 | 1 | ste_finetune | 0.2627 | 0.2739 | 0.3120 | 0.3286 |
| 6 | 1 | pretrain_eval (loaded init) | 0.4953 | 0.3808 | 0.3164 | 0.3003 |
| 6 | 1 | ste_finetune | 0.3188 | 0.3145 | 0.3257 | 0.3338 |
| 8 | 1 | pretrain_eval (loaded init) | 0.5016 | 0.3851 | 0.3178 | 0.3014 |
| 8 | 1 | ste_finetune | 0.3250 | 0.3100 | 0.2936 | 0.2845 |
| 10 | 1 | pretrain_eval (loaded init) | 0.5026 | 0.3868 | 0.3187 | 0.3019 |
| 10 | 1 | ste_finetune | 0.2122 | 0.2218 | 0.2308 | 0.2394 |
| 0.125 | 2 | pretrain_eval (loaded init) | 0.3171 | 0.2745 | 0.2503 | 0.2460 |
| 0.125 | 2 | ste_finetune | 0.3428 | 0.2820 | 0.2104 | 0.1841 |
| 0.25 | 2 | pretrain_eval (loaded init) | 0.3171 | 0.2745 | 0.2503 | 0.2460 |
| 0.25 | 2 | ste_finetune | 0.3486 | 0.2746 | 0.2028 | 0.1698 |
| 0.5 | 2 | pretrain_eval (loaded init) | 0.3171 | 0.2745 | 0.2503 | 0.2460 |
| 0.5 | 2 | ste_finetune | 0.3314 | 0.2613 | 0.2010 | 0.1692 |
| 0.75 | 2 | pretrain_eval (loaded init) | 0.4600 | 0.3580 | 0.2822 | 0.2662 |
| 0.75 | 2 | ste_finetune | 0.3496 | 0.2978 | 0.2463 | 0.2484 |
| 1 | 2 | pretrain_eval (loaded init) | 0.4578 | 0.3602 | 0.2851 | 0.2679 |
| 1 | 2 | ste_finetune | 0.2212 | 0.2304 | 0.2098 | 0.1936 |
| 1.25 | 2 | pretrain_eval (loaded init) | 0.4565 | 0.3599 | 0.2862 | 0.2678 |
| 1.25 | 2 | ste_finetune | 0.2552 | 0.2955 | 0.3246 | 0.3341 |
| 1.5 | 2 | pretrain_eval (loaded init) | 0.4606 | 0.3611 | 0.2870 | 0.2669 |
| 1.5 | 2 | ste_finetune | 0.3493 | 0.3102 | 0.2698 | 0.2525 |
| 2 | 2 | pretrain_eval (loaded init) | 0.4762 | 0.3707 | 0.2878 | 0.2679 |
| 2 | 2 | ste_finetune | 0.2741 | 0.2532 | 0.2200 | 0.2197 |
| 4 | 2 | pretrain_eval (loaded init) | 0.4785 | 0.3705 | 0.2878 | 0.2678 |
| 4 | 2 | ste_finetune | 0.2876 | 0.3240 | 0.3435 | 0.3524 |
| 6 | 2 | pretrain_eval (loaded init) | 0.4808 | 0.3719 | 0.2909 | 0.2688 |
| 6 | 2 | ste_finetune | 0.2961 | 0.3287 | 0.3464 | 0.3427 |
| 8 | 2 | pretrain_eval (loaded init) | 0.4858 | 0.3751 | 0.2951 | 0.2707 |
| 8 | 2 | ste_finetune | 0.3223 | 0.3468 | 0.3561 | 0.3593 |
| 10 | 2 | pretrain_eval (loaded init) | 0.4873 | 0.3771 | 0.2947 | 0.2714 |
| 10 | 2 | ste_finetune | 0.3444 | 0.3573 | 0.3625 | 0.3583 |

*ID = `n_digits` train length test split; OOD = longer digit counts from `ood_digits`.*
