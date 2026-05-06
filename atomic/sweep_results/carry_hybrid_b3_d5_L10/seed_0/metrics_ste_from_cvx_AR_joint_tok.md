# Carry hybrid b3 d5 L10 — AR joint token accuracy

- **Source:** `metrics_ste_from_cvx.json` (carry hybrid b3 d5 L10)
- **finetune_variant:** `ste_from_cvx`

| λ_carry | phase | AR joint (ID) | AR joint OOD d=10 | AR joint OOD d=25 | AR joint OOD d=50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.125 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 0.125 | ste_finetune | 0.2438 | 0.1948 | 0.1566 | 0.1558 |
| 0.25 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 0.25 | ste_finetune | 0.2458 | 0.1950 | 0.1573 | 0.1536 |
| 0.5 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 0.5 | ste_finetune | 0.2412 | 0.1920 | 0.1531 | 0.1472 |
| 0.75 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 0.75 | ste_finetune | 0.2371 | 0.1847 | 0.1469 | 0.1500 |
| 1 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 1 | ste_finetune | 0.2410 | 0.1839 | 0.1425 | 0.1456 |
| 1.25 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 1.25 | ste_finetune | 0.2419 | 0.1932 | 0.1494 | 0.1526 |
| 1.5 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 1.5 | ste_finetune | 0.2472 | 0.2069 | 0.1556 | 0.1497 |
| 2 | pretrain_eval (loaded init) | 0.2088 | 0.1879 | 0.1737 | 0.1706 |
| 2 | ste_finetune | 0.2354 | 0.2200 | 0.1655 | 0.1576 |
| 4 | pretrain_eval (loaded init) | 0.2472 | 0.2071 | 0.1809 | 0.1748 |
| 4 | ste_finetune | 0.1338 | 0.1315 | 0.1413 | 0.1387 |
| 6 | pretrain_eval (loaded init) | 0.2612 | 0.2156 | 0.1835 | 0.1766 |
| 6 | ste_finetune | 0.1343 | 0.1714 | 0.2082 | 0.2129 |
| 8 | pretrain_eval (loaded init) | 0.2671 | 0.2195 | 0.1851 | 0.1774 |
| 8 | ste_finetune | 0.1545 | 0.1807 | 0.1996 | 0.2168 |
| 10 | pretrain_eval (loaded init) | 0.2687 | 0.2211 | 0.1860 | 0.1777 |
| 10 | ste_finetune | 0.1291 | 0.1641 | 0.1907 | 0.1979 |

*ID / OOD: mean joint token accuracy over (sequence, timestep) in evaluated splits; OOD digit counts from `ood_eval`.*
