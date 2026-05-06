# DFA random ablation summary

- **Sweep:** `dfa_random_ablation_T9_ntrain5000_20260505_163759`
- **Manifest T / n_train:** T=9, n_train=5000
- **Stationary variants (dfa_type):** dense, bias_accept, bias_reject
- **Instances per variant:** 3

Each **inst** row: metrics averaged over **seeds 0–2**. **mean** row: average over instances in that (|Q|,|Σ|,type) group.

**Pipeline:** `ste_pre` → `cvx←ste` → `ft_ste` → `cvx_g` → `ft_cvx`.
**OOD:** last-step accuracy, length multipliers ×2 / ×5 / ×10 (T=18, 45, 90).

## |Q|=5, |Σ|=2, `bias_accept`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.849 | 0.742/0.766/0.734 | 0.896 | 0.871 | 0.755 | 0.910 | 0.869 | 0.741 |
| i1 | 0.874 | 0.763/0.766/0.770 | 0.912 | 0.881 | 0.802 | 0.906 | 0.873 | 0.783 |
| i2 | 0.710 | 0.521/0.486/0.509 | 0.736 | 0.735 | 0.499 | 0.742 | 0.674 | 0.506 |
| **mean** | 0.811 | 0.675/0.673/0.671 | 0.848 | 0.829 | 0.685 | 0.853 | 0.806 | 0.676 |

## |Q|=5, |Σ|=2, `bias_reject`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.875 | 0.804/0.804/0.829 | 0.917 | 0.886 | 0.817 | 0.894 | 0.886 | 0.800 |
| i1 | 0.889 | 0.706/0.594/0.546 | 0.884 | 0.904 | 0.548 | 0.874 | 0.882 | 0.592 |
| i2 | 0.683 | 0.631/0.646/0.658 | 0.718 | 0.695 | 0.636 | 0.740 | 0.681 | 0.596 |
| **mean** | 0.816 | 0.713/0.681/0.678 | 0.840 | 0.828 | 0.667 | 0.836 | 0.816 | 0.663 |

## |Q|=5, |Σ|=2, `dense`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.899 | 0.849/0.838/0.854 | 0.910 | 0.918 | 0.842 | 0.899 | 0.914 | 0.820 |
| i1 | 0.776 | 0.651/0.664/0.649 | 0.839 | 0.819 | 0.666 | 0.818 | 0.800 | 0.646 |
| i2 | 0.825 | 0.782/0.757/0.755 | 0.832 | 0.826 | 0.761 | 0.852 | 0.815 | 0.753 |
| **mean** | 0.833 | 0.761/0.753/0.752 | 0.860 | 0.854 | 0.757 | 0.856 | 0.843 | 0.739 |

## |Q|=5, |Σ|=3, `bias_accept`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.842 | 0.804/0.788/0.797 | 0.854 | 0.857 | 0.790 | 0.777 | 0.843 | 0.807 |
| i1 | 0.671 | 0.624/0.622/0.621 | 0.674 | 0.692 | 0.616 | 0.646 | 0.673 | 0.625 |
| i2 | 0.902 | 0.849/0.830/0.840 | 0.906 | 0.913 | 0.823 | 0.842 | 0.902 | 0.828 |
| **mean** | 0.805 | 0.759/0.747/0.752 | 0.811 | 0.821 | 0.743 | 0.755 | 0.806 | 0.753 |

## |Q|=5, |Σ|=3, `bias_reject`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.903 | 0.881/0.868/0.887 | 0.902 | 0.910 | 0.892 | 0.817 | 0.906 | 0.882 |
| i1 | 0.602 | 0.558/0.537/0.552 | 0.591 | 0.608 | 0.567 | 0.562 | 0.596 | 0.570 |
| i2 | 0.646 | 0.623/0.648/0.633 | 0.646 | 0.655 | 0.637 | 0.648 | 0.642 | 0.622 |
| **mean** | 0.717 | 0.687/0.685/0.691 | 0.713 | 0.725 | 0.699 | 0.676 | 0.715 | 0.691 |

## |Q|=5, |Σ|=3, `dense`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.755 | 0.694/0.686/0.711 | 0.756 | 0.779 | 0.698 | 0.701 | 0.750 | 0.718 |
| i1 | 0.658 | 0.616/0.628/0.611 | 0.660 | 0.668 | 0.623 | 0.590 | 0.652 | 0.618 |
| i2 | 0.638 | 0.575/0.560/0.571 | 0.643 | 0.662 | 0.562 | 0.619 | 0.626 | 0.558 |
| **mean** | 0.684 | 0.628/0.625/0.631 | 0.686 | 0.703 | 0.628 | 0.637 | 0.676 | 0.631 |

## |Q|=5, |Σ|=5, `bias_accept`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.711 | 0.694/0.685/0.690 | 0.707 | 0.721 | 0.716 | 0.582 | 0.707 | 0.680 |
| **mean** | 0.711 | 0.694/0.685/0.690 | 0.707 | 0.721 | 0.716 | 0.582 | 0.707 | 0.680 |

## |Q|=5, |Σ|=5, `dense`

| inst | ste_pre ID | ste_pre OOD ×2/×5/×10 | cvx←ste ID | ft_ste ID | ft_ste OOD×10 | cvx_g ID | ft_cvx ID | ft_cvx OOD×10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i0 | 0.644 | 0.623/0.619/0.622 | 0.629 | 0.641 | 0.623 | 0.558 | 0.633 | 0.606 |
| i1 | 0.585 | 0.580/0.589/0.575 | 0.583 | 0.607 | 0.593 | 0.575 | 0.596 | 0.582 |
| i2 | 0.665 | 0.635/0.642/0.637 | 0.656 | 0.665 | 0.634 | 0.569 | 0.660 | 0.652 |
| **mean** | 0.631 | 0.613/0.617/0.611 | 0.623 | 0.638 | 0.617 | 0.567 | 0.630 | 0.613 |

## Cross-cell comparison — final in-distribution `ft_cvx` ID accuracy

(Mean over instances with data; `—` if no finished cell.)

| \|Q\| | \|Σ\| | dense | bias_accept | bias_reject |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 2 | 0.843 | 0.806 | 0.816 |
| 5 | 3 | 0.676 | 0.806 | 0.715 |
| 5 | 5 | 0.630 | 0.707 | — |

## Cross-cell — `ft_cvx` OOD ×10 last-step (mean over instances)

| \|Q\| | \|Σ\| | dense | bias_accept | bias_reject |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 2 | 0.739 | 0.676 | 0.663 |
| 5 | 3 | 0.631 | 0.753 | 0.691 |
| 5 | 5 | 0.613 | 0.680 | — |

## Trends

- **Larger alphabet (|Q|=5, `dense`):** final `ft_cvx` ID |Σ|=2 → 0.843, |Σ|=3 → 0.676, |Σ|=5 → 0.630 — accuracy drops as |Σ| grows (2→3→5 in this sweep).
- **Stationary / DFA type at |Q|=5, |Σ|=2:** final `ft_cvx` ID — `dense` 0.843; `bias_accept` 0.806; `bias_reject` 0.816. `dense` is most stable; **`bias_accept` / `bias_reject` track close** to each other here, with **high instance variance** (see per-inst tables; e.g. some `bias_accept` draws hurt OOD badly).
- **CVX layer on frozen STE (`cvx←ste` vs `ste_pre`):** mean Δ ID last-step ≈ **+0.011**, median ≈ **+0.004** over 22 finished cells (a few instances tick **down** slightly; largest gains ~0.06 on hard `dense` draws).
- **Length generalisation:** after full pipeline, **OOD ×10** last-step averages **~0.681** vs **first-stage ID ~0.755** (cell-wise pool); long-sequence accuracy stays well below ID whenever the model relies on finite-length training.
- **Sweep coverage (this manifest):** 22 bench cells on disk; only |Q|∈{5} and |Σ|∈{2, 3, 5} appear — planned q10/q15 rows are **not** in this manifest yet.
