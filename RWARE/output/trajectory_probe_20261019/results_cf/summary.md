# Counterfactual split of the reward probe (E1, MLP length-1 predictors; seed 1)

| source | window | windows | Q_diff / queries | mean |r_red - r_blue| |
|---|---|---|---|---|
| Stage04 (curriculum) | carry | 64 | 604 / 768 | 0.0313 |
| Stage04 (curriculum) | random | 64 | 480 / 768 | 0.0510 |
| Stage04 (curriculum) | carry_memory | 8 | 37 / 96 | 0.0019 |
| scratch All (from scratch) | carry | 64 | 392 / 768 | 0.0091 |
| scratch All (from scratch) | random | 64 | 276 / 768 | 0.0178 |
| scratch All (from scratch) | carry_memory | 53 | 243 / 636 | 0.0019 |

## 8 models (neither source) (8)

| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry | 0.92 | 0.92 | 0.84 | 0.68 | 0.01 |
| Stage04 (curriculum) | random | 0.86 | 0.86 | 0.80 | 0.50 | 0.32 |
| Stage04 (curriculum) | carry_memory | 0.41 | 0.42 | 0.42 | 0.14 | 0.09 |
| scratch All (from scratch) | carry | 0.88 | 0.92 | 0.81 | 0.38 | 0.05 |
| scratch All (from scratch) | random | 0.81 | 0.84 | 0.75 | 0.42 | 0.21 |
| scratch All (from scratch) | carry_memory | 0.55 | 0.50 | 0.58 | -0.01 | -0.08 |

## curriculum excl. sources (5)

| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry | 1.00 | 1.00 | 0.95 | 0.77 | 0.02 |
| Stage04 (curriculum) | random | 0.95 | 0.95 | 0.88 | 0.53 | 0.38 |
| Stage04 (curriculum) | carry_memory | 0.40 | 0.45 | 0.40 | 0.01 | -0.06 |
| scratch All (from scratch) | carry | 0.97 | 1.00 | 0.95 | 0.33 | 0.06 |
| scratch All (from scratch) | random | 0.93 | 0.95 | 0.90 | 0.36 | 0.27 |
| scratch All (from scratch) | carry_memory | 0.55 | 0.47 | 0.57 | -0.03 | -0.15 |

## from scratch excl. sources (3)

| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry | 0.79 | 0.79 | 0.67 | 0.54 | -0.03 |
| Stage04 (curriculum) | random | 0.71 | 0.71 | 0.67 | 0.46 | 0.21 |
| Stage04 (curriculum) | carry_memory | 0.42 | 0.38 | 0.46 | 0.37 | 0.33 |
| scratch All (from scratch) | carry | 0.71 | 0.79 | 0.58 | 0.46 | 0.02 |
| scratch All (from scratch) | random | 0.62 | 0.67 | 0.50 | 0.52 | 0.10 |
| scratch All (from scratch) | carry_memory | 0.54 | 0.54 | 0.58 | 0.02 | 0.04 |
