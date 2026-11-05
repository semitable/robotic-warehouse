# Counterfactual split of the reward probe (E1, MLP length-1 predictors; seed 1)

| source | window | windows | Q_diff / queries | mean |r_red - r_blue| |
|---|---|---|---|---|
| Stage04 (curriculum) | carry | 6 | 60 / 72 | 0.0319 |
| Stage04 (curriculum) | random | 8 | 43 / 96 | 0.0412 |
| scratch All (from scratch) | carry | 8 | 46 / 96 | 0.0232 |
| scratch All (from scratch) | random | 8 | 43 / 96 | 0.0290 |

## 8 models (neither source) (2)

| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry | 0.25 | 0.25 | 0.25 | 0.15 | -0.05 |
| Stage04 (curriculum) | random | 0.31 | 0.31 | 0.31 | -0.18 | 0.11 |
| scratch All (from scratch) | carry | 0.38 | 0.38 | 0.38 | -0.13 | 0.34 |
| scratch All (from scratch) | random | 0.25 | 0.31 | 0.25 | 0.06 | -0.21 |

## curriculum excl. sources (2)

| source / window | AUROC all | AUROC Q_diff | AUROC Q_same | corr own (Q_diff) | corr other (Q_diff) |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry | 0.25 | 0.25 | 0.25 | 0.15 | -0.05 |
| Stage04 (curriculum) | random | 0.31 | 0.31 | 0.31 | -0.18 | 0.11 |
| scratch All (from scratch) | carry | 0.38 | 0.38 | 0.38 | -0.13 | 0.34 |
| scratch All (from scratch) | random | 0.25 | 0.31 | 0.25 | 0.06 | -0.21 |
