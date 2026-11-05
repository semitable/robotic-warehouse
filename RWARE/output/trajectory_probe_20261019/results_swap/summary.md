# Source swap (E1): same evaluated models, probe windows from two source policies
AUROC mean over models (same team closer = 1, chance .5; seed 1; 6 pairs per model).

## policy — 8 models that are neither source (8)

| source | window | len1 | traj_all | traj_late | traj_last | traj_last2 |
|---|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.48 | 0.47 | 0.52 | 0.64 | 0.66 |
| Stage04 (curriculum) | carry (n=64) | 0.58 | 0.56 | 0.48 | 0.53 | 0.59 |
| Stage04 (curriculum) | random (n=64) | 0.58 | 0.53 | 0.52 | 0.56 | 0.53 |
| scratch All (from scratch) | carry_memory (n=53) | 0.42 | 0.33 | 0.31 | 0.31 | 0.30 |
| scratch All (from scratch) | carry (n=64) | 0.50 | 0.45 | 0.45 | 0.53 | 0.45 |
| scratch All (from scratch) | random (n=64) | 0.62 | 0.52 | 0.53 | 0.59 | 0.53 |

## policy — curriculum (Stage01-03, sp E1) excl. sources (5)

| source | window | len1 | traj_all | traj_late | traj_last | traj_last2 |
|---|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.45 | 0.42 | 0.53 | 0.72 | 0.75 |
| Stage04 (curriculum) | carry (n=64) | 0.65 | 0.62 | 0.50 | 0.60 | 0.70 |
| Stage04 (curriculum) | random (n=64) | 0.70 | 0.62 | 0.62 | 0.68 | 0.62 |
| scratch All (from scratch) | carry_memory (n=53) | 0.35 | 0.23 | 0.20 | 0.23 | 0.20 |
| scratch All (from scratch) | carry (n=64) | 0.55 | 0.47 | 0.47 | 0.57 | 0.47 |
| scratch All (from scratch) | random (n=64) | 0.75 | 0.60 | 0.62 | 0.70 | 0.60 |

## policy — from scratch excl. sources (3)

| source | window | len1 | traj_all | traj_late | traj_last | traj_last2 |
|---|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.54 | 0.54 | 0.50 | 0.50 | 0.50 |
| Stage04 (curriculum) | carry (n=64) | 0.46 | 0.46 | 0.46 | 0.42 | 0.42 |
| Stage04 (curriculum) | random (n=64) | 0.38 | 0.38 | 0.33 | 0.38 | 0.38 |
| scratch All (from scratch) | carry_memory (n=53) | 0.54 | 0.50 | 0.50 | 0.46 | 0.46 |
| scratch All (from scratch) | carry (n=64) | 0.42 | 0.42 | 0.42 | 0.46 | 0.42 |
| scratch All (from scratch) | random (n=64) | 0.42 | 0.38 | 0.38 | 0.42 | 0.42 |

## reward — 8 models that are neither source (8)

| source | window | mlp_len1 | gru_len1 | gru_traj | gru_last |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.41 | 0.41 | 0.53 | 0.62 |
| Stage04 (curriculum) | carry (n=64) | 0.92 | 0.75 | 0.81 | 0.89 |
| Stage04 (curriculum) | random (n=64) | 0.86 | 0.77 | 0.78 | 0.77 |
| scratch All (from scratch) | carry_memory (n=53) | 0.55 | 0.47 | 0.58 | 0.58 |
| scratch All (from scratch) | carry (n=64) | 0.88 | 0.72 | 0.80 | 0.89 |
| scratch All (from scratch) | random (n=64) | 0.81 | 0.77 | 0.80 | 0.52 |

## reward — curriculum (Stage01-03, sp E1) excl. sources (5)

| source | window | mlp_len1 | gru_len1 | gru_traj | gru_last |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.40 | 0.40 | 0.65 | 0.78 |
| Stage04 (curriculum) | carry (n=64) | 1.00 | 0.90 | 0.90 | 1.00 |
| Stage04 (curriculum) | random (n=64) | 0.95 | 0.90 | 1.00 | 0.95 |
| scratch All (from scratch) | carry_memory (n=53) | 0.55 | 0.53 | 0.68 | 0.68 |
| scratch All (from scratch) | carry (n=64) | 0.97 | 0.85 | 0.95 | 1.00 |
| scratch All (from scratch) | random (n=64) | 0.93 | 0.85 | 0.97 | 0.65 |

## reward — from scratch excl. sources (3)

| source | window | mlp_len1 | gru_len1 | gru_traj | gru_last |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry_memory (n=8) | 0.42 | 0.42 | 0.33 | 0.38 |
| Stage04 (curriculum) | carry (n=64) | 0.79 | 0.50 | 0.67 | 0.71 |
| Stage04 (curriculum) | random (n=64) | 0.71 | 0.54 | 0.42 | 0.46 |
| scratch All (from scratch) | carry_memory (n=53) | 0.54 | 0.38 | 0.42 | 0.42 |
| scratch All (from scratch) | carry (n=64) | 0.71 | 0.50 | 0.54 | 0.71 |
| scratch All (from scratch) | random (n=64) | 0.62 | 0.62 | 0.50 | 0.29 |
