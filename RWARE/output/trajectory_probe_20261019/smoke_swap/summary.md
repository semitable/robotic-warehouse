# Source swap (E1): same evaluated models, probe windows from two source policies
AUROC mean over models (same team closer = 1, chance .5; seed 1; 6 pairs per model).

## policy — 8 models that are neither source (2)

| source | window | len1 | traj_all | traj_late | traj_last | traj_last2 |
|---|---|---|---|---|---|---|
| Stage04 (curriculum) | carry (n=6) | 0.69 | 0.69 | 0.62 | 0.62 | 0.56 |
| Stage04 (curriculum) | random (n=8) | 0.88 | 0.81 | 0.88 | 1.00 | 0.88 |
| scratch All (from scratch) | carry (n=8) | 0.56 | 0.50 | 0.31 | 0.56 | 0.44 |
| scratch All (from scratch) | random (n=8) | 0.94 | 0.88 | 0.88 | 0.88 | 0.75 |

## policy — curriculum (Stage01-03, sp E1) excl. sources (2)

| source | window | len1 | traj_all | traj_late | traj_last | traj_last2 |
|---|---|---|---|---|---|---|
| Stage04 (curriculum) | carry (n=6) | 0.69 | 0.69 | 0.62 | 0.62 | 0.56 |
| Stage04 (curriculum) | random (n=8) | 0.88 | 0.81 | 0.88 | 1.00 | 0.88 |
| scratch All (from scratch) | carry (n=8) | 0.56 | 0.50 | 0.31 | 0.56 | 0.44 |
| scratch All (from scratch) | random (n=8) | 0.94 | 0.88 | 0.88 | 0.88 | 0.75 |

## reward — 8 models that are neither source (2)

| source | window | mlp_len1 | gru_len1 | gru_traj | gru_last |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry (n=6) | 0.25 | 0.25 | 0.38 | 0.50 |
| Stage04 (curriculum) | random (n=8) | 0.31 | 0.50 | 0.50 | 0.50 |
| scratch All (from scratch) | carry (n=8) | 0.38 | 0.50 | 0.56 | 0.62 |
| scratch All (from scratch) | random (n=8) | 0.25 | 0.50 | 0.50 | 0.62 |

## reward — curriculum (Stage01-03, sp E1) excl. sources (2)

| source | window | mlp_len1 | gru_len1 | gru_traj | gru_last |
|---|---|---|---|---|---|
| Stage04 (curriculum) | carry (n=6) | 0.25 | 0.25 | 0.38 | 0.50 |
| Stage04 (curriculum) | random (n=8) | 0.31 | 0.50 | 0.50 | 0.50 |
| scratch All (from scratch) | carry (n=8) | 0.38 | 0.50 | 0.56 | 0.62 |
| scratch All (from scratch) | random (n=8) | 0.25 | 0.50 | 0.50 | 0.62 |
