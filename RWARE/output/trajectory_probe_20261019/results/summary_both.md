# Trajectory vs length-1 probes (AUROC; same team closer = 1, chance .5; seed 1)

## policy — E1 curriculum (Stage01-04, sp E1)

| window type | len1 | traj_all | traj_late |
|---|---|---|---|
| carry | 0.56 | 0.58 | 0.54 |
| carry_memory | 0.35 | 0.54 | 0.48 |
| random | 0.60 | 0.56 | 0.54 |

## policy — E1 from scratch (6.4M)

| window type | len1 | traj_all | traj_late |
|---|---|---|---|
| carry | 0.56 | 0.41 | 0.41 |
| carry_memory | 0.69 | 0.69 | 0.69 |
| random | 0.53 | 0.41 | 0.41 |

## policy — E1-F checkpoints (1.54M)

| window type | len1 | traj_all | traj_late |
|---|---|---|---|
| carry | 0.38 | 0.35 | 0.35 |
| carry_memory | 0.38 | 0.42 | 0.47 |
| random | 0.40 | 0.45 | 0.53 |

## reward — E1 curriculum (Stage01-04, sp E1)

| window type | len1 | traj_step | traj_sum |
|---|---|---|---|
| carry | 1.00 | 1.00 | 0.94 |
| carry_memory | 0.38 | 0.25 | 0.25 |
| random | 1.00 | 1.00 | 0.50 |

## reward — E1 from scratch (6.4M)

| window type | len1 | traj_step | traj_sum |
|---|---|---|---|
| carry | 0.62 | 0.38 | 0.25 |
| carry_memory | 0.75 | 0.25 | 0.25 |
| random | 0.50 | 0.50 | 0.25 |

## reward — E1-F checkpoints (1.54M)

| window type | len1 | traj_step | traj_sum |
|---|---|---|---|
| carry | 0.50 | 0.69 | 0.75 |
| carry_memory | 0.56 | 0.75 | 0.81 |
| random | 0.50 | 0.75 | 0.81 |
