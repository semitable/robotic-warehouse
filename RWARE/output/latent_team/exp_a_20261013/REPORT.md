# Experiment A — team identifiability and consensus premise

Generated 2026-10-13T18:43:28+09:00. Frozen models, stochastic masked own-policy rollouts, 20x500 steps, reset seeds 5200001–5200020. Early-window MC targets use the first 200 steps (finite horizon, zero tail).

AUROC and recovery use dependent episode x pair samples and only 2 same / 4 cross pairs; they are descriptive, not significance tests. canonical_js is one model-level value per pair.

## AUROC (same > cross)

| Model | canonical_js | history_js | occupancy_js | goal_evidence | critic_gap | blend_gain |
|---|---:|---:|---:|---:|---:|---:|
| stage03_init_e1 | 0.500 | 0.282 | 0.379 | 1.000 | 0.242 | 0.325 |
| e1_ippo_s1 | 0.375 | 0.267 | 0.408 | 1.000 | 0.001 | 0.251 |
| e1_pgct_s1 | 0.375 | 0.267 | 0.333 | 1.000 | 0.000 | 0.290 |
| e2_ippo_s1 | 0.375 | 0.810 | 0.867 | 1.000 | 0.452 | 0.493 |
| e2_pgct_s1 | 0.500 | 0.853 | 0.821 | 1.000 | 0.554 | 0.470 |
| stage04_pgct | 0.500 | 0.291 | 0.365 | 1.000 | 0.013 | 0.289 |

## Teammate recovery (chance .333)

| Model | canonical_js | history_js | occupancy_js | goal_evidence | critic_gap | blend_gain |
|---|---:|---:|---:|---:|---:|---:|
| stage03_init_e1 | 0.250 | 0.013 | 0.062 | 1.000 | 0.050 | 0.150 |
| e1_ippo_s1 | 0.000 | 0.000 | 0.113 | 1.000 | 0.000 | 0.113 |
| e1_pgct_s1 | 0.000 | 0.000 | 0.013 | 1.000 | 0.000 | 0.125 |
| e2_ippo_s1 | 0.250 | 0.675 | 0.825 | 1.000 | 0.250 | 0.263 |
| e2_pgct_s1 | 0.250 | 0.850 | 0.725 | 1.000 | 0.450 | 0.225 |
| stage04_pgct | 0.250 | 0.000 | 0.075 | 0.994 | 0.000 | 0.113 |

## Same − cross score (larger = teammates scored as more similar/compatible)

| Model | canonical_js | history_js | occupancy_js | goal_evidence | critic_gap | blend_gain |
|---|---:|---:|---:|---:|---:|---:|
| stage03_init_e1 | 0.021 | -0.043 | -0.021 | 0.918 | -1.991 | -1.083 |
| e1_ippo_s1 | 0.002 | -0.055 | -0.010 | 0.924 | -8.839 | -1.693 |
| e1_pgct_s1 | -0.033 | -0.049 | -0.022 | 0.928 | -7.371 | -1.259 |
| e2_ippo_s1 | -0.013 | 0.016 | 0.064 | 0.864 | -0.003 | -0.004 |
| e2_pgct_s1 | -0.007 | 0.019 | 0.062 | 0.861 | -0.001 | 0.001 |
| stage04_pgct | -0.026 | -0.036 | -0.029 | 0.892 | -8.196 | -1.385 |

## Goal evidence accumulated over episodes

| Model | episodes | pairs with evidence | recovery | AUROC |
|---|---:|---:|---:|---:|
| stage03_init_e1 | 1 | 6/6 | 1.000 | 1.000 |
| stage03_init_e1 | 2 | 6/6 | 1.000 | 1.000 |
| stage03_init_e1 | 5 | 6/6 | 1.000 | 1.000 |
| stage03_init_e1 | 10 | 6/6 | 1.000 | 1.000 |
| stage03_init_e1 | 20 | 6/6 | 1.000 | 1.000 |
| e1_ippo_s1 | 1 | 6/6 | 1.000 | 1.000 |
| e1_ippo_s1 | 2 | 6/6 | 1.000 | 1.000 |
| e1_ippo_s1 | 5 | 6/6 | 1.000 | 1.000 |
| e1_ippo_s1 | 10 | 6/6 | 1.000 | 1.000 |
| e1_ippo_s1 | 20 | 6/6 | 1.000 | 1.000 |
| e1_pgct_s1 | 1 | 6/6 | 1.000 | 1.000 |
| e1_pgct_s1 | 2 | 6/6 | 1.000 | 1.000 |
| e1_pgct_s1 | 5 | 6/6 | 1.000 | 1.000 |
| e1_pgct_s1 | 10 | 6/6 | 1.000 | 1.000 |
| e1_pgct_s1 | 20 | 6/6 | 1.000 | 1.000 |
| e2_ippo_s1 | 1 | 6/6 | 1.000 | 1.000 |
| e2_ippo_s1 | 2 | 6/6 | 1.000 | 1.000 |
| e2_ippo_s1 | 5 | 6/6 | 1.000 | 1.000 |
| e2_ippo_s1 | 10 | 6/6 | 1.000 | 1.000 |
| e2_ippo_s1 | 20 | 6/6 | 1.000 | 1.000 |
| e2_pgct_s1 | 1 | 6/6 | 1.000 | 1.000 |
| e2_pgct_s1 | 2 | 6/6 | 1.000 | 1.000 |
| e2_pgct_s1 | 5 | 6/6 | 1.000 | 1.000 |
| e2_pgct_s1 | 10 | 6/6 | 1.000 | 1.000 |
| e2_pgct_s1 | 20 | 6/6 | 1.000 | 1.000 |
| stage04_pgct | 1 | 6/6 | 1.000 | 1.000 |
| stage04_pgct | 2 | 6/6 | 1.000 | 1.000 |
| stage04_pgct | 5 | 6/6 | 1.000 | 1.000 |
| stage04_pgct | 10 | 6/6 | 1.000 | 1.000 |
| stage04_pgct | 20 | 6/6 | 1.000 | 1.000 |

## Critic premise per pair (mean over episodes)

| Model | 1-3 | 2-4 | 1-2 | 1-4 | 2-3 | 3-4 |
|---|---:|---:|---:|---:|---:|---:|
| stage03_init_e1 blend_gain | -0.3198 | -2.4267 | +0.8791 | -1.2072 | -0.2186 | -0.6145 |
| stage03_init_e1 donor_excess_mse | +2.5448 | +8.3970 | +1.8719 | +3.6853 | +1.1184 | +2.0675 |
| e1_ippo_s1 blend_gain | -2.3251 | -3.1532 | -1.6834 | -1.2955 | -1.0908 | -0.1164 |
| e1_ippo_s1 donor_excess_mse | +10.3715 | +13.2285 | +6.5323 | +6.4081 | +4.4285 | +0.5052 |
| e1_pgct_s1 blend_gain | -1.8128 | -2.5508 | -1.4022 | -1.0283 | -1.2593 | -0.0023 |
| e1_pgct_s1 donor_excess_mse | +9.6188 | +10.7293 | +5.6139 | +5.2976 | +5.5714 | +0.0541 |
| e2_ippo_s1 blend_gain | +0.0105 | -0.0121 | +0.0008 | -0.0026 | +0.0079 | +0.0061 |
| e2_ippo_s1 donor_excess_mse | -0.0090 | +0.0474 | +0.0112 | +0.0316 | +0.0018 | +0.0013 |
| e2_pgct_s1 blend_gain | -0.0067 | -0.0179 | +0.0116 | +0.0001 | -0.0262 | -0.0400 |
| e2_pgct_s1 donor_excess_mse | +0.0215 | +0.1026 | -0.0063 | +0.0442 | +0.0746 | +0.1445 |
| stage04_pgct blend_gain | -2.4805 | -2.2576 | -1.6113 | -0.7024 | -1.7841 | +0.1599 |
| stage04_pgct donor_excess_mse | +11.5273 | +10.4763 | +6.2161 | +5.2157 | +6.7347 | -0.2307 |
