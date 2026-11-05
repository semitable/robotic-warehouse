# VMAS Team Discovery for PGCT

Continuous-physics environment (D142), a multi-team variant of the VMAS `discovery` scenario
(Bettini et al. 2022, [proroklab/VectorizedMultiAgentSimulator](https://github.com/proroklab/VectorizedMultiAgentSimulator),
`vmas` 1.5.2). Holonomic point-mass robots with LIDARs move in a 2×2 arena; a target is covered when at least
`agents_per_target` agents are within the covering range, every covering agent is rewarded individually, and
covered targets respawn elsewhere. This variant adds **hidden teams**:

- every agent belongs to a hidden team and every target has a type (one per team);
- covering ignores teams (any agents count), but a covering agent is paid only for targets of its own type;
- each agent has one LIDAR per target type (types are visible, like coloured targets) plus an agent LIDAR;
  which type pays is never observed; `oracle_team_ids()` is for the Oracle baseline and evaluation only.

**VMAS is GPLv3** and an optional dependency (`pip install vmas==1.5.2`, which pulls `gym` 0.26 and
`pyglet<=1.5.27`; only `vmas`, `gym` and `gym-notices` are needed here because videos are drawn with Pillow).
Nothing from VMAS is vendored; `pgct_vmas/scenario.py` imports it.

## Information separation (`pgct_vmas/adapter.py`)

| kind | content |
|---|---|
| actor observation | VMAS discovery's observation: own position, own velocity, LIDAR distances per target type, agent LIDAR (4 + 2×12 + 12 = 40) |
| critic input | identical to the actor observation |
| private reward | own covering reward for own-type targets |
| public probe view | velocity and LIDARs; the absolute position is dropped. LIDAR rays point in fixed world directions for every agent, so the same event looks the same to everyone. float32 |
| communication | Euclidean distance ≤ `comm_range` (0.6) |
| evaluation only | `oracle_team_ids`, per-team summaries, team badges in videos |

Actions: VMAS discrete actions (9): index a → x choice a // 3, y choice a % 3 (0 stay, 1 −u, 2 +u).

VMAS keeps one random state per process for all its environments. The adapter swaps in its own copy around
every VMAS call, so evaluation and video environments never shift the training environment's random stream
(`tests/test_discovery.py`).

## Configurations (fixed before any PGCT result, D142)

| file | discovery | purpose |
|---|---|---|
| `configs/main_4p2t.json` | 4 agents (2+2), 3 targets per type, one agent covers a target, 100-step episodes | main |
| `configs/coop_4p2t.json` | `agents_per_target` 2 (the VMAS discovery default) | coordination variant; teammates are often paid in the same step |

PPO, gate and Max-GAE as in LBF (relative reward-event rule). Budget 2M steps.

**Cost**: one VMAS environment on one CPU thread runs about 140 environment steps/s here with random actions
(LIDAR ray casting is about 40% of a step) and about 84 steps/s during IPPO training (E96). VMAS is built for
thousands of batched environments; the PGCT trainer steps one environment. A 2M-step job therefore takes about
6–7 hours; vectorised rollouts would need a trainer change.

## Commands (repository root, PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 -Only VMAS
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Experiment VMAS -Name vmas_smoke -Seeds 1 -Smoke -NoTrack
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\run_experiment.ps1' -Experiment VMAS -Name vmas_main -Seeds 1,2,3 -MaxConcurrent 3"
powershell -ExecutionPolicy Bypass -File scripts\aggregate_experiment.ps1 -Experiment VMAS -Name vmas_main
```

Direct entry points (with `VMAS;PGCT` on `PYTHONPATH`): `python -m pgct_vmas.train|evaluate|video|aggregate`.
Runs go to `VMAS/runs/<name>/` (git-ignored); W&B project `pgct-vmas`.

## Tests (`VMAS/tests`, 12; skipped without `vmas`)

Observation layout and probe view, own-type-only pay and respawn, the two-agent variant with a cross-team
helper, seeded reproducibility and RNG isolation from other VMAS environments, physics and observations
independent of team labels, contact, every method runs with shared initial weights, label rule, exact
reduction to IPPO / Oracle, and the plan → train → video → evaluate → aggregate command lines.
