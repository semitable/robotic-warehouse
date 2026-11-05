# Multi-team Level-Based Foraging (LBF) for PGCT

Second main environment after RWARE (D141, D142). Agents forage on a grid; food has a level and is collected
when the levels of the agents loading it together reach that level. The original LBF (Christianos et al. 2020,
[semitable/lb-foraging](https://github.com/semitable/lb-foraging), `lbforaging` 2.0.0, MIT) already pays
**individual** rewards: each loader gets its level-weighted share. This variant adds **hidden teams**:

- every agent belongs to a hidden team, every food has a type (one per team), shown in its own observation layer;
- the physics ignores teams (anyone can help lift any food), but a loader is paid only for food of its own team;
- LBF's food-level rule is applied per team (max level = sum of that team's three lowest player levels), and the
  reward normaliser is the team's spawned food (each team can earn 1 per episode);
- observations never say which food type pays; `oracle_team_ids()` is for the Oracle baseline and evaluation only.

With one team the simulator reproduces `lbforaging` 2.0.0 step for step (positions, food, rewards, episode end,
grid observations; `tests/test_lbforaging_equivalence.py`). Two deliberate differences: simultaneous loaders are
processed in agent-index order (no effect under LBF's food spacing), and the time limit is truncation, not
termination.

Why LBF fits PGCT (D141): individual reward is the benchmark's own setting; teammates do the same job, so their
reward moments overlap (no Overcooked-style division of labour); same author and benchmark family as RWARE
(SEAC, EPyMARL); fast (about 22,000 environment steps/s for 4 agents with random actions on this PC).

## Information separation (`pgct_lbf/adapter.py`)

| kind | content |
|---|---|
| actor observation | LBF grid observation: a (2s+1)×(2s+1) window (sight s) of agent levels (self at the centre), one food-level layer per type, access |
| critic input | identical to the actor observation |
| private reward | own LBF share for own-team food only |
| public probe view | centre (2r+1)×(2r+1) crop of all layers, r = `probe_radius` (2: a food next to the agent and every cell next to that food). LBF has no headings, so every agent sees the same event alike. uint8 |
| communication | Chebyshev distance ≤ `comm_range` |
| evaluation only | `oracle_team_ids`, per-team summaries, the team badge in videos |

## Configurations (fixed before any PGCT result, D142)

| file | foraging | purpose |
|---|---|---|
| `configs/main_4p2t.json` | 12×12, 4 agents (2+2), 3 food per team, sight 2, 50 steps, LBF level rule per team | main: some food needs both teammates |
| `configs/ind_4p2t.json` | every food level 1 (LBF `-ind`) | no co-loading: teammates are never paid in the same step, identification rests on probes alone |
| `configs/coop_4p2t.json` | every food at the team maximum (LBF `-coop`) | co-loading always needed: report the reward-correlation baseline next to PGCT |

PPO, gate and Max-GAE values follow the RWARE/Overcooked PGCT-D2 runs, with the relative reward-event rule (D139)
fixed for new environments. Budget 5M steps; evaluation every 50 updates with 20 deterministic episodes.

## Commands (repository root, PowerShell)

```powershell
$env:PGCT_PYTHON = "C:\Users\95101\anaconda3\envs\marl-policy-sharing\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 -Only LBF
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Experiment LBF -Name lbf_smoke -Seeds 1 -Smoke -NoTrack
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\run_experiment.ps1' -Experiment LBF -Name lbf_main -Seeds 1,2,3 -MaxConcurrent 3"
powershell -ExecutionPolicy Bypass -File scripts\aggregate_experiment.ps1 -Experiment LBF -Name lbf_main
```

Direct entry points (with `LBF;PGCT` on `PYTHONPATH`): `python -m pgct_lbf.train|evaluate|video|aggregate`.
Runs go to `LBF/runs/<name>/` (git-ignored). The six methods are those of `pgct_core.trainer.METHODS`; the queue
tracks W&B project `pgct-lbf` and records a greedy play video at every evaluation (`-NoTrack`, `-NoVideo`).

## Tests (`LBF/tests`, 22)

LBF rules (spacing, level rule per team, LOAD shares, team-only pay, penalty, collisions, swaps, end of
episode), observation layers and probe crop, observations independent of team labels, contact, the comparison
with `lbforaging` (skipped unless `pip install lbforaging==2.0.0 --no-deps`), a scripted identification check
(the gate with the main-config values approves exactly the hidden teammates), every method runs with shared
initial weights, non-oracle methods never read labels, gate methods reduce exactly to IPPO / Oracle, and the
plan → train → video → evaluate → aggregate command lines.

## Known limits

- Food levels depend on each team's player levels (LBF's rule applied per team). This is physical information
  available to everyone, not a label, but it is one more way teams shape an episode.
- Co-loading (main, coop) pays teammates in the same step, which helps a reward-correlation baseline; `ind`
  removes it by construction.
- Food does not respawn (LBF): an agent sees at most three own-team reward events per episode.
