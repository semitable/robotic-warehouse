# Environments for PGCT: what fits and why (2026-11-06, D141, D142)

PGCT assumes agents that learn independently from **individual** rewards, latent teams defined as agents with
the **same reward function**, and teammates whose **reward moments overlap** (similar jobs), so that one agent's
reward predictor responds to another's reward events (identification) and one agent's critic values the other's
situations sensibly (critic transfer). Lessons from Overcooked (D135–D140): the public probe view must look the
same to every agent, every teammate needs its own reward events, and a division of labour (cook vs server) breaks
both premises.

Selection criteria: individual reward is the benchmark's own setting; teammates do the same kind of job; teams can
be hidden as reward functions without leaking identity into observations; IPPO does not trivially solve the task
(room for critic transfer); viewpoint-consistent public views; fixed agent population; runs on Windows with
PyTorch at reasonable speed.

## In the repository

| environment | role | status |
|---|---|---|
| RWARE (`RWARE/`) | main; individual reward is RWARE's own default | running (rware-main-amax-v2) |
| Multi-team LBF (`LBF/`) | second main environment; LBF's own individual reward, same benchmark family as RWARE | built 2026-11-06, tests 22, pilot only |
| VMAS Team Discovery (`VMAS/`) | continuous-physics generality check | built 2026-11-06, tests 12, smoke/pilot only |
| Overcooked-style kitchen (`Overcooked/`) | boundary case: division of labour weakens identification | built 2026-11-05; kept, not deleted |

## Reviewed and not built

**MetaDrive** (multi-agent roundabout / intersection / tollgate / bottleneck / parking lot).
- The agent population is not fixed: terminated vehicles are replaced by newly spawned ones during an episode
  ([MetaDrive RL environments](https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html)).
  PGCT keeps one learner, one reward predictor and one critic per persistent agent, so respawned vehicles break
  identification and critic transfer. (Disabling respawn ends the episode once all vehicles are gone.)
- No natural hidden teams: every vehicle is self-interested with its own route; "same destination" teams would
  be an invented structure.
- Heavy: Panda3D 3D simulation, a 55 MB wheel plus asset download, Python < 3.12, continuous control by default.
- Verdict: not suitable.

**MAgent2** (battle, adversarial pursuit, gather, combined arms, ...).
- Team identity is in the observation: battle observes `my_team_presence`, `my_team_hp`, `other_team_presence`,
  `other_team_hp` ([MAgent2 battle](https://magent2.farama.org/environments/battle/)). Identification is trivial.
- Teams are adversarial (team vs team), not latent cooperative groups; agents are killed during an episode, so
  the population shrinks.
- Its point is hundreds of agents; PGCT keeps per-agent networks and pairwise probe comparisons.
- A hidden-team variant would need a custom MAgent2 configuration with per-agent reward rules, losing the
  benchmark character. Windows wheels exist for 0.3.4 (Python 3.10–3.13).
- Verdict: not suitable as is; not built.

Rejected earlier (D79, D141): SMAC/SMACv2 (fixed teams, shared reward), Melting Pot (no Windows support),
MPE simple_spread (shared, dense distance reward), Hanabi (fully cooperative, no hidden teams).

## Design rules for the new environments

- Physics ignores teams; only the reward function differs (the RWARE principle: anyone can carry any shelf).
- Fixed before any PGCT result: environment, reward, budgets, main and variant configs (D138 principle; D142).
- The same gate values for every new environment, including the relative reward-event rule (D139).
- Co-occurring teammate rewards (LBF co-loading, VMAS two-agent covering) favour a reward-correlation baseline;
  each environment has a variant without them (LBF `ind`, VMAS main) and that baseline must be reported.
