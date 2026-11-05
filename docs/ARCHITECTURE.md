# Architecture

## Repository layout (after the 2026-11 split)

```
<repo>/                         (git repository; cloned as robotic-warehouse/)
├── PGCT/                       shared, environment-independent algorithm
│   ├── pgct_core/              importable package (no RWARE / Overcooked imports)
│   │   ├── interfaces.py       ProbeProjection, EnvAdapter (the six kinds of information)
│   │   ├── reward_gate.py      reward predictors, Q_ij = E_i ∪ E_j, distance, EMA, maturity, approval, snapshot
│   │   ├── max_gae.py          donor rules, donor critics, Max-GAE (D2), GAE
│   │   ├── networks.py         per-agent GRU actor-critic (MLP encoder)
│   │   ├── ppo.py              recurrent IPPO update
│   │   ├── trainer.py          generic training loop + methods ippo/oracle/all/pgct_d2/pgct_id_only/random_gate
│   │   └── metrics.py          identification / learning metrics (evaluation side)
│   └── tests/                  core tests + toy environment (template for new environments)
├── RWARE/                      existing warehouse experiment (moved, behaviour unchanged)
│   ├── rware/                  environment package (setup.py → `rware`)
│   ├── examples/               original recurrent IPPO/PGCT trainer and earlier runners (frozen)
│   ├── latent_team/            experiment runners; reward_probe_ac_v2.py / max_gae.py are RWARE adapters over pgct_core
│   ├── tests/                  RWARE tests + pgct_core equivalence tests + legacy/ frozen pre-split copies
│   ├── runs/, output/          RWARE checkpoints and analysis outputs (runs/ is git-ignored)
│   └── README.md
├── Overcooked/                 new experiment
│   ├── pgct_overcooked/        kitchen simulator, adapter, train/evaluate/aggregate entry points
│   ├── configs/                main_4p2t.json (4 agents, 2 hidden teams), secondary_2p1t.json
│   ├── scripts/                PowerShell launchers
│   ├── tests/
│   └── docs/EXPERIMENT_DESIGN.md
├── docs/                       ARCHITECTURE / MIGRATION / ALGORITHM + research context and history
├── scripts/                    run_tests.ps1, migrate_local_layout.ps1
├── pyproject.toml              installs pgct_core and pgct_overcooked (optional; paths are bootstrapped)
└── research records            RESEARCH_STATE.md, DECISIONS.md, EXPERIMENTS.md, CONVERSATION_LOG.md, paper/, ...
```

Dependency direction: `RWARE → pgct_core ← Overcooked`. `pgct_core` imports only numpy and torch.

## The pre-split architecture (audit, 2026-11-05, commit 122615f)

The algorithm lived inside the RWARE fork:

- `examples/train_recurrent_ippo_consensus.py` (≈4,700 lines): recurrent IPPO trainer for RWARE, with
  RWARE diagnostics inside the rollout loop, plus the *earlier* PGCT variants (policy-JS probe gate
  `graph_mode=policy`, critic peer-distillation loss λ=.01, consensus). Not modified by the split.
- `latent_team/train_from_scratch.py`: plans jobs (one manifest per experiment, same initial weights per
  seed) and runs them by monkeypatching trainer functions inside scoped context managers:
  `video_retention.install` → `exp_b_oracle_consensus.matched_transfer` (transfer log, warm-up hold) →
  `reward_probe_ac[_v2].install` (coordinate masking, gate observation) → `max_gae.install` (contacts,
  Max-GAE advantage injection).
- **Authoritative PGCT-D2** = mode `pgct_max` with `--reward-probe-ac --ac-gate v2 --ac-distance action_max
  --ac-config latent_team/acv2_nomirror.json` (`latent_team/run_main_amax_v2.ps1`). The gate is
  `latent_team/reward_probe_ac_v2.py` with `contact_snapshot=True` (set by the planner for
  `pgct_max`/`oracle_mature_max`), the critic access is `latent_team/max_gae.py` (D2: η .5, shrink).
- Earlier / other implementations kept for record: v1 exact-match reward gate (`reward_probe_ac.py`,
  `--ac-gate v1`), the policy-JS gate and critic distillation (`pgct`, `oracle`, `all`, `wrong` modes),
  `random_gated.py` (budget-matched random distillation), `probes_v2.py` (probe bank v2), MAPPO/SePS/HetDPS
  (`marl_baselines.py`), Max-GAE D1 (`oracle_maxd1`).

### What is environment independent

| component | pre-split location | now |
|---|---|---|
| reward predictor, ring buffers, event oversampling | reward_probe_ac_v2.py | pgct_core.reward_gate |
| probe set E_i ∪ E_j (+ optional augmentation hook) | reward_probe_ac_v2.py | pgct_core.reward_gate |
| pooled / action_max distance, EMA, rounds, maturity, approval, snapshot | reward_probe_ac_v2.py | pgct_core.reward_gate |
| audit (same/cross accounting for logs) | reward_probe_ac_v2.py | pgct_core.reward_gate |
| donor rules, donor critic evaluation, uplift, Max-GAE blend, EV, warm-up mask | max_gae.py | pgct_core.max_gae |
| GAE | trainer `compute_gae` | pgct_core.max_gae.compute_gae (identical copy; RWARE keeps using the trainer's) |
| recurrent actor-critic (MLP), IPPO update | trainer | pgct_core.networks / ppo (bit-identical, tested) |

### What stays RWARE specific

- Public RWARE view (`reward_probe_ac.public_view`: channel selection of the 20-ego/13-channel semantic
  observation), RWARE mirror (`mirror_view`/`mirror_action`), 5 actions, coordinate masking wrapper.
- `ContactEnv` (agent x/y from the warehouse), trainer monkeypatch adapters (`install`), the RWARE trainer,
  environment candidates, request/goal mechanics, reward shaping, video retention, W&B logging, runners.

## pgct_core in one update

```
rollout start ─ gate.snapshot()                (frozen probes / maturity / predictor version)
rollout       ─ actions from own GRU policies; adapter.step → own rewards
              ─ gate.observe(public views, actions, own rewards)   (each predictor sees only its own data)
              ─ contact |= adapter.contact_matrix()
update        ─ gate(update, contact, active = update > warmup)  → compare contact pairs, then refit
              ─ donors = rule(method; contact, gate.approved(), [labels only for oracle])
              ─ donors? Max-GAE actor advantages (own returns for the critic) : plain IPPO advantages
              ─ ppo_update; logs (metrics.jsonl, gate.jsonl), eval, checkpoints
```

RWARE runs the same gate/Max-GAE objects inside its own trainer through `latent_team/*install`; the
Overcooked experiment uses `pgct_core.trainer.run_training`.

## Adding another MARL environment

1. **Adapter** implementing `pgct_core.interfaces.EnvAdapter`: `n_agents`, `obs_dim`, `n_actions`,
   `reset(seed)`, `step(actions) -> (obs [N, D], rewards [N], terminated, truncated, info)`,
   `contact_matrix()`, `oracle_team_ids()`; optional `team_summary(episodes, team_ids)` for evaluation.
   Observations are each agent's local view; all agents' critic inputs must share one space (Max-GAE runs a
   donor critic on the receiver's inputs). `info` must not carry team labels; `info["events"]` may carry
   per-agent counters (logged as `episode_<key>_per_agent`, `eval_<key>_total`).
2. **Projection** implementing `ProbeProjection`: `view_size`, `n_actions`, `view_dtype`,
   `public_view(observation)`. Include only what any nearby agent could see; exclude team identity,
   coordinates and private task information; make the semantics identical for every agent (egocentric views
   are fine). Leave `mirror` unimplemented (`NoMirror`) unless a symmetry augmentation is justified.
3. **Configs**: reuse `PPOConfig`, `RewardGateConfig`, `MaxGAEConfig`, `TrainerConfig`; keep gate values
   equal to RWARE PGCT-D2 unless there is an environment-specific reason, and record it.
4. **Tests**: mechanics; public views independent of hidden labels; a scripted identification check (as in
   `Overcooked/tests/test_identification.py`); every method runs; non-oracle methods never read labels.

`PGCT/tests/toy_env.py` is a complete 60-line example of steps 1–2.

## Environment-specific configuration

| setting | where | RWARE | Overcooked |
|---|---|---|---|
| environment, rewards, layout | env config | `env_candidates.py`, trainer `env_kwargs_json` | `configs/*.json` → `kitchen` |
| observation / probe radius, comm range | adapter | sensor range 5, probe radius 2, `communication_range` 3/5 | `adapter`: 3 / 1 / 3 |
| PPO, gate, Max-GAE | shared dataclasses | RWARE trainer config + `ACV2Config` + `MaxGAEConfig` | same dataclasses (`ppo`, `gate`, `max_gae`) |
| encoder | network | semantic CNN (RWARE trainer) | MLP (`pgct_core.networks`) |
