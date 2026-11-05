# PGCT: latent team identification and selective critic access in decentralized MARL

PGCT (Probe-Gated Critic Transfer) lets agents that learn **independently from individual rewards** find
their latent teammates by comparing how their own reward predictors respond to common reward probes built
from real experience (`Q_ij = E_i ∪ E_j`), and gives approved, physically met teammates access to each
other's critics through **Max-GAE**. No team labels, shared rewards, parameter sharing or centralized
critic are used by the method. PGCT-D2 is the current reference variant.

This repository is organised as one shared algorithm and independent experiment modules:

| folder | content |
|---|---|
| [`PGCT/`](PGCT) | `pgct_core`: environment-independent gate, Max-GAE, recurrent IPPO, generic trainer, metrics; `pgct_runner`: shared plans, CLI, W&B, videos, aggregation for LBF/VMAS |
| [`RWARE/`](RWARE) | multi-robot warehouse experiments (the original study; behaviour unchanged by the split) |
| [`Overcooked/`](Overcooked) | 4-agent, 2-hidden-team Overcooked-style kitchen experiment (boundary case, D138/D141) |
| [`LBF/`](LBF) | multi-team Level-Based Foraging: second main environment (D141, D142) |
| [`VMAS/`](VMAS) | VMAS Team Discovery: continuous-physics check (optional `vmas`, GPLv3) |
| [`docs/`](docs) | [ARCHITECTURE](docs/ARCHITECTURE.md), [MIGRATION](docs/MIGRATION.md), [ALGORITHM](docs/ALGORITHM.md), [ENVIRONMENTS](docs/ENVIRONMENTS.md), research context and history |
| [`scripts/`](scripts) | `run_tests.ps1`, `run_experiment.ps1` / `aggregate_experiment.ps1` (LBF, VMAS), `migrate_local_layout.ps1` |

Research records stay at the root: `RESEARCH_STATE.md`, `DECISIONS.md`, `EXPERIMENTS.md`,
`CONVERSATION_LOG.md`, `paper/`, `presentation/` (see `AGENTS.md`, `CLAUDE.md`).

## Quick start (Windows PowerShell, repository root)

```powershell
$env:PGCT_PYTHON = "C:\Users\95101\anaconda3\envs\marl-policy-sharing\python.exe"   # or your env
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

`pgct_core` needs only `numpy` and `torch`; RWARE additionally needs `gymnasium`, `networkx`, `pyglet<2`
and `wandb`. Installation is optional (`pip install -e .` installs `pgct_core` and `pgct_overcooked`);
otherwise every entry point adds `PGCT/` to `sys.path` itself.

**RWARE** (run from `RWARE/`, all previous commands unchanged): see [`RWARE/README.md`](RWARE/README.md).

```powershell
cd RWARE
python latent_team\train_from_scratch.py --env-variant b6_corner_indiv_comm5_pen005 --modes ippo pgct_max --reward-probe-ac --ac-gate v2 --ac-distance action_max --ac-config latent_team\acv2_nomirror.json --seeds 1 --timesteps 64 --smoke-test --output-dir runs\smoke_pgct_d2_check
```

**Overcooked**: see [`Overcooked/README.md`](Overcooked/README.md).

```powershell
powershell -ExecutionPolicy Bypass -File Overcooked\scripts\run_overcooked.ps1 -Name oc_smoke -Seeds 1 -Smoke
powershell -ExecutionPolicy Bypass -File Overcooked\scripts\aggregate.ps1 -Name oc_smoke
```

The Overcooked queue launcher tracks new jobs in W&B project `pgct-overcooked` by default (one run per
method/seed, grouped by `-Name`). Pass `-NoTrack` for local-only runs or `-WandbMode offline` to sync later.

**LBF** and **VMAS**: see [`LBF/README.md`](LBF/README.md), [`VMAS/README.md`](VMAS/README.md) and
[`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) (why these, why not MetaDrive / MAgent2).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Experiment LBF -Name lbf_smoke -Seeds 1 -Smoke -NoTrack
powershell -ExecutionPolicy Bypass -File scripts\aggregate_experiment.ps1 -Experiment LBF -Name lbf_smoke
```

## What is shared and what is environment specific

Shared (`PGCT/pgct_core`): reward predictors, probe sets, pooled/action-max response distance, EMA,
maturity, approval, rollout-start snapshot, donor rules, Max-GAE, GAE, recurrent IPPO, the generic trainer
and identification metrics. Environment specific: observations and their public probe projection,
communication contacts, rewards, the evaluation-only team labels, trainers' environment glue and configs.
Adding another environment means writing an adapter and a projection
([ARCHITECTURE § Adding another MARL environment](docs/ARCHITECTURE.md#adding-another-marl-environment)).

## License

MIT (see `LICENSE`; the RWARE environment is © Filippos Christianos, see `RWARE/CITATION.cff`).
