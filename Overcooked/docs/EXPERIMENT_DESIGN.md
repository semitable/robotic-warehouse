# Overcooked experiment design

**Question.** Does reward-response-based latent team identification (PGCT-D2) generalise from RWARE to a
different cooperative domain, and does approved critic access (Max-GAE) help there?

Status (2026-11-05): implemented and tested; smoke and short sanity runs only. **No learning or
identification result exists yet**; nothing below is a result claim.

## 1. Choice of simulator

Investigated (2026-11-05):

| implementation | agents | teams / recipes | rewards | API / deps | verdict |
|---|---|---|---|---|---|
| [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) (MIT) | built and documented for 2 (human-AI coordination); some featurisation comments mention N | one shared order list, no per-team recipes | shared team reward with per-agent sparse/shaped breakdowns in the step info | own MDP/env API, pygame; PyPI releases infrequent, DRL code deprecated | would need deep changes for 4 agents with team-specific recipes |
| JaxMARL Overcooked / OvercookedV2 ([paper](https://arxiv.org/abs/2311.10090)) | 2 | fully cooperative | shared | JAX (not installed; the PGCT code is PyTorch) | mismatch of framework, 2 agents |
| Melting Pot *collaborative cooking* ([PettingZoo port](https://meltingpot-pz.pages.dev/substrates/collaborative_cooking/)) | small fixed groups | one tomato soup | shared | dmlab2d (no Windows support) | cannot run here |
| N-player research extensions (e.g. [arXiv 2506.17560](https://web3.arxiv.org/pdf/2506.17560)) | N | not aimed at hidden teams | — | research code | not a maintained dependency |

None supports a **four-agent, two-team** kitchen with **team-specific recipes** and **individual rewards**,
which the scientific question needs. Forcing four agents into a two-player benchmark would be invalid, and
a two-agent cooperative task cannot test latent multi-team identification (every pair is a teammate pair).

**Decision**: a compact, original N-agent kitchen (`pgct_overcooked/env/kitchen.py`, ≈250 lines, no
third-party code or layouts) that follows the common Overcooked research mechanics — ingredient and dish
dispensers, counters, pots that cook three ingredients (cooking starts automatically when full), serving
windows; actions stay / four moves (turn, and move if free) / interact with the faced cell; simultaneous moves
with conflicts cancelled — and adds hidden teams with team-specific recipes. Trade-offs: results are not
directly comparable with published Overcooked-AI numbers; the mechanics are simple and fully unit-tested;
the simulator runs on CPU in pure Python/numpy (rollout collection ≈100 steps/s for 4 agents including
observation encoding and the four policy forward passes, measured on this PC). A standard two-agent cooperative configuration (`secondary_2p1t.json`) is included as a
secondary benchmark only.

## 2. Main environment (`configs/main_4p2t.json`)

- Layout `two_team_kitchen` (11×7): left-right mirror symmetric, the mirror swaps onion and tomato
  dispensers; four shared pots, two dish dispensers, two serving windows; agents start in the four corners.
- 4 agents, 2 hidden teams of 2; team 0 cooks **onion soup**, team 1 **tomato soup**. The assignment is a
  random permutation drawn from the run seed (`team_seed: null`), so agent indices carry no team meaning.
- Horizon 400, cook time 10.

### Individual rewards

An agent is paid only for **its own** actions that advance **its own** team's recipe:

| event (by agent i) | reward to i | to everyone else |
|---|---|---|
| serve a soup of its team's recipe | +1.0 | 0 |
| serve any other soup (other team's, mixed) | 0 | 0 |
| put its team's ingredient into a pot | +0.1 (shaping) | 0 |
| take its team's soup out of a pot | +0.2 (shaping) | 0 |

Nothing is summed or shared, so teammates have equal reward *functions* but independent reward *streams*
(as in RWARE's individual setting, D81). Teammates can still take different roles (one cooks, one serves); the
team reward function, not the action, defines a team. Teams interact physically through shared pots: putting
the other ingredient into a pot ruins it (mixed soup, worth nothing), which creates natural cross-team
interference. Reward events for the gate are `r >= 0.5`, i.e. own-recipe deliveries; shaping never counts as
an event.

## 3. Information given to PGCT

| | content |
|---|---|
| actor observation | egocentric 7×7 window (radius 3) × 32 channels + own facing + own held item (1,579 values); no coordinates, team, recipe or agent identity |
| critic input | the same vector (shared space for all agents) |
| private reward | own scalar reward only |
| public probe view | centre 3×3 crop (radius 1) of the same channels + own facing/held (299 values, binary, uint8) |
| communication | Chebyshev distance ≤ 3, any step of the rollout |
| evaluation only | `oracle_team_ids()`, per-team summaries |

Comparability of probes across agents: the view is egocentric and contains only physical state, so an
identical local transition has an identical encoding whoever experienced it. The recipe is never observed:
each agent has its own network, so its objective is learned from its reward. Unlike RWARE (which must drop
private request channels), nothing has to be removed from the observation.

**Correction (2026-11-05, E88).** "Identical encoding" does not make the *same kind of event* comparable.
The view is egocentric but not rotated, so serving at the west window (facing west, window on the left of
the crop) and at the east window (facing east, window on the right) are different inputs. In the 500k pilot
(seed 1) the two tomato teammates served at opposite windows; each predictor answered its teammate's
deliveries with 0.04 (own: 1.0), so the same-team distance stayed ≈ 0.92 and no pair was ever approved. The
onion team's cook made only 2 deliveries and never matured (abstention by design). PGCT-D2 therefore stayed
bit-identical to IPPO. Nothing was changed in the running experiments.

**Probe v2 (D134, `configs/main_4p2t_probe_v2.json`, `adapter.probe_frame = "facing"`).** Each probe
transition is expressed in the agent's heading frame before the step: the 3×3 crop is rotated so the heading
points up, other agents' heading channels and the agent's own heading bits become relative, and move actions
become forward/right/back/left. A rotation, not a mirror: the two cells beside a station keep their side, so
west and east window deliveries still differ in the two front corners (dish dispenser vs counter). Only the
gate's public view changes; observations, rewards, PPO, gate values and Max-GAE are as in v1, so `ippo`,
`oracle` and `all` give bit-identical runs under both configs (tested). Checks before training:
- scripted opposite-window teammates (`tests/test_probe_v2.py`): v1 same-team EMA .53/.54 (not approved), v2
  .32/.27 (approved exactly the teammates), cross-team ≥ .92 in both;
- replay of the 500k pilot policies with identical experience for both frames: see E90 in `EXPERIMENTS.md`.
The cook who never serves still abstains in v2: identification needs each agent's own reward events.

v2 in training (`oc_pilot_500k_probe_v2`, seed 1, 0.5M): same-team EMA 0.92 → 0.58, cross-team 0.82 / 1.00,
still no approval (τ = 0.5), so PGCT-D2 again equalled IPPO.

**Probe v3 (D135, `configs/main_4p2t_probe_v3.json`, `probe_frame = "facing_cross"`).** The v2 frame restricted to
the agent's cell and its four neighbours; the four diagonal cells (where the west and east windows differ: dish
dispenser vs counter) are dropped. Serving at either window becomes the same input
(`docs/img/probe_views_v1_v2_v3.png`, made by `docs/make_probe_view_figure.py`). Everything that decides an
Overcooked reward is in the front cell and the agent's own held item. Scripted opposite-window teammates:
same-team EMA 0.001 / 0.004, cross-team 0.96–0.97 (v2: 0.32 / 0.27). Policy-replay results: E92.

**Reward-event rule (D138–D139).** With the absolute rule (event ⇔ r ≥ 0.5) only deliveries are events, so an
agent that cooks but never serves never matures. The environment's rewards are frozen at the current version
(D138: not changed to suit PGCT). Instead the gate uses the scale-free relative rule shared with RWARE
(`configs/main_4p2t_probe_v3_rel.json`): an event is a positive own reward ≥ 0.1 × the largest own reward so far
(`docs/ALGORITHM.md`, "Reward-event rule"). This brings ingredient and soup events in for every agent. It does not
solve the contrast problem (agents that never handle the other team's items cannot tell the teams apart; E93).

Critic compatibility: all critic inputs live in one space with identical semantics, so a donor critic
`V_j` can be run on the receiver's history. `V_j` estimates returns under **j's** reward function; for a true
teammate this is the same function (the case Max-GAE is meant for), for a wrongly approved agent it is not,
which is exactly what the identification metrics and the `all`/`random_gate` controls measure. Max-GAE was
not modified.

## 4. Methods (identical environment, budget, seeds, evaluation)

`ippo`, `oracle` (labels: baseline only), `all`, `pgct_d2`, `pgct_id_only`, `random_gate` — see
`docs/ALGORITHM.md` §9. Gate and Max-GAE values are the RWARE PGCT-D2 main-run values (action_max, no mirror,
rollout-start snapshot, τ .5, β .35, 3 rounds, n_min 8, η .5 shrink, warm-up 100 updates); they were not tuned
on Overcooked. PPO values follow the RWARE reference (rollout 1024, lr 2e-4, entropy .05) with the MLP encoder.
Same seed ⇒ same initial weights for every method (tested).

## 5. Metrics (logged per update; aggregated by `pgct_overcooked.aggregate`)

- Identification (labels on the evaluation side): same/different-team EMA distances, AUROC of −distance,
  final precision/recall/F1 and false-approval rate of the approval matrix, time-averaged false-approval
  rate and recall, first env step with a correct and with a wrong approval, gate stability (fraction of
  unchanged updates) and approval flips, mature agents, approved arcs (same/cross).
- Learning: evaluation return (mean and per agent), own/other deliveries per agent and per team, AUC of the
  evaluation curve (sample efficiency), training episode returns, multi-seed mean ± std.
- Critic access: donor arcs per update (same/cross team), receivers, Max-GAE `mean_b`, `select_frac`,
  `adv_shift`, `eta_eff`, critic explained variance per agent.
- Cost: probe exchanges and approximate bytes (`probes × (2·view + 1)` + float32 responses), critic bytes
  (`donor arcs × critic parameters × 4`), wall time of rollout / gate (predictor training + comparison) /
  Max-GAE (donor critic evaluation) / PPO, steps per second, peak RSS.

Field names match RWARE where they exist (`max_gae_*`, `ac_v2_*`).

## 6. Built-in checks (tests)

- Mechanics, rewards per team, collisions, horizon (`tests/test_kitchen.py`).
- Observations and public views are identical under different hidden team assignments.
- **Scripted identification**: when every agent sometimes serves each soup type, the unchanged gate approves
  exactly the true teammates (AUROC 1, F1 1, same-team distance < .2, cross-team > .8;
  `tests/test_identification.py`). This shows the probe representation carries the needed information; it is
  not evidence that learning agents will generate such experience.
- Every method trains; identical initial weights; closed gate ⇒ `pgct_d2`/`pgct_id_only`/`random_gate` give
  IPPO's exact weights; oracle approvals injected ⇒ `pgct_d2` gives `oracle`'s exact weights; non-oracle
  methods never read labels (`tests/test_training.py`).

## 7. Threats to validity and open choices

- **Experience coverage.** A predictor answers probes from the other team's events by extrapolation if the
  agent never experienced them (the RWARE limitation E85/D129). Early random exploration produces some
  wrong-soup deliveries; trained specialists may not. The cross-team distance must be monitored over time.
- **Sparse events.** Gate maturity needs 20 own deliveries; until then there is no identification and no
  critic access. Time to first maturity is part of the result.
- **Learning difficulty and budget.** Four agents with a long recipe chain may need several million steps.
  The 500k pilot (two jobs in parallel, nothing else running) ran at ≈135–150 env steps/s per job and learned
  (own deliveries 0 → ≈25 per episode, greedy evaluation return ≈8.5 per agent at 0.5M); an early sanity run
  measured only ≈37 steps/s while the test suites shared the CPU. 5M steps take roughly 10 hours per job at
  the pilot rate; 6 methods × 3 seeds = 18 jobs. Run them yourself (scripts below), e.g. 2–4 at a time.
- **Shaping.** The shaping terms are individual and recipe-specific. An ablation without shaping is one
  override away (`--set kitchen.pot_reward=0 kitchen.soup_pickup_reward=0`).
- **Distance mode.** `action_max` was chosen for RWARE (D108). A pooled-distance ablation:
  `--set gate.distance_mode=pooled`.
- **Encoder.** MLP on a 1,579-dim egocentric vector; a CNN encoder may learn faster but would add a second
  change; kept identical across methods.

## 8. How to run

New queue-launched jobs are tracked in W&B project `pgct-overcooked` by default. The plan name is the W&B
group and each `(method, seed)` is a separate run; `wandb.json` records the remote ID/URL in the job folder.
Local `metrics.jsonl`, `gate.jsonl`, and `eval.jsonl` remain authoritative and are written whether W&B is
online, offline, disabled, or encounters a logging failure.

See `Overcooked/README.md`.
