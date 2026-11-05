# Max-GAE (D) — 2026-11-02 Claude Code

> 2026-11-05 repository split (D131): the Max-GAE math now lives in `PGCT/pgct_core/max_gae.py`; `RWARE/latent_team/max_gae.py` keeps the RWARE adapter (`ContactEnv`, `install`) with unchanged names and behaviour. Paths below are relative to `RWARE/`. See `docs/MIGRATION.md`.

Implementation of D91–D95: actor advantages use the max over met teammates' critics; critics keep learning from their own returns. Code: `latent_team/max_gae.py`. Tests: `tests/test_max_gae.py`. Defaults of existing modes are unchanged.

## Computation

- For receiver i with donor set G_i (fixed for one update), each donor critic j runs recurrently over i's own rollout observations. In this simulation all networks live in one process; this stands for sharing critic weights on contact.
- b = clip(max_j V_j(h_i) − V_i(h_i), 0, cap), and f = V_i + b.
- A_max = GAE(λ_f) of r + γ f(s') − f(s), with the same done and timeout handling as the trainer. The donor's timeout value is computed on the captured timeout observation.
- Actor advantage = A_self + η Δ, where Δ = A_max − A_self. GAE is linear in the value function, so with λ_f = λ this is GAE(V + η·min(b, cap)).
  - `raw`: Δ is used as is (D1 with η = 1).
  - `shrink` (default for D2 since D98): Δ is multiplied per receiver by s = min(1, std(A_self)/std(Δ)). This equals f = V + η·s·min(b, cap) with η·s ≤ η. Large Δ (laggards) is brought down to the receiver's own advantage scale; small Δ is left as is, so b → 0 returns to IPPO.
  - `relative` (record only): s without the min. It also inflates a near-zero Δ to std(A_self); with b ~ 1e-6 the shift was the same as with b ~ 1 (E66). Do not use it.
  - Reason for scaling at all: in the 1M branch measurement, raw Δ was 2.6–12× std(A_self) for laggards. After PPO's per-agent normalization, raw D2 (η=.5) and D1 became nearly the same update.
- Value loss = the receiver's own GAE returns (unchanged).
- Donor hidden states start cold at the rollout boundary. Until 8 observations have passed (`window`) or until the first reset, b = 0 and the actor advantage is A_self (Δ = 0). Zeroing only b left a spurious positive Δ of about +b on those steps (E66).
- With no donors, the trainer's `update_ippo` runs untouched. With donors, actor advantages are injected at its single `compute_gae` call; the PPO loop itself is the trainer's.

## Modes (`latent_team/train_from_scratch.py`)

All Max-GAE modes use the IPPO trainer path: the trainer's communication/consensus mechanism and critic peer loss are disabled. The Max-GAE adapter records physical contacts separately. Donors are agents met within the communication range (Chebyshev, `communication_range` = 3) at any step of the rollout.

| mode | donors | η |
|---|---|---|
| `oracle_max` | same team (ground-truth labels) | `--max-gae-eta` (default .5), `--max-gae-scale` (default shrink): D2 |
| `oracle_maxd1` | same team | 1, raw: D1 |
| `all_max` | anyone met | same as `oracle_max` |
| `allk_max` | random among those met, as many as `oracle_max` would use | same as `oracle_max` |
| `pgct_max` | met AND v2 reward-approved | same as `oracle_max` |
| `oracle_mature_max` | met AND same true team AND both reward predictors mature | same as `oracle_max` |

`pgct_max` and `oracle_mature_max` require `--reward-probe-ac --ac-gate v2`. Both learn fresh reward predictors on their own rollout experience (also when branching from a checkpoint). The branch loader copies actor/critic weights, not v2 buffers, optimizer or gate history.

At each update the Max-GAE wrapper calls the v2 controller exactly once on the completed rollout's contact matrix, before PPO updates. Refits and pair comparisons follow the unchanged v2 interval (default 5); pair comparisons are active only after transfer warm-up. This direct call replaces the proposed matched-transfer activation adapter: it avoids the IPPO path's always-false `active` flag and gives Oracle and PGCT the same rollout contact rule. The matched-transfer adapter only logs v2 fields and returns zero peer allocation. Evaluation never enters the reward buffers.

**Rollout-start snapshot (D103, from D102 option ii).** Comparing on rollout contacts means a pair may have parted long before the update. So for these two modes the plan sets `contact_snapshot=True` in the v2 config (other modes keep the old order):
- When a rollout starts, the controller freezes, per agent, the recent own event views (probes), the reward-predictor maturity and the predictor version. Predictors are refitted only after a rollout, so the frozen version is exactly the predictor used during the rollout. This is what an agent can send on any contact in that rollout.
- At the update the controller first compares contact pairs with the frozen probes, frozen maturity and the not-yet-refitted predictors, and only then refits. Experience gathered after the snapshot never enters a comparison. A refit between snapshot and comparison raises an error.
- `maturity()` and therefore `approved()` and `oracle_mature` use the frozen maturity flags (all False before the first snapshot).
- Older plans of these modes without the flag are refused at run time ("re-plan").
- Critic weights were already causal: donor critics are the weights fixed during the rollout.

`ACV2Controller.approved()` derives a symmetric matrix from retained pair rounds, EMA distance and maturity, without using the legacy contact-masked `gate`. Defaults: rounds >= 3, EMA < .5, both reward predictors mature. Losing contact does not erase approval evidence; Max-GAE still requires contact in the current rollout before selecting a donor. Team labels only enter the PGCT audit, never reward approval or gate donor selection.

Reward-predictor maturity is distinct from critic accuracy. The former follows the unchanged v2 event rules. Critic filtering is optional (see below); its default is disabled pending EV evidence, as agreed in D99. This implementation does not make reward-free agents identifiable or resolve the late cross-team errors in E65.

Other options:
- `--max-gae-lambda`: λ for A_max. Default is the trainer's λ.
- `--max-gae-cap`: upper bound on b. Default is none.
- `--max-gae-min-critic-ev FLOAT`: optional donor-only EV floor, measured before PPO on the donor's own current rollout. Nonfinite EV fails the enabled filter. Default `None` disables this filter; a threshold has not been selected. For a matched PGCT/Oracle comparison, use the same value on both arms.
- `--init-checkpoint PATH`: branch every mode from saved weights. Optimizers are fresh and the step counter restarts at 0. Job names get `_br<update>`. The plan is refused if a job's `team_goal_positions` differ from the checkpoint's. Candidate maps swap the goal sides on even seeds, so branch a seed-1 checkpoint with odd seeds (1, 3, 5).
- `--transfer-warmup N`: override the warm-up updates. Use 0 for a branch.

## Logs

- `max_gae.json` holds the configuration.
- `max_gae.jsonl` holds one row per update: contact matrix, donor matrix, per-agent selection rate, mean b, and normalized shift.
- W&B `update/max_gae_*` metrics:
  - `donor_arcs`, `same_team_arcs`, `cross_team_arcs`, `receivers`, `contact_arcs`, `arc_changes`
  - `select_frac`: steps with b > 0
  - `mean_b`, `max_b`, `capped_frac`
  - `adv_shift`: mean |A_actor − A_self| / std(A_self) for receivers with donors
  - `eta_eff`: std(A_actor − A_self) / std(Δ), the weight actually given to the max term (η·s for `shrink`)
  - `critic_ev_agent_k`: every agent's critic explained variance on its own rollout against its own value targets, logged every update, donors or not.
  - `critic_mature_agent_k`: whether the donor passes the optional EV filter (all true when disabled).
  - `reward_mature_agent_k`, `reward_mature_agents`: v2 reward maturity, including receivers that never get a donor.
  - `approved_arcs`, `approved_same_team_arcs`, `approved_cross_team_arcs`, `same_team_approval_coverage`: contact-independent approval and ground-truth audit; the denominator is the actual number of directed true-team pairs (12 for 6 agents in two teams of 3).

For the two new modes, `max_gae.jsonl` also holds the `approved`, `reward_mature`, `critic_mature` arrays, the filter-enabled flag, last probe-update index, v2 pair details and `env_steps`. The `donors` matrix and `max_gae_*_arcs` count actual Max-GAE donor use. `transfer.jsonl.actual_allocation` remains zero because it describes critic distillation; its zero arcs do not mean Max-GAE was inactive. Approval, current rollout contact and actual donor use are separate quantities.

## Verified (PGCT-D2, 2026-11-02, E68)

- D103 (2026-11-02, Claude Code, E71): 5 more tests in `tests/test_pgct_max.py` (snapshot comparison equals the rollout-start computation, refit guard, plan flag only for the two modes, refusal of old plans, real-trainer natural approval path with permissive wiring-only settings); 77 passed. The real-trainer tests need a short `--basetemp`: the `oracle_mature_max` job paths exceed Windows' 260 characters under a deep temporary folder.
- `tests/test_pgct_max.py`: 15 new tests; combined with the existing 57, 72 passed. Artifacts: `output/pgct_d2_validation_20261102/pytest_final.xml` and `pytest_final/test_real_trainer_closed_gate_0/`.
- Actual trainer smoke: corner goals, individual reward, masked coordinates, seed 1, common A+C IPPO s1 checkpoint at update 1000 (1,024,000 steps); 32-step rollout x 2 updates = 64 steps per job, CPU. Nine jobs, no W&B or long training.
- With v2 actually refitting but no mature approvals, both new modes finish bit-identically to IPPO.
- Diagnostic oracle approval/maturity injection (test-only) makes both new modes bit-identical to `oracle_max`, with positive actor shifts and zero cross-team donors.
- A test-only EV floor of 1e9 blocks every donor even with oracle approval and returns both modes bit-identically to IPPO. A one-update warm-up blocks update 1 and releases donors on update 2.
- Copied-critic check: raw/shrink actor advantages agree with self advantages to absolute tolerance 1e-6.
- Synthetic tests cover contact-independent symmetric approval, maturity regression, strict distance/round bounds, contact intersection, label independence, invalid inputs, and plan-time v2 enforcement. These establish wiring, not learning performance.

For a later user-launched second-stage branch, use the same branch command as T164 with `--modes oracle_mature_max pgct_max` and a **new output folder**. Retain `--reward-probe-ac --ac-gate v2 --seeds 1 3 5 --transfer-warmup 0`. No second-stage long run has been started by this implementation.

## Verified (D98 changes, 2026-11-02, E67)

- Unit tests: 19 in `tests/test_max_gae.py`; with the A+C and latent_team tests, 57 passed.
- Real-trainer smoke runs (64 steps):
  - With donors held off, `oracle_max` final weights are bit-identical to `ippo`.
  - The 1M branch runs with `shrink`. Leaders whose b was zero after the mask had shift 0.
  - The recommended branch plan (`--ac-gate v2`, 4 modes × seeds 1, 3, 5) completed with identical initial weights.
  - Seed 2 from the seed-1 checkpoint is refused at planning.

## Verified (first implementation, E64)

- Unit tests: 11 passed. Existing A+C and latent_team tests: 38 passed.
- Real-trainer smoke runs with donors held off: `oracle_max` final weights are bit-identical to `ippo`, both from scratch and from the 1M branch.
- Branch smoke run from `runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt`: all modes run and share the same initial-weight hash (`fc4a533b…`), and oracle donors are same-team only.
- Measurement run: 4 updates of 1,024 steps from that checkpoint, `oracle_max` and `all_max` with raw η=.5, scratchpad only.
  - Every pair met within a rollout (30/30 arcs), so rollout-level contact removes most of the communication limit.
  - Laggards (0-index 0, 1, 4) had b > 0 on 99% of steps, with mean b .30–.53 and raw shift 2.6–12.
  - Leaders 2, 3, 5 had shift 0–.4.
  - Throughput was about 25.6 sps, against 26.1 for the original IPPO near 1M.

## Limits

MAMBA-like, without MAMBA's guarantees (D94, D95). Donor critics are extrapolated to the receiver's histories, not re-estimated by oracle roll-outs. The actor receives a deliberately biased auxiliary signal. Results from a branch run are comparisons from a common checkpoint; they are not from-scratch IPPO comparisons.
