# Max-GAE (D) — 2026-11-02 Claude Code

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

All four modes use the IPPO trainer path: no environment communication and no critic peer loss. Donors are agents met within the communication range (Chebyshev, `communication_range` = 3) at any step of the rollout.

| mode | donors | η |
|---|---|---|
| `oracle_max` | same team (ground-truth labels) | `--max-gae-eta` (default .5), `--max-gae-scale` (default shrink): D2 |
| `oracle_maxd1` | same team | 1, raw: D1 |
| `all_max` | anyone met | same as `oracle_max` |
| `allk_max` | random among those met, as many as `oracle_max` would use | same as `oracle_max` |

Other options:
- `--max-gae-lambda`: λ for A_max. Default is the trainer's λ.
- `--max-gae-cap`: upper bound on b. Default is none.
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
  - `critic_ev_agent_k`: every agent's critic explained variance on its own rollout against its own value targets, logged every update, donors or not. Logged only; donor maturity is not gated yet.

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
