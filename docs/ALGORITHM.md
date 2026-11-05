# PGCT-D2 algorithm (shared core)

PGCT (Probe-Gated Critic Transfer) asks whether agents that learn **independently from individual
rewards** can find their latent teammates from local experience, and whether giving them selective access to
approved teammates' critics helps learning. PGCT-D2 is the current reference variant (RWARE mode
`pgct_max`, decisions D91–D109 in `DECISIONS.md`). Its implementation lives in `PGCT/pgct_core`; this page
states the method and maps each symbol to code and configuration. Nothing here was changed by the 2026-11
repository split (see `docs/MIGRATION.md` for the equivalence evidence).

## Setting

- Agents `i = 1..N`, each with its own recurrent actor-critic (GRU), trained by IPPO on its **own scalar
  reward** `r_i`. No reward is summed or shared, no parameters are shared, no centralized critic.
- A hidden team assignment determines which agents share a reward function. It is never observed and never
  enters a PGCT decision. Only the Oracle baseline and evaluation/audit code read it.
- Agents can exchange messages only while in physical contact (Chebyshev distance `<= r_c`). A rollout
  "contact" means the pair met at **any** step of that rollout.

## 1. Reward predictor (per agent)

Each agent `i` fits `r̂_i(v, a, v')` to its own rewards, where `v, v'` are **public views** before/after a
transition and `a` is its action:

- MLP: `[v, onehot(a), v'] -> 128 -> 64 -> 1` (ReLU), Adam lr `1e-3`.
- Buffers: own transitions (FIFO 20,000) and own reward **events** `r_i >= 0.5` (FIFO 512).
- Refit every `P = 5` updates: 50 steps of batch 256 with 25 % events oversampled. Refits also run during the
  transfer warm-up so that votes are ready.
- Private RNGs (predictor init and sampling) leave the trainer's random streams untouched.

Code: `pgct_core.reward_gate.RewardPredictor`, `Ring`, `features`. Config: `RewardGateConfig`
(`buffer_size`, `event_buffer_size`, `event_reward`, `event_fraction`, `hidden`, `learning_rate`,
`train_steps`, `batch_size`, `interval`).

## 2. Public reward probes

`E_i` = public views `(v, a, v')` of agent `i`'s 32 most recent own reward events. For a pair in contact the
probe set is built from **actual experience only**:

    Q_ij = E_i ∪ E_j

Only views are exchanged, never rewards. The left-right mirror augmentation used in earlier RWARE runs is
**off by default** in `pgct_core` (`mirror=False`); RWARE's legacy `ACV2Config` still defaults to `True` for
old commands, and the PGCT-D2 main runs pass `mirror: false` (`RWARE/latent_team/acv2_nomirror.json`). No
synthetic trajectories are generated.

What counts as a public view is environment specific (`ProbeProjection.public_view`):

| | RWARE | Overcooked kitchen |
|---|---|---|
| view | own loaded/direction/highway bits + 11 public physical channels within radius 2 | own facing + held item + all 32 physical channels within radius 1 |
| excluded | absolute coordinates, own-team request channel, goal/return ego info | nothing to exclude: observations hold no team or recipe information |
| actions | 5 | 6 |

## 3. Maturity (abstention)

Agent `i` votes only if `events_seen_i >= 20`, it has been refit at least once, and
`mean_{q in E_i} r̂_i(q) >= 0.3`. An agent that never earned a reward event is never paired.
Code: `RewardPredictor.mature`, `RewardProbeGate.maturity`.

## 4. Pairwise reward-response distance

Responses `p = r̂_i(Q_ij)`, `q = r̂_j(Q_ij)`:

    pooled:      d_ij = Σ_q |p_q − q_q| / Σ_q (|p_q| + |q_q|)          (0 = same responses, 1 = disjoint)
    action_max:  d_ij = max_g d_ij^g over public actions g in which BOTH agents have >= n_min = 8 own events

`action_max` (D108) prevents event kinds that every team is paid for from diluting the kind on which teams
differ; with no qualifying action the pair is not compared that round. Since `pooled = Σ_g w_g d^g`, the
pooled distance never exceeds the action maximum. Code: `response_distance`, `action_max_distance`.
Config: `distance_mode` (`"pooled"` default in the dataclass; PGCT-D2 runs use `"action_max"`), `min_probes`.

## 5. EMA, rounds and approval

    D̄_ij ← d_ij                       (first round)
    D̄_ij ← (1 − β) D̄_ij + β d_ij     (β = 0.35)

A round counts only when both predictors were refit since the pair's previous counted round. Approval is
symmetric and does **not** require current contact:

    A_ij = [rounds_ij >= 3] ∧ [D̄_ij < τ = 0.5] ∧ mature_i ∧ mature_j

Losing contact does not erase the evidence; critic access still requires contact in the current rollout.
Code: `RewardProbeGate.compare`, `approved`. Config: `ema_beta`, `min_rounds`, `distance_threshold`.

## 6. Rollout-start snapshot (D103)

Because pairs are compared on rollout contacts, each agent's **sendable** state is frozen when a rollout
starts (recent own event views, maturity, predictor version). At the update the gate compares contact pairs
with that frozen state, and only then refits. A refit between snapshot and comparison raises an error.
Code: `RewardProbeGate.snapshot`, config `contact_snapshot=True` (forced for every gate method of the
generic trainer and for RWARE `pgct_max` / `oracle_mature_max`).

## 7. Approved critic access: Max-GAE (D2)

Donor set for receiver `i` at an update (after the transfer warm-up of 100 updates):

    G_i = { j ≠ i : contact_ij in this rollout ∧ A_ij }

Each donor critic `V_j` runs recurrently over the receiver's **own** critic inputs from a cold start (reset
after each episode end; timeout observations bootstrapped like the trainer). With `V_i` the receiver's critic:

    b_t   = clip( max_{j∈G_i} V_j(h_{i,t}) − V_i(h_{i,t}), 0, cap )      (cap: none)
    f_t   = V_i(h_{i,t}) + b_t
    A_max = GAE_{λ_f}(r_i, f)        A_self = GAE_λ(r_i, V_i)        Δ = A_max − A_self
    s_i   = min(1, std(A_self) / std(Δ))                              ('shrink')
    A_actor = A_self + η s_i Δ                                        (η = 0.5)

- Steps before the donor hidden state has seen `window = 8` observations (and before the first reset) keep
  `A_actor = A_self`.
- Critics keep learning from their **own** returns; only the actor advantage changes.
- No donors at all → the plain IPPO update runs (bit-identical to IPPO).
- Critic weights are those fixed during the rollout (shared on contact); an optional donor explained-variance
  floor (`min_critic_ev`) is disabled by default.

Code: `pgct_core.max_gae` (`select_donors`, `donor_values`, `warm_mask`, `uplift`, `max_targets`,
`blended_advantages`, `explained_variance`). Config: `MaxGAEConfig` (`eta`, `scale`, `gae_lambda`, `cap`,
`window`, `min_critic_ev`).

## 8. Learner (PPO/GRU)

Per agent, separate networks and optimizers: MLP encoder (Tanh) → GRUCell 128 → heads; Adam lr 2e-4
(eps 1e-5); rollout 1024 steps; sequences of 64; 8 sequence chunks per minibatch; 4 epochs; γ 0.99, λ 0.95;
clip 0.2; value clip 0.2; value coef 0.5; entropy 0.05; grad-norm 0.5; per-agent advantage normalisation;
actor update skipped for an agent with an all-zero reward rollout. (RWARE PGCT-D2 runs use the RWARE
trainer's semantic CNN encoder; the generic trainer uses the MLP encoder.)
Code: `pgct_core.networks`, `pgct_core.ppo` (bit-identical to the RWARE trainer's IPPO update, tested).

## 9. Methods compared

| method (generic / RWARE mode) | critic access donors | labels used |
|---|---|---|
| `ippo` / `ippo` | none | no |
| `oracle` / `oracle_max` | contact ∧ same true team | yes (baseline) |
| `all` / `all_max` | contact | no |
| `pgct_d2` / `pgct_max` | contact ∧ `A_ij` | no |
| `pgct_id_only` | none (gate runs and is logged) | no |
| `random_gate` | per receiver, `|G_i^PGCT|` donors drawn at random among contacts | no |
| — / `allk_max` | as many random contacts as the oracle set | count only |
| — / `oracle_mature_max` | contact ∧ same true team ∧ both mature | yes (diagnostic) |

## 10. Information rules (enforced in code and tests)

- The gate and donor selection of every non-oracle method never read team labels
  (`Overcooked/tests/test_training.py::test_non_oracle_methods_never_read_team_labels_without_audit`).
- Public views contain no team identity, coordinates or private task information
  (`Overcooked/tests/test_kitchen.py`, `RWARE/tests/test_reward_probe_ac*.py`).
- Audit metrics (`*_same_team_*`, `ac_v2_audit_*`, identification metrics) are computed with labels on the
  evaluation side and never fed back.
