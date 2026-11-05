# E1 from-scratch comparison — draft text for the paper (IJCAI)

Source run: `runs/scratch_20261015_093349_stage2` (stopped by the user at ~6.5M of a planned 15M steps).
All numbers: `numbers.json`; figures: `figures/*.pdf`; tables: `tables/*.tex`; regenerate with
`python output/paper_e1_20261018/make_paper_materials.py`.

> **Status of the evidence.** One training seed. Treat everything below as a *pilot* observation that motivates
> the multi-seed experiment, not as a result. Sentences marked **[needs seeds]** must not appear in the paper
> until seeds 2–3 (or more) confirm them, with mean ± CI. The wording below is already hedged accordingly.

---

## Experimental setup (paragraph)

We use a two-team variant of the multi-robot warehouse (RWARE) on an 11×10 map with four agents (agents 1, 3
in team red; agents 2, 4 in team blue). Each team has two outstanding shelf requests at any time and its own
delivery goal; the two goals are adjacent at the bottom centre of the map, and half of each team's shelves are
placed in the other team's half of the map. Observations are egocentric 11×11 semantic grids that encode
*team-relative* information (whether a shelf is requested by the observer's team, whether a goal is the
observer's), and rewards are individual: an agent is rewarded only for its own deliveries and returns,
plus small shaping terms. All agents use independent recurrent PPO actors and critics trained from random
initialisation (no curriculum); all methods start from bit-identical initial weights. The methods differ only in
which peers' critics a learner distils from: **IPPO** (no sharing), **Oracle-team** (ground-truth teammates),
**All-peer** (every agent), and **PGCT** (peers whose policies, measured by the Jensen–Shannon divergence on a fixed
bank of 48 canonical probe observations, are below a threshold of 0.12 after EMA smoothing). Distillation is an
auxiliary loss λ_peer·E_q[(V_i(q) − sg(V_j(q)))²] on the probe inputs, applied only between agents within
communication range. Policies are evaluated every 51,200 environment steps on 20 episodes of 500 steps with a
deterministic masked policy and fixed reset seeds. Hyper-parameters are in Table A1.

## Results (paragraphs)

**Learning speed and final throughput (Fig. 1, Table 1).** In this seed, sharing critics with the ground-truth
team accelerated the onset of deliveries (first ≥1 delivery per episode at 1.64M steps vs. 2.05M for IPPO) and
yielded the highest throughput over the common horizon of 6.4M steps (mean 13.2 vs. 10.4 deliveries per
episode over training; 32.2 ± 1.7 vs. 23.2 ± 1.1 over the last five evaluations). PGCT lay between the two
(onset 1.79M, mean 11.7, final 27.3 ± 1.4). Sharing with every agent delayed learning substantially (onset
3.99M, mean 3.7) but was still improving fastest at the end of the run (19.4 ± 1.6). **[needs seeds]**

**Sharing determines how many agents learn (Fig. 2).** The throughput differences are largely explained by the
number of agents that learned to deliver at all: two under IPPO (one per team), three under PGCT and Oracle-team,
and all four under All-peer. Every agent that learned contributed roughly 10–12 deliveries per episode, whereas
agents that did not learn never picked up a requested shelf in evaluation although they kept moving. We
hypothesise that with individual rewards and only two requests per team, the first teammate to learn serves most
requests, leaving the other with too little reward to learn from; distilling a competent peer's value estimates
gives the lagging agent a learning signal without its own rewards. Consistent with this, under Oracle-team the
lagging teammate of agent 1 (agent 3) started delivering at 3.58M while it never did under IPPO. Sharing is not
sufficient, however: agent 4 never learned under Oracle-team despite distilling from agent 2 throughout.
**[needs seeds; the identity of the lagging agent is seed-dependent]**

**Policy-similarity gating does not recover teams in E1 (Fig. 3, Table 2).** During the first 2M steps the
PGCT gate admitted transfer on 91% of updates, with more cross-team (3.09 arcs/update) than same-team (1.86)
transfer — largely because all pairs are similar early on. After 2M steps same-team transfer almost vanished
(0.10 and 0.03 arcs/update in 2–4M and 4–6.4M), and the remaining links were cross-team (agents 3–4, later
1–2). Oracle-team transferred within teams on 82–86% of updates and All-peer on essentially every update.

**Why: policy similarity on canonical probes is not a team signal here (Fig. 4, Table 3).** From about 2.5M steps
onwards the probe JS between teammates was *larger* than between non-teammates at almost every checkpoint for
all four methods (the only exception: IPPO at 4.1M) (e.g. at 6M: IPPO 0.27 vs. 0.18, Oracle-team 0.29 vs. 0.23), and all means exceeded the 0.12 gate threshold. With
team-relative observations and adjacent goals, both teams face the same decision problem, so policy
similarity reflects how far and in which style an agent has learned (e.g. forward-heavy long-haul carriers vs.
home-zone workers) rather than team membership. Sharing critics within the true team did not make teammates'
actors more similar either.

**Reproducibility (Fig. A1).** Re-running with the same seed and identical initial weights reproduced IPPO
exactly up to 2.46M steps, but the three sharing variants diverged from the earlier run at 1.38–1.59M steps,
presumably through non-deterministic GPU operations in the distillation path (not yet verified). The divergence
was consequential: All-peer's delivery onset moved from 1.89M to 3.99M. Any comparison between methods
therefore requires multiple seeds.

## Figure captions

- **Figure 1.** Evaluation deliveries per episode (20 episodes × 500 steps, deterministic policy) during training
  from random initialisation in E1, one training seed. Thin lines: raw evaluations every 51,200 steps; thick
  lines: centred 5-point moving average. All methods start from identical weights and differ only in the critic
  distillation partners.
- **Figure 2.** Deliveries per episode of each agent (5-point moving average). Red: team 1 (agents 1, 3); blue:
  team 2 (agents 2, 4); solid: agents 1–2, dashed: agents 3–4. Panel titles give the number of agents delivering
  at least once per episode over the last five evaluations.
- **Figure 3.** Realised critic transfer (from the logged donor allocation, after the 100-update warm-up). Left:
  directed transfer arcs per update, same-team (solid) vs. cross-team (dotted), averaged in 0.5M-step bins.
  Right: for PGCT, the share of updates in which each agent pair exchanged critic information.
- **Figure 4.** Difference between the mean probe JS of teammates and of non-teammates on the 48 canonical probes
  (zero hidden state), computed from saved checkpoints every ~0.5M steps. Values above zero mean that
  teammates' policies are *less* similar than non-teammates'.
- **Figure A1.** Same seed and initial weights, two runs (this run, solid; earlier interrupted run, dotted).
  IPPO is reproduced exactly; sharing variants diverge after 1.4–1.6M steps.

## Limitations to state explicitly

1. One training seed; no confidence intervals. Run stopped at ~6.5M; curves had not converged.
2. Oracle-team uses privileged team labels and is a reference, not an upper bound.
3. The E1 environment is team-symmetric (team-relative observations, adjacent goals); conclusions about team
   inference must be re-tested where teams' objectives differ (E1-F, goals at opposite corners; running).
4. The canonical probe bank has only 26 distinct inputs out of 48 and contains no "carrying, goal not visible"
   state; its sensitivity is analysed separately.
5. Distillation uses the same probe inputs for all sharing methods; all transfer is restricted to agents within
   communication range.
