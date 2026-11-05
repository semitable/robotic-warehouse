"""pgct_core: environment-independent PGCT (Probe-Gated Critic Transfer), PGCT-D2 variant.

Components
- reward_gate : individual reward predictors, Q_ij = E_i u E_j reward probes, pairwise response distance
                (pooled / action_max), EMA, maturity, symmetric approval, rollout-start snapshot.
- max_gae     : donor selection (gate / oracle / all / shuffled), donor critics on the receiver's history,
                Max-GAE actor advantages (D2 'shrink'), shared GAE.
- networks    : per-agent recurrent (GRU) actor-critic with an MLP encoder.
- ppo         : recurrent IPPO update (same math as the RWARE trainer's IPPO path).
- trainer     : generic training loop for environments behind pgct_core.interfaces.EnvAdapter.
- metrics     : identification and learning metrics (evaluation side; the only place labels are used).
"""
from .interfaces import EnvAdapter, NoMirror, ProbeProjection
from .max_gae import (DONOR_MODES, SCALES, MaxGAEConfig, blended_advantages, compute_gae, donor_values,
                      explained_variance, max_targets, select_donors, shuffled_gate_donors, uplift, warm_mask)
from .reward_gate import (DISTANCE_MODES, RewardGateConfig, RewardPredictor, RewardProbeGate, Ring,
                          action_max_distance, audit, features, response_distance)

__version__ = "0.1.0"
