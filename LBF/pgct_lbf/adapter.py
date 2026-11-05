"""Multi-team LBF -> pgct_core.EnvAdapter / ProbeProjection.

Information separation (pgct_core.interfaces):
1. Local actor observation: the LBF grid observation, a (2s+1)x(2s+1) window around the agent (sight s) of
   1 + T + 1 layers: agent levels (the agent itself at the centre), food levels per food type, access. Raw LBF
   values (integer levels, 0/1 access). No coordinates, no team, no "own food type".
2. Local critic input: identical to the actor observation (one shared space for all agents).
3. Private reward: the agent's own LBF share for food of its own team only.
4. Public probe view: the centre (2r+1)x(2r+1) crop of all layers (r = probe_radius). Everything in it is
   public physical state (who stands where with which level, which food lies where). With r = 2 the crop
   covers a food adjacent to the agent and every cell next to that food, i.e. everything that decides whether
   a LOAD succeeds. LBF has no headings: all agents share the same north-up frame, so the same event looks
   the same to every agent. Integer views (uint8 storage, as in RWARE).
5. Communication: agents within Chebyshev distance ``comm_range``.
6. Evaluation only: ``oracle_team_ids`` and per-team summaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from pgct_core.interfaces import NoMirror

from .env.foraging import ACTION_NAMES, EVENT_KEYS, N_ACTIONS, ForagingConfig, MultiTeamForaging


@dataclass
class AdapterConfig:
    probe_radius: int = 2    # public probe view radius (must not exceed the LBF sight)
    comm_range: int = 3      # Chebyshev communication range

    def __post_init__(self):
        if self.probe_radius < 0 or self.comm_range < 0:
            raise ValueError("probe_radius and comm_range must be nonnegative")


def layer_names(n_teams):
    return ["agent_level"] + [f"food_type{t}_level" for t in range(n_teams)] + ["access"]


class ForagingProbeProjection(NoMirror):
    """Public view = centre (2r+1)x(2r+1) crop of every observation layer (all layers are public)."""
    view_dtype = np.uint8
    n_actions = N_ACTIONS

    def __init__(self, n_layers, sight, probe_radius):
        if not 0 <= probe_radius <= sight:
            raise ValueError("probe_radius must be in [0, sight]")
        self.n_layers, self.sight, self.probe_radius = int(n_layers), int(sight), int(probe_radius)
        side, lo, k = 2 * self.sight + 1, self.sight - self.probe_radius, 2 * self.probe_radius + 1
        index = np.arange(self.n_layers * side * side).reshape(self.n_layers, side, side)
        self.index = index[:, lo: lo + k, lo: lo + k].reshape(-1)
        self.view_size = int(self.index.size)
        self.obs_size = self.n_layers * side * side

    def public_view(self, observation):
        observation = np.asarray(observation)
        if observation.shape[-1] != self.obs_size:
            raise ValueError("observation does not match the LBF adapter's window")
        return np.rint(observation[..., self.index]).astype(np.uint8)


class ForagingAdapter:
    """pgct_core.interfaces.EnvAdapter for multi-team LBF."""

    def __init__(self, foraging_config: Optional[ForagingConfig] = None,
                 adapter_config: Optional[AdapterConfig] = None):
        self.env = MultiTeamForaging(foraging_config or ForagingConfig())
        self.config = adapter_config or AdapterConfig()
        self.n_agents = self.env.n_agents
        self.n_actions = N_ACTIONS
        self.sight = self.env.config.sight
        self.n_layers = self.env.n_teams + 2
        self.obs_dim = self.n_layers * (2 * self.sight + 1) ** 2
        self.projection = ForagingProbeProjection(self.n_layers, self.sight, self.config.probe_radius)

    def describe(self):
        return dict(foraging=asdict(self.env.config), adapter=asdict(self.config), n_agents=self.n_agents,
                    obs_dim=self.obs_dim, probe_view_size=self.projection.view_size,
                    layers=layer_names(self.env.n_teams), actions=list(ACTION_NAMES))

    def observations(self):
        s = self.sight
        layers = self.env.layers()
        out = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i, (r, c) in enumerate(self.env.positions):
            out[i] = layers[:, r: r + 2 * s + 1, c: c + 2 * s + 1].reshape(-1)
        return out

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self.observations()

    def step(self, actions):
        rewards, terminated, truncated, info = self.env.step(actions)
        return self.observations(), rewards, terminated, truncated, info

    def contact_matrix(self):
        pos = self.env.positions
        return np.abs(pos[:, None] - pos[None]).max(-1) <= self.config.comm_range

    def oracle_team_ids(self):
        """Ground truth: Oracle baseline and evaluation only."""
        return self.env.oracle_team_ids()

    def team_summary(self, episodes, team_ids):
        """Evaluation-only per-team aggregation of per-agent episode counters."""
        team_ids = np.asarray(team_ids)
        out = {}
        for t in np.unique(team_ids):
            members = team_ids == t
            for key in EVENT_KEYS:
                vals = [np.sum(np.asarray(e["events"].get(key, np.zeros(len(team_ids))))[members]) for e in episodes]
                out[f"eval_team{int(t)}_{key}"] = float(np.mean(vals))
            out[f"eval_team{int(t)}_return"] = float(np.mean([np.sum(np.asarray(e["returns"])[members])
                                                              for e in episodes]))
        return out
