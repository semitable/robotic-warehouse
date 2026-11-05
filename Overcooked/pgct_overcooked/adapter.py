"""Overcooked kitchen -> pgct_core.EnvAdapter / ProbeProjection.

Information separation (pgct_core.interfaces):
1. Local actor observation: egocentric (2R+1)x(2R+1) window of the channels below plus ego features (own
   facing, own held item). No absolute coordinates, no team, no recipe, no other agent's identity.
2. Local critic input: identical to the actor observation (one shared space for all agents, so an approved
   teammate's critic can be evaluated on the receiver's history by Max-GAE).
3. Private reward: the agent's own scalar reward only.
4. Public probe view: the centre (2r+1)x(2r+1) crop of the observation plus the ego features. Every channel
   is public physical state (anyone next to the agent sees it), so the view is a strict subset of the
   observation and contains nothing team-specific. Views are binary (uint8 storage, as in RWARE).
5. Communication: agents within Chebyshev distance ``comm_range`` of each other.
6. Evaluation only: ``oracle_team_ids`` and per-team summaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from pgct_core.interfaces import NoMirror

CHANNEL_NAMES = (
    ["counter", "onion_dispenser", "tomato_dispenser", "dish_dispenser", "pot", "serve"]
    + [f"obj_{o}" for o in ("onion", "tomato", "dish", "soup_onion", "soup_tomato", "soup_mixed")]
    + [f"pot_onion_{k}" for k in (1, 2, 3)] + [f"pot_tomato_{k}" for k in (1, 2, 3)] + ["pot_cooking", "pot_ready"]
    + ["agent", "agent_facing_n", "agent_facing_e", "agent_facing_s", "agent_facing_w"]
    + [f"agent_held_{o}" for o in ("onion", "tomato", "dish", "soup_onion", "soup_tomato", "soup_mixed")]
    + ["out_of_bounds"])
CH = {name: k for k, name in enumerate(CHANNEL_NAMES)}
CH["agent_facing"] = CH["agent_facing_n"]
N_CHANNELS = len(CHANNEL_NAMES)
EGO_SIZE = 4 + 7  # facing one-hot + held one-hot (nothing, onion, tomato, dish, soup_onion, soup_tomato, soup_mixed)
HELD_INDEX = {None: 0, "onion": 1, "tomato": 2, "dish": 3, "soup_onion": 4, "soup_tomato": 5, "soup_mixed": 6}
AGENT_CHANNELS = slice(CH["agent"], CH["agent_held_soup_mixed"] + 1)


@dataclass
class AdapterConfig:
    obs_radius: int = 3      # actor/critic window radius
    probe_radius: int = 1    # public probe view radius (must not exceed obs_radius)
    comm_range: int = 3      # Chebyshev communication range

    def __post_init__(self):
        if not 0 <= self.probe_radius <= self.obs_radius:
            raise ValueError("probe_radius must be in [0, obs_radius]")
        if self.comm_range < 0:
            raise ValueError("comm_range must be nonnegative")


def obs_size(radius):
    return EGO_SIZE + N_CHANNELS * (2 * radius + 1) ** 2


class KitchenProbeProjection(NoMirror):
    """Public view = ego features + centre crop of the egocentric window (all channels are public)."""
    view_dtype = np.uint8
    n_actions = 6

    def __init__(self, obs_radius, probe_radius):
        self.obs_radius, self.probe_radius = int(obs_radius), int(probe_radius)
        self.view_size = obs_size(self.probe_radius)
        side, lo = 2 * self.obs_radius + 1, self.obs_radius - self.probe_radius
        grid_index = np.arange(N_CHANNELS * side * side).reshape(N_CHANNELS, side, side)
        crop = grid_index[:, lo: lo + 2 * self.probe_radius + 1, lo: lo + 2 * self.probe_radius + 1].reshape(-1)
        self.index = np.concatenate([np.arange(EGO_SIZE), EGO_SIZE + crop])

    def public_view(self, observation):
        observation = np.asarray(observation)
        if observation.shape[-1] != obs_size(self.obs_radius):
            raise ValueError("observation does not match the kitchen adapter's window")
        return observation[..., self.index].astype(np.uint8)


class KitchenAdapter:
    """pgct_core.interfaces.EnvAdapter for the multi-team kitchen."""

    def __init__(self, kitchen_config=None, adapter_config: Optional[AdapterConfig] = None, layout_rows=None):
        from .env.kitchen import Kitchen, KitchenConfig, N_ACTIONS
        self.env = Kitchen(kitchen_config or KitchenConfig(), layout_rows)
        self.config = adapter_config or AdapterConfig()
        self.n_agents = self.env.n_agents
        self.n_actions = N_ACTIONS
        self.obs_dim = obs_size(self.config.obs_radius)
        self.projection = KitchenProbeProjection(self.config.obs_radius, self.config.probe_radius)

    def describe(self):
        return dict(kitchen=asdict(self.env.config), adapter=asdict(self.config), n_agents=self.n_agents,
                    obs_dim=self.obs_dim, probe_view_size=self.projection.view_size, channels=list(CHANNEL_NAMES))

    def observations(self):
        R = self.config.obs_radius
        grid = self.env.grid_channels()
        H, W = grid.shape[1:]
        padded = np.zeros((N_CHANNELS, H + 2 * R, W + 2 * R), dtype=np.float32)
        padded[CH["out_of_bounds"]] = 1
        padded[:, R: R + H, R: R + W] = grid
        padded[CH["out_of_bounds"], R: R + H, R: R + W] = 0
        out = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i, agent in enumerate(self.env.agents):
            r, c = agent.pos
            window = padded[:, r: r + 2 * R + 1, c: c + 2 * R + 1].copy()
            window[AGENT_CHANNELS, R, R] = 0  # self is described by the ego features, not as an "other agent"
            out[i, agent.facing] = 1
            out[i, 4 + HELD_INDEX[agent.held]] = 1
            out[i, EGO_SIZE:] = window.reshape(-1)
        return out

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self.observations()

    def step(self, actions):
        rewards, terminated, truncated, info = self.env.step(actions)
        return self.observations(), rewards, terminated, truncated, info

    def contact_matrix(self):
        pos = np.asarray([a.pos for a in self.env.agents])
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
            for key in ("deliveries_own", "deliveries_other"):
                vals = [np.sum(np.asarray(e["events"].get(key, np.zeros(len(team_ids))))[members]) for e in episodes]
                out[f"eval_team{int(t)}_{key}"] = float(np.mean(vals))
            out[f"eval_team{int(t)}_return"] = float(np.mean([np.sum(np.asarray(e["returns"])[members])
                                                              for e in episodes]))
        return out
