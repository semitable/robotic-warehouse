"""VMAS Team Discovery -> pgct_core.EnvAdapter / ProbeProjection.

Information separation (pgct_core.interfaces):
1. Local actor observation: VMAS discovery's observation (own position, own velocity, one LIDAR per target type,
   agent LIDAR). No team and no "own target type".
2. Local critic input: identical to the actor observation (one shared space for all agents).
3. Private reward: the agent's own covering reward (own-type targets only).
4. Public probe view: own velocity and the LIDAR distances (all public physical state); the absolute position is
   dropped. LIDAR rays point in fixed world directions for every agent (point masses have no heading), so the
   same event looks the same to every agent. Continuous views (float32 storage).
5. Communication: agents within Euclidean distance ``comm_range``.
6. Evaluation only: ``oracle_team_ids`` and per-team summaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from pgct_core.interfaces import NoMirror

EVENT_KEYS = ("covered_own", "covered_other")
N_ACTIONS = 9   # VMAS discrete actions: index a -> (x choice a // 3, y choice a % 3); choice 0 stay, 1 -u, 2 +u


@dataclass
class DiscoveryConfig:
    team_sizes: Tuple[int, ...] = (2, 2)
    targets_per_type: int = 3
    agents_per_target: int = 1         # VMAS discovery default: 2 (see configs: coop variant)
    covering_range: float = .25
    lidar_range: float = .35
    n_lidar_rays_targets: int = 12     # VMAS discovery: 15 (one LIDAR); here one LIDAR per target type
    n_lidar_rays_agents: int = 12
    x_semidim: float = 1.0
    y_semidim: float = 1.0
    min_dist_between_entities: float = .2
    covering_rew_coeff: float = 1.0
    agent_collision_penalty: float = 0.0
    time_penalty: float = 0.0
    horizon: int = 200                 # VMAS max_steps (truncation)
    team_assignment: str = "shuffle"   # "shuffle" (permutation drawn from team_seed) or "fixed"
    team_seed: int = 0

    def __post_init__(self):
        self.team_sizes = tuple(int(s) for s in self.team_sizes)
        if not self.team_sizes or min(self.team_sizes) < 1:
            raise ValueError("team_sizes must be positive")
        if self.targets_per_type < 1 or self.agents_per_target < 1 or self.horizon < 1:
            raise ValueError("targets_per_type, agents_per_target and horizon must be positive")
        if self.team_assignment not in ("shuffle", "fixed"):
            raise ValueError("team_assignment must be 'shuffle' or 'fixed'")

    def team_ids(self):
        teams = np.repeat(np.arange(len(self.team_sizes)), self.team_sizes)
        if self.team_assignment == "shuffle":
            teams = np.random.default_rng(self.team_seed).permutation(teams)
        return teams.astype(np.int64)


@dataclass
class AdapterConfig:
    comm_range: float = .6     # Euclidean communication range (arena side 2.0)

    def __post_init__(self):
        if self.comm_range < 0:
            raise ValueError("comm_range must be nonnegative")


class DiscoveryProbeProjection(NoMirror):
    """Public view = observation without the two absolute-position entries."""
    view_dtype = np.float32
    n_actions = N_ACTIONS

    def __init__(self, obs_dim):
        self.obs_dim = int(obs_dim)
        self.view_size = self.obs_dim - 2

    def public_view(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1] != self.obs_dim:
            raise ValueError("observation does not match the VMAS adapter")
        return observation[..., 2:].copy()


class DiscoveryAdapter:
    """pgct_core.interfaces.EnvAdapter for VMAS Team Discovery (one environment, CPU)."""

    def __init__(self, discovery_config: Optional[DiscoveryConfig] = None,
                 adapter_config: Optional[AdapterConfig] = None):
        import torch  # noqa: F401  (VMAS needs torch)
        import vmas
        from .scenario import TeamDiscovery
        self.discovery = discovery_config or DiscoveryConfig()
        self.config = adapter_config or AdapterConfig()
        self._team_ids = self.discovery.team_ids()
        kwargs = {k: v for k, v in asdict(self.discovery).items()
                  if k not in ("team_sizes", "horizon", "team_assignment", "team_seed")}
        # VMAS keeps ONE random state for all environments of a process (a class attribute that its local_seed
        # decorator swaps in around every call). Each adapter keeps its own copy, so an evaluation or video
        # environment never shifts the training environment's random stream.
        from vmas.simulator.environment.environment import Environment
        self._shared_rng = Environment.vmas_random_state
        self._own_rng = None
        self.env = self._isolated(vmas.make_env, TeamDiscovery(), num_envs=1, device="cpu", continuous_actions=False,
                                  max_steps=self.discovery.horizon, seed=0, terminated_truncated=True,
                                  team_ids=self._team_ids.tolist(), n_types=len(self.discovery.team_sizes), **kwargs)
        self.n_agents = len(self._team_ids)
        self.n_actions = N_ACTIONS
        self.obs_dim = 4 + len(self.discovery.team_sizes) * self.discovery.n_lidar_rays_targets \
            + self.discovery.n_lidar_rays_agents
        self.projection = DiscoveryProbeProjection(self.obs_dim)

    def describe(self):
        return dict(discovery=asdict(self.discovery), adapter=asdict(self.config), n_agents=self.n_agents,
                    obs_dim=self.obs_dim, probe_view_size=self.projection.view_size, n_actions=self.n_actions,
                    vmas_scenario="pgct_vmas.scenario.TeamDiscovery")

    def _isolated(self, fn, *args, **kwargs):
        """Run a VMAS call with this adapter's own copy of the VMAS random state."""
        outside = list(self._shared_rng)
        if self._own_rng is not None:
            self._shared_rng[:] = self._own_rng
        try:
            return fn(*args, **kwargs)
        finally:
            self._own_rng = list(self._shared_rng)
            self._shared_rng[:] = outside

    @staticmethod
    def _stack(observations):
        return np.stack([o[0].detach().cpu().numpy() for o in observations]).astype(np.float32)

    def reset(self, seed=None):
        out = self._stack(self._isolated(self.env.reset, seed=seed))
        if out.shape != (self.n_agents, self.obs_dim):
            raise RuntimeError(f"unexpected VMAS observation shape {out.shape}")
        return out

    def step(self, actions):
        import torch
        actions = [torch.tensor([int(a)]) for a in np.asarray(actions).reshape(-1)]
        obs, rewards, terminated, truncated, infos = self._isolated(self.env.step, actions)
        rewards = np.array([float(r[0]) for r in rewards], dtype=np.float64)
        events = {k: np.array([float(info[k][0]) for info in infos]) for k in EVENT_KEYS}
        return self._stack(obs), rewards, bool(terminated[0]), bool(truncated[0]), dict(events=events)

    def positions(self):
        return np.stack([a.state.pos[0].detach().cpu().numpy() for a in self.env.agents])

    def target_state(self):
        """(positions [m, 2], types [m]) of the targets (rendering and tests)."""
        scenario = self.env.scenario
        pos = np.stack([t.state.pos[0].detach().cpu().numpy() for t in scenario._targets])
        return pos, scenario.target_types.cpu().numpy()

    def contact_matrix(self):
        pos = self.positions()
        return np.linalg.norm(pos[:, None] - pos[None], axis=-1) <= self.config.comm_range

    def oracle_team_ids(self):
        """Ground truth: Oracle baseline and evaluation only."""
        return self._team_ids.copy()

    def team_summary(self, episodes, team_ids):
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
