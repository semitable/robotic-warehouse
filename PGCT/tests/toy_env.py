"""Smallest possible pgct_core environment (also the template in docs/ARCHITECTURE.md "adding an environment").

Ring of 4 cells, one agent per team member; each step an agent may press button A or B on its cell.
Team 0 is paid for A, team 1 for B (individual reward, own press only). The public view is the agent's
one-hot cell plus one-hot of the button it pressed; nothing in it names a team.
"""
import numpy as np

from pgct_core.interfaces import NoMirror


class ToyProjection(NoMirror):
    view_dtype = np.uint8
    n_actions = 3  # 0 wait, 1 press A, 2 press B

    def __init__(self, cells):
        self.view_size = cells + 2

    def public_view(self, observation):
        return np.asarray(observation[: self.view_size]).astype(np.uint8)


class ToyAdapter:
    def __init__(self, teams=(0, 0, 1, 1), cells=4, horizon=20):
        self.teams, self.cells, self.horizon = np.asarray(teams), cells, horizon
        self.n_agents, self.n_actions = len(teams), 3
        self.obs_dim = cells + 2
        self.projection = ToyProjection(cells)
        self.rng = np.random.default_rng()

    def _obs(self, pressed):
        obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        obs[np.arange(self.n_agents), self.pos] = 1
        for i, a in enumerate(pressed):
            if a:
                obs[i, self.cells + a - 1] = 1
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t, self.pos = 0, self.rng.integers(0, self.cells, self.n_agents)
        return self._obs([0] * self.n_agents)

    def step(self, actions):
        actions = np.asarray(actions)
        rewards = np.where(actions == 0, 0.0, np.where(actions - 1 == self.teams, 1.0, 0.0)).astype(np.float32)
        self.pos = (self.pos + self.rng.integers(-1, 2, self.n_agents)) % self.cells
        self.t += 1
        return self._obs(actions), rewards, False, self.t >= self.horizon, dict(events=dict(presses=(actions > 0) * 1.0))

    def contact_matrix(self):
        return np.ones((self.n_agents, self.n_agents), bool)

    def oracle_team_ids(self):
        return self.teams.copy()
