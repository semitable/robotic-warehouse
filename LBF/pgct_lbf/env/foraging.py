"""Multi-team Level-Based Foraging (LBF) for PGCT: the lbforaging 2.0.0 rules plus hidden teams.

Rules kept from LBF (Christianos et al. 2020; github.com/semitable/lb-foraging, lbforaging 2.0.0):
- square grid; each episode draws player levels in [min_player_level, max_player_level] and random free cells;
- food is placed on rows/columns 1..n-2 with no other food in its 3x3 neighbourhood or within distance 2 along
  its row and column (so no cell touches two foods);
- six actions NONE, NORTH, SOUTH, WEST, EAST, LOAD; an invalid action (into the border or a food cell, LOAD
  with no food adjacent) becomes NONE; when two or more players target the same cell none of them moves;
- LOAD: the players adjacent (4-neighbourhood) to a food who LOAD in the same step pool their levels; the food
  is collected if the sum reaches its level, and each loader scores level_i * food_level / (sum of loader
  levels * normaliser). A failed attempt costs ``penalty`` (0 by default);
- grid observation: (2*sight+1)^2 windows of an agent-level layer, food-level layers and an access layer.

Extension for latent teams: every agent belongs to a hidden team and every food has a type (one per team).
The physics ignores teams (anyone can help lift any food). The reward is the LBF share, paid only to loaders
of the food's own team; the normaliser is the total level of that team's spawned food (LBF: all spawned food),
so each team can earn 1 per episode. Food of team t gets LBF's level range computed from team t's players
(LBF: max food level = sum of the three lowest player levels). Observations show food types (one layer per
type) but never which type an agent is paid for; ``oracle_team_ids()`` exists for the Oracle baseline and
evaluation only. With one team this is exactly LBF (tests/test_lbforaging_equivalence.py).

Deliberate differences from lbforaging: simultaneous loaders are processed in agent-index order instead of
Python set order (no difference under the LBF food spacing, where no cell touches two foods), and the time
limit is reported as truncation, not termination (correct bootstrapping in GAE).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("none", "north", "south", "west", "east", "load")
DELTAS = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}
EVENT_KEYS = ("collected_own", "collected_other", "load_failed")


@dataclass
class ForagingConfig:
    field_size: int = 12
    team_sizes: Tuple[int, ...] = (2, 2)
    food_per_team: int = 3
    min_player_level: int = 1
    max_player_level: int = 2
    min_food_level: int = 1
    max_food_level: Optional[int] = None   # None: LBF rule per team (sum of the team's three lowest levels)
    force_coop: bool = False               # LBF "-coop": every food at its maximum level
    sight: int = 2                         # LBF "-2s": partial observability
    horizon: int = 50                      # LBF max_episode_steps
    normalize_reward: bool = True
    penalty: float = 0.0
    team_assignment: str = "shuffle"       # "shuffle" (permutation drawn from team_seed) or "fixed" (in order)
    team_seed: int = 0

    def __post_init__(self):
        self.team_sizes = tuple(int(s) for s in self.team_sizes)
        if not self.team_sizes or min(self.team_sizes) < 1:
            raise ValueError("team_sizes must be positive")
        if self.field_size < 4 or self.food_per_team < 1 or self.sight < 1 or self.horizon < 1:
            raise ValueError("field_size >= 4, food_per_team >= 1, sight >= 1 and horizon >= 1 are required")
        if not 1 <= self.min_player_level <= self.max_player_level:
            raise ValueError("player levels must satisfy 1 <= min <= max")
        if self.min_food_level < 1 or (self.max_food_level is not None and self.max_food_level < self.min_food_level):
            raise ValueError("food levels must satisfy 1 <= min <= max")
        if self.team_assignment not in ("shuffle", "fixed"):
            raise ValueError("team_assignment must be 'shuffle' or 'fixed'")
        if self.penalty < 0:
            raise ValueError("penalty must be nonnegative")


class MultiTeamForaging:
    def __init__(self, config: Optional[ForagingConfig] = None):
        self.config = config or ForagingConfig()
        c = self.config
        self.n_agents = sum(c.team_sizes)
        self.n_teams = len(c.team_sizes)
        teams = np.repeat(np.arange(self.n_teams), c.team_sizes)
        if c.team_assignment == "shuffle":
            teams = np.random.default_rng(c.team_seed).permutation(teams)
        self._team_ids = teams.astype(np.int64)
        self.rows = self.cols = int(c.field_size)
        self.rng = np.random.default_rng()
        self.reset()

    # ----------------------------------------------------------------- evaluation-only information
    def oracle_team_ids(self) -> np.ndarray:
        """Ground truth. Oracle baseline and evaluation only; never part of observations or step info."""
        return self._team_ids.copy()

    # ----------------------------------------------------------------- episode
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.field = np.zeros((self.rows, self.cols), np.int64)     # food level (0: no food)
        self.food_type = np.full((self.rows, self.cols), -1, np.int64)
        self.positions = np.full((self.n_agents, 2), -1, np.int64)
        self.levels = np.zeros(self.n_agents, np.int64)
        self._spawn_players()
        self._spawn_food()
        self.t = 0
        return self

    def _occupied(self, r, c):
        return bool(np.any((self.positions[:, 0] == r) & (self.positions[:, 1] == c)))

    def _empty(self, r, c):
        return self.field[r, c] == 0 and not self._occupied(r, c)

    def _spawn_players(self):
        c = self.config
        for i in range(self.n_agents):
            for _ in range(1000):
                r, col = int(self.rng.integers(0, self.rows)), int(self.rng.integers(0, self.cols))
                if self._empty(r, col):
                    self.positions[i] = (r, col)
                    self.levels[i] = int(self.rng.integers(c.min_player_level, c.max_player_level + 1))
                    break

    def food_level_range(self, team):
        """LBF level range for food of `team` (computed from that team's players in this episode)."""
        c = self.config
        hi = (c.max_food_level if c.max_food_level is not None
              else int(np.sum(np.sort(self.levels[self._team_ids == team])[:3])))
        hi = max(hi, c.min_food_level)
        return (hi if c.force_coop else c.min_food_level), hi

    def _spawn_food(self):
        c = self.config
        foods = [(t, *self.food_level_range(t)) for t in range(self.n_teams) for _ in range(c.food_per_team)]
        foods = [foods[k] for k in self.rng.permutation(len(foods))]
        placed, attempts = 0, 0
        while placed < len(foods) and attempts < 1000:
            attempts += 1
            r, col = int(self.rng.integers(1, self.rows - 1)), int(self.rng.integers(1, self.cols - 1))
            if (self.field[max(r - 1, 0): r + 2, max(col - 1, 0): col + 2].sum() > 0
                    or self.field[max(r - 2, 0): r + 3, col].sum() + self.field[r, max(col - 2, 0): col + 3].sum() > 0
                    or not self._empty(r, col)):
                continue
            team, lo, hi = foods[placed]
            self.field[r, col] = lo if lo == hi else int(self.rng.integers(lo, hi + 1))
            self.food_type[r, col] = team
            placed += 1
        self.food_spawned = np.array([self.field[self.food_type == t].sum() for t in range(self.n_teams)], np.float64)

    # ----------------------------------------------------------------- rules
    def adjacent_food_location(self, r, c):
        """First adjacent food in LBF order (north, south, west, east), or None."""
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.rows and 0 <= cc < self.cols and self.field[rr, cc] > 0:
                return rr, cc
        return None

    def valid(self, i, action):
        r, c = self.positions[i]
        if action == NONE:
            return True
        if action == LOAD:
            return self.adjacent_food_location(r, c) is not None
        dr, dc = DELTAS[int(action)]
        rr, cc = r + dr, c + dc
        return 0 <= rr < self.rows and 0 <= cc < self.cols and self.field[rr, cc] == 0

    def step(self, actions):
        c = self.config
        actions = [int(a) if self.valid(i, int(a)) else NONE for i, a in enumerate(np.asarray(actions).reshape(-1))]
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions")
        self.t += 1
        rewards = np.zeros(self.n_agents, np.float64)
        events = {k: np.zeros(self.n_agents, np.float64) for k in EVENT_KEYS}
        targets = {}
        for i, a in enumerate(actions):
            r, col = self.positions[i]
            if a in DELTAS:
                r, col = r + DELTAS[a][0], col + DELTAS[a][1]
            targets.setdefault((int(r), int(col)), []).append(i)
        for cell, members in targets.items():
            if len(members) == 1:          # two or more players aiming at one cell: none of them moves
                self.positions[members[0]] = cell
        remaining = [i for i, a in enumerate(actions) if a == LOAD]
        while remaining:
            i = remaining.pop(0)
            fr, fc = self.adjacent_food_location(*self.positions[i])
            group = [i] + [j for j in remaining if abs(self.positions[j, 0] - fr) + abs(self.positions[j, 1] - fc) == 1]
            remaining = [j for j in remaining if j not in group]
            level, team = int(self.field[fr, fc]), int(self.food_type[fr, fc])
            total = int(self.levels[group].sum())
            if total < level:
                rewards[group] -= c.penalty
                events["load_failed"][group] += 1
                continue
            for j in group:
                if self._team_ids[j] == team:
                    share = float(self.levels[j] * level)
                    if c.normalize_reward:
                        share /= float(total * self.food_spawned[team])
                    rewards[j] += share
                    events["collected_own"][j] += 1
                else:
                    events["collected_other"][j] += 1
            self.field[fr, fc] = 0
            self.food_type[fr, fc] = -1
        terminated = bool(self.field.sum() == 0)
        truncated = bool(not terminated and self.t >= c.horizon)
        return rewards, terminated, truncated, dict(events=events)

    # ----------------------------------------------------------------- observation layers
    def layers(self):
        """[1 + n_teams + 1, rows + 2s, cols + 2s] padded LBF grid layers: agent levels, food levels per type,
        access (1 = inside the field and free). The adapter crops (2s+1)^2 windows from them."""
        s = self.config.sight
        out = np.zeros((self.n_teams + 2, self.rows + 2 * s, self.cols + 2 * s), np.float32)
        inner = (slice(s, s + self.rows), slice(s, s + self.cols))
        for i in range(self.n_agents):
            out[0, self.positions[i, 0] + s, self.positions[i, 1] + s] = self.levels[i]
        for t in range(self.n_teams):
            out[(1 + t, *inner)] = np.where(self.food_type == t, self.field, 0)
        access = np.zeros((self.rows + 2 * s, self.cols + 2 * s), np.float32)
        access[inner] = (self.field == 0)
        for i in range(self.n_agents):
            access[self.positions[i, 0] + s, self.positions[i, 1] + s] = 0
        out[-1] = access
        return out
