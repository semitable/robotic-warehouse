from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from rware.multi_team_warehouse import (
    TeamRewardMode,
    _balanced_assignments,
    _coerce_team_reward_mode,
    _coerce_team_sizes,
)


class GridAction(Enum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class MultiTeamGrid(gym.Env):
    """Synthetic partially observable grid with hidden reward teams.

    Agents observe local geometry, nearby messages, other agents, and target
    types, but never observe their hidden team IDs. Correct target collections
    generate correlated rewards inside the hidden team, making the environment
    useful for latent team discovery and restricted intra-team sharing.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        n_agents: int = 6,
        n_teams: int = 2,
        msg_bits: int = 0,
        sensor_range: int = 2,
        communication_range: Optional[int] = None,
        max_steps: Optional[int] = 200,
        team_assignments: Optional[Sequence[int]] = None,
        shuffle_team_assignments_on_reset: bool = False,
        targets_per_team: Optional[Sequence[int]] = 2,
        obstacle_density: float = 0.05,
        target_layout: str = "zones",
        team_reward_mode: TeamRewardMode = TeamRewardMode.TEAM,
        target_reward: float = 1.0,
        wrong_team_penalty: float = 0.0,
        step_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        normalised_coordinates: bool = True,
        reveal_team_info: bool = False,
        render_mode: str = "human",
    ):
        if len(grid_size) != 2:
            raise ValueError("grid_size must be (height, width)")
        if n_teams < 1:
            raise ValueError("n_teams must be positive")
        if n_teams > n_agents:
            raise ValueError("n_teams cannot exceed n_agents")
        if sensor_range < 0:
            raise ValueError("sensor_range must be non-negative")
        if not 0.0 <= obstacle_density < 1.0:
            raise ValueError("obstacle_density must be in [0, 1)")

        self.grid_size = tuple(int(v) for v in grid_size)
        self.n_agents = int(n_agents)
        self.n_teams = int(n_teams)
        self.msg_bits = int(msg_bits)
        self.sensor_range = int(sensor_range)
        self.communication_range = (
            int(communication_range)
            if communication_range is not None
            else int(sensor_range)
        )
        self.max_steps = max_steps
        self.obstacle_density = float(obstacle_density)
        self.target_layout = target_layout
        self.team_reward_mode = _coerce_team_reward_mode(team_reward_mode)
        self.target_reward = float(target_reward)
        self.wrong_team_penalty = float(wrong_team_penalty)
        self.step_penalty = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.normalised_coordinates = normalised_coordinates
        self.reveal_team_info = reveal_team_info
        self.render_mode = render_mode

        if team_assignments is None:
            base_assignments = _balanced_assignments(n_agents, n_teams)
        else:
            base_assignments = np.asarray(team_assignments, dtype=np.int32)
            if base_assignments.shape != (n_agents,):
                raise ValueError("team_assignments must contain one team id per agent")
            if np.any(base_assignments < 0) or np.any(base_assignments >= n_teams):
                raise ValueError("team_assignments must be in [0, n_teams)")
            missing_teams = set(range(n_teams)) - set(base_assignments.tolist())
            if missing_teams:
                raise ValueError(f"Each team must contain at least one agent: missing {missing_teams}")

        self._base_agent_team_ids = base_assignments
        self.agent_team_ids = base_assignments.copy()
        self.shuffle_team_assignments_on_reset = shuffle_team_assignments_on_reset
        self.targets_per_team = _coerce_team_sizes(
            targets_per_team, n_teams, fallback_total=n_teams
        )

        sa_action_space = [len(GridAction), *self.msg_bits * (2,)]
        if len(sa_action_space) == 1:
            single_agent_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            single_agent_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(
            tuple(self.n_agents * [single_agent_action_space])
        )

        self._cell_features = 2 + self.msg_bits + self.n_teams
        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
        self._obs_length = 2 + self._obs_sensor_locations * self._cell_features
        obs_low = np.zeros(self._obs_length, dtype=np.float32)
        obs_high = np.ones(self._obs_length, dtype=np.float32)
        if not self.normalised_coordinates:
            obs_high[0] = max(1, self.grid_size[0] - 1)
            obs_high[1] = max(1, self.grid_size[1] - 1)
        single_agent_obs_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Tuple(
            tuple(self.n_agents * [single_agent_obs_space])
        )

        self._cur_steps = 0
        self.obstacles = np.zeros(self.grid_size, dtype=bool)
        self.agent_positions = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.messages = np.zeros((self.n_agents, self.msg_bits), dtype=np.float32)
        self.target_positions: Dict[Tuple[int, int], int] = {}
        self.team_collection_counts = np.zeros(self.n_teams, dtype=np.int64)
        self._last_collection_events = []
        self._last_blocked = np.zeros(self.n_agents, dtype=bool)

    def get_oracle_team_assignments(self) -> np.ndarray:
        return self.agent_team_ids.copy()

    def get_team_members(self) -> List[List[int]]:
        return [
            np.flatnonzero(self.agent_team_ids == team_id).astype(np.int32).tolist()
            for team_id in range(self.n_teams)
        ]

    def get_neighbor_adjacency(self) -> np.ndarray:
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=np.int8)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.abs(self.agent_positions[i] - self.agent_positions[j]).sum()
                if dist <= self.communication_range:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
        return adjacency

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed, options=options)

        if self.shuffle_team_assignments_on_reset:
            self.agent_team_ids = self.np_random.permutation(self._base_agent_team_ids)
        else:
            self.agent_team_ids = self._base_agent_team_ids.copy()

        self._cur_steps = 0
        self.messages[:] = 0.0
        self.team_collection_counts[:] = 0
        self._last_collection_events = []
        self._last_blocked[:] = False

        self._make_obstacles()
        self.target_positions = {}
        self._make_targets()
        self._spawn_agents()

        return tuple([self._make_obs(agent_id) for agent_id in range(self.n_agents)]), self._get_info()

    def _make_obstacles(self):
        self.obstacles = self.np_random.random(self.grid_size) < self.obstacle_density

    def _team_x_bounds(self, team_id: int) -> Tuple[int, int]:
        height, width = self.grid_size
        del height
        start = int(np.floor(width * team_id / self.n_teams))
        end = int(np.floor(width * (team_id + 1) / self.n_teams))
        return start, max(start + 1, end)

    def _free_cells(self, team_id: Optional[int] = None, exclude_agents: bool = True):
        occupied_targets = set(self.target_positions.keys())
        occupied_agents = {
            tuple(pos.tolist()) for pos in self.agent_positions
        } if exclude_agents else set()

        x_start, x_end = 0, self.grid_size[1]
        if team_id is not None and self.target_layout == "zones":
            x_start, x_end = self._team_x_bounds(team_id)
        elif self.target_layout not in ("zones", "uniform"):
            raise ValueError("target_layout must be 'zones' or 'uniform'")

        cells = []
        for y in range(self.grid_size[0]):
            for x in range(x_start, x_end):
                pos = (y, x)
                if self.obstacles[y, x]:
                    continue
                if pos in occupied_targets or pos in occupied_agents:
                    continue
                cells.append(pos)
        return cells

    def _sample_free_cell(self, team_id: Optional[int] = None, exclude_agents: bool = True):
        candidates = self._free_cells(team_id=team_id, exclude_agents=exclude_agents)
        if not candidates and team_id is not None:
            candidates = self._free_cells(team_id=None, exclude_agents=exclude_agents)
        if not candidates:
            raise RuntimeError("No free cells available for sampling")
        choice = self.np_random.integers(0, len(candidates))
        return candidates[int(choice)]

    def _make_targets(self):
        for team_id in range(self.n_teams):
            for _ in range(self.targets_per_team[team_id]):
                pos = self._sample_free_cell(team_id=team_id, exclude_agents=False)
                self.target_positions[pos] = team_id

    def _spawn_agents(self):
        positions = []
        for _ in range(self.n_agents):
            pos = self._sample_free_cell(team_id=None, exclude_agents=False)
            while pos in positions:
                pos = self._sample_free_cell(team_id=None, exclude_agents=False)
            positions.append(pos)
        self.agent_positions = np.asarray(positions, dtype=np.int32)

    def _agent_at(self, y: int, x: int) -> Optional[int]:
        matches = np.flatnonzero(
            (self.agent_positions[:, 0] == y) & (self.agent_positions[:, 1] == x)
        )
        if len(matches) == 0:
            return None
        return int(matches[0])

    def _make_obs(self, agent_id: int) -> np.ndarray:
        obs = np.zeros(self._obs_length, dtype=np.float32)
        idx = 0
        y, x = self.agent_positions[agent_id]
        if self.normalised_coordinates:
            obs[idx] = y / max(1, self.grid_size[0] - 1)
            obs[idx + 1] = x / max(1, self.grid_size[1] - 1)
        else:
            obs[idx] = y
            obs[idx + 1] = x
        idx += 2

        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                cell_y = int(y + dy)
                cell_x = int(x + dx)
                if (
                    cell_y < 0
                    or cell_x < 0
                    or cell_y >= self.grid_size[0]
                    or cell_x >= self.grid_size[1]
                ):
                    obs[idx] = 1.0
                    idx += self._cell_features
                    continue

                if self.obstacles[cell_y, cell_x]:
                    obs[idx] = 1.0
                idx += 1

                other_id = self._agent_at(cell_y, cell_x)
                if other_id is not None:
                    obs[idx] = 1.0
                    if self.msg_bits:
                        obs[idx + 1 : idx + 1 + self.msg_bits] = self.messages[other_id]
                idx += 1 + self.msg_bits

                target_team = self.target_positions.get((cell_y, cell_x))
                if target_team is not None:
                    obs[idx + target_team] = 1.0
                idx += self.n_teams

        return obs

    def _get_info(self):
        info = {
            "collections": len(self._last_collection_events),
            "neighbor_adjacency": self.get_neighbor_adjacency(),
        }
        if self.reveal_team_info:
            info.update(
                {
                    "agent_team_ids": self.agent_team_ids.copy(),
                    "team_members": self.get_team_members(),
                    "team_collection_counts": self.team_collection_counts.copy(),
                    "collection_events": list(self._last_collection_events),
                    "target_team_counts": np.bincount(
                        list(self.target_positions.values()),
                        minlength=self.n_teams,
                    ),
                }
            )
        return info

    def _parse_actions(self, actions):
        requested_actions = []
        for agent_id, action in enumerate(actions):
            if self.msg_bits > 0:
                requested_actions.append(GridAction(action[0]))
                self.messages[agent_id] = np.asarray(action[1:], dtype=np.float32)
            else:
                requested_actions.append(GridAction(action))
        return requested_actions

    def _requested_positions(self, requested_actions):
        desired = self.agent_positions.copy()
        for agent_id, action in enumerate(requested_actions):
            y, x = self.agent_positions[agent_id]
            if action == GridAction.UP:
                desired[agent_id] = (y - 1, x)
            elif action == GridAction.DOWN:
                desired[agent_id] = (y + 1, x)
            elif action == GridAction.LEFT:
                desired[agent_id] = (y, x - 1)
            elif action == GridAction.RIGHT:
                desired[agent_id] = (y, x + 1)
        return desired

    def _resolve_collisions(self, requested_actions, desired):
        attempted_move = np.asarray(
            [action != GridAction.NOOP for action in requested_actions], dtype=bool
        )
        blocked = np.zeros(self.n_agents, dtype=bool)

        for agent_id, (y, x) in enumerate(desired):
            if (
                y < 0
                or x < 0
                or y >= self.grid_size[0]
                or x >= self.grid_size[1]
                or self.obstacles[y, x]
            ):
                desired[agent_id] = self.agent_positions[agent_id]
                blocked[agent_id] = attempted_move[agent_id]

        for _ in range(self.n_agents):
            changed = False
            desired_tuples = [tuple(pos.tolist()) for pos in desired]
            current_tuples = [tuple(pos.tolist()) for pos in self.agent_positions]

            for pos in set(desired_tuples):
                ids = [idx for idx, target in enumerate(desired_tuples) if target == pos]
                if len(ids) <= 1:
                    continue
                for agent_id in ids:
                    if tuple(desired[agent_id]) != current_tuples[agent_id]:
                        desired[agent_id] = self.agent_positions[agent_id]
                        blocked[agent_id] = True
                        changed = True

            stationary_positions = {
                current_tuples[idx]
                for idx in range(self.n_agents)
                if desired_tuples[idx] == current_tuples[idx]
            }
            for agent_id in range(self.n_agents):
                if tuple(desired[agent_id]) == current_tuples[agent_id]:
                    continue
                if tuple(desired[agent_id]) in stationary_positions:
                    desired[agent_id] = self.agent_positions[agent_id]
                    blocked[agent_id] = True
                    changed = True

            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    if (
                        tuple(desired[i]) == current_tuples[j]
                        and tuple(desired[j]) == current_tuples[i]
                        and current_tuples[i] != current_tuples[j]
                    ):
                        desired[i] = self.agent_positions[i]
                        desired[j] = self.agent_positions[j]
                        blocked[i] = attempted_move[i]
                        blocked[j] = attempted_move[j]
                        changed = True

            if not changed:
                break

        return desired, blocked & attempted_move

    def _reward_collection(self, rewards: np.ndarray, team_id: int, collector_id: int) -> List[int]:
        rewarded_agents: List[int] = []
        if self.team_reward_mode == TeamRewardMode.ALL:
            rewards += self.target_reward
            rewarded_agents = list(range(self.n_agents))
        elif self.team_reward_mode == TeamRewardMode.INDIVIDUAL:
            rewards[collector_id] += self.target_reward
            rewarded_agents = [collector_id]
        elif self.team_reward_mode == TeamRewardMode.TEAM:
            team_members = np.flatnonzero(self.agent_team_ids == team_id)
            rewards[team_members] += self.target_reward
            rewarded_agents = team_members.astype(np.int32).tolist()
        else:
            raise ValueError(f"Unsupported team reward mode: {self.team_reward_mode}")
        return rewarded_agents

    def _respawn_target(self, team_id: int):
        pos = self._sample_free_cell(team_id=team_id, exclude_agents=True)
        self.target_positions[pos] = team_id

    def step(self, actions):
        assert len(actions) == self.n_agents
        self._last_collection_events = []

        requested_actions = self._parse_actions(actions)
        desired = self._requested_positions(requested_actions)
        desired, blocked = self._resolve_collisions(requested_actions, desired)
        self.agent_positions = desired
        self._last_blocked = blocked

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        if self.step_penalty:
            rewards += self.step_penalty
        if self.collision_penalty:
            rewards[blocked] -= self.collision_penalty

        for agent_id, pos in enumerate(self.agent_positions):
            pos_tuple = tuple(pos.tolist())
            target_team = self.target_positions.get(pos_tuple)
            if target_team is None:
                continue

            agent_team = int(self.agent_team_ids[agent_id])
            if agent_team != target_team:
                if self.wrong_team_penalty:
                    rewards[agent_id] -= self.wrong_team_penalty
                continue

            rewarded_agents = self._reward_collection(rewards, target_team, agent_id)
            self.team_collection_counts[target_team] += 1
            del self.target_positions[pos_tuple]
            self._respawn_target(target_team)
            self._last_collection_events.append(
                {
                    "agent_id": agent_id,
                    "agent_team_id": agent_team,
                    "target_team_id": target_team,
                    "position": pos_tuple,
                    "rewarded_agents": rewarded_agents,
                }
            )

        self._cur_steps += 1
        done = bool(self.max_steps and self._cur_steps >= self.max_steps)
        truncated = False

        obs = tuple([self._make_obs(agent_id) for agent_id in range(self.n_agents)])
        return obs, list(rewards), done, truncated, self._get_info()

    def render(self):
        height, width = self.grid_size
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        image[self.obstacles] = np.asarray([35, 35, 35], dtype=np.uint8)

        colors = np.asarray(
            [
                [220, 70, 70],
                [60, 120, 230],
                [70, 170, 100],
                [210, 150, 45],
                [150, 90, 200],
                [40, 180, 180],
            ],
            dtype=np.uint8,
        )

        for (y, x), team_id in self.target_positions.items():
            image[y, x] = colors[team_id % len(colors)]

        for agent_id, (y, x) in enumerate(self.agent_positions):
            team_id = int(self.agent_team_ids[agent_id])
            if self.reveal_team_info:
                image[y, x] = np.maximum(colors[team_id % len(colors)] // 2, 30)
            else:
                image[y, x] = np.asarray([20, 20, 20], dtype=np.uint8)

        if self.render_mode == "rgb_array":
            return image

        print("\n".join("".join(self._render_char(y, x) for x in range(width)) for y in range(height)))
        return None

    def _render_char(self, y: int, x: int) -> str:
        agent_id = self._agent_at(y, x)
        if agent_id is not None:
            return "A"
        if self.obstacles[y, x]:
            return "#"
        target_team = self.target_positions.get((y, x))
        if target_team is not None:
            return str(target_team)
        return "."

    def close(self):
        return None
