"""Semantic local observation wrapper for RWARE experiments.

The wrapper keeps the environment dynamics unchanged and only replaces each
agent's returned observation with a flat vector containing:

    ego features || channel-major semantic local grid

The spatial part can be reshaped to ``[C, S, S]`` where
``S = 1 + 2 * sensor_range``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np

from rware.warehouse import Action, Direction, _LAYER_AGENTS, _LAYER_SHELFS


SEMANTIC_EGO_FEATURES: Tuple[str, ...] = (
    "self_x",
    "self_y",
    "self_carrying_shelf",
    "self_dir_up",
    "self_dir_down",
    "self_dir_left",
    "self_dir_right",
    "self_on_highway",
    "self_prev_action_noop",
    "self_prev_action_forward",
    "self_prev_action_left",
    "self_prev_action_right",
    "self_prev_action_toggle_load",
    "self_blocked_steps",
    "self_waiting_steps",
    "self_carrying_requested_shelf",
    "self_return_phase",
    "self_shelf_home_known",
    "self_shelf_home_dx",
    "self_shelf_home_dy",
)


SEMANTIC_SPATIAL_CHANNELS: Tuple[str, ...] = (
    "out_of_bounds",
    "highway",
    "storage_cell",
    "shelf",
    "requested_shelf",
    "agent",
    "agent_dir_up",
    "agent_dir_down",
    "agent_dir_left",
    "agent_dir_right",
    "agent_loaded",
    "goal",
    "relevant_goal",
)


@dataclass(frozen=True)
class SemanticObservationSpec:
    ego_dim: int
    spatial_channels: int
    spatial_size: int
    obs_dim: int
    ego_features: Tuple[str, ...] = SEMANTIC_EGO_FEATURES
    spatial_channel_names: Tuple[str, ...] = SEMANTIC_SPATIAL_CHANNELS


def semantic_observation_spec(env: gym.Env) -> SemanticObservationSpec:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    sensor_range = int(getattr(unwrapped, "sensor_range"))
    spatial_size = 1 + 2 * sensor_range
    ego_dim = len(SEMANTIC_EGO_FEATURES)
    spatial_channels = len(SEMANTIC_SPATIAL_CHANNELS)
    obs_dim = ego_dim + spatial_channels * spatial_size * spatial_size
    return SemanticObservationSpec(
        ego_dim=ego_dim,
        spatial_channels=spatial_channels,
        spatial_size=spatial_size,
        obs_dim=obs_dim,
    )


def get_semantic_observation_spec(env: gym.Env) -> SemanticObservationSpec:
    spec = getattr(env, "semantic_observation_spec", None)
    if isinstance(spec, SemanticObservationSpec):
        return spec
    return semantic_observation_spec(env)


def _requested_shelf_ids_for_observation(env: Any, agent: Any) -> Set[int]:
    if hasattr(env, "_requested_shelves_for_observation"):
        shelves = env._requested_shelves_for_observation(agent)
    else:
        shelves = getattr(env, "request_queue", [])
    return {int(getattr(shelf, "id", -1)) for shelf in shelves}


def _all_goal_cells(env: Any) -> Set[Tuple[int, int]]:
    return {
        (int(goal[0]), int(goal[1]))
        for goal in getattr(env, "goals", [])
        if len(tuple(goal)) >= 2
    }


def _relevant_goal_cells(env: Any, agent: Any) -> Set[Tuple[int, int]]:
    if hasattr(env, "agent_team_ids") and int(agent.id) > 0:
        agent_idx = int(agent.id) - 1
        if getattr(agent, "carrying_shelf", None) is not None:
            shelf_id = int(getattr(agent.carrying_shelf, "id", -1))
            team_id = int(
                getattr(env, "shelf_team_ids", {}).get(
                    shelf_id,
                    int(env.agent_team_ids[agent_idx]),
                )
            )
        else:
            team_id = int(env.agent_team_ids[agent_idx])

        if hasattr(env, "_goals_for_shelf_team"):
            return {
                (int(goal[0]), int(goal[1]))
                for goal in env._goals_for_shelf_team(team_id)
            }
        goals_by_team = getattr(env, "goals_by_team", None)
        if goals_by_team is not None and 0 <= team_id < len(goals_by_team):
            return {
                (int(goal[0]), int(goal[1]))
                for goal in goals_by_team[team_id]
            }
    return _all_goal_cells(env)


def _write_ego_features(env: Any, agent: Any, out: np.ndarray) -> None:
    height, width = tuple(map(int, getattr(env, "grid_size")))
    agent_idx = int(getattr(agent, "id", 0)) - 1
    if bool(getattr(env, "normalised_coordinates", False)):
        out[0] = float(agent.x) / float(max(width - 1, 1))
        out[1] = float(agent.y) / float(max(height - 1, 1))
    else:
        out[0] = float(agent.x)
        out[1] = float(agent.y)
    out[2] = 1.0 if getattr(agent, "carrying_shelf", None) is not None else 0.0
    out[3 + int(agent.dir.value)] = 1.0
    out[7] = 1.0 if bool(env._is_highway(int(agent.x), int(agent.y))) else 0.0

    prev_action = int(Action.NOOP.value)
    prev_actions = getattr(env, "_last_requested_actions", None)
    if prev_actions is not None and 0 <= agent_idx < len(prev_actions):
        prev_action = int(prev_actions[agent_idx])
    if 0 <= prev_action <= int(Action.TOGGLE_LOAD.value):
        out[8 + prev_action] = 1.0

    blocked_steps = getattr(env, "_consecutive_failed_forward_steps", None)
    if blocked_steps is not None and 0 <= agent_idx < len(blocked_steps):
        out[13] = min(float(blocked_steps[agent_idx]), 10.0) / 10.0

    waiting_steps = getattr(env, "_stationary_action_streak", None)
    if waiting_steps is not None and 0 <= agent_idx < len(waiting_steps):
        out[14] = min(float(waiting_steps[agent_idx]), 10.0) / 10.0

    # Proprioceptive task state: the carried shelf's destination is retained
    # through the return trip, even when its home is outside the local grid.
    # No other agent's identity, goal, or remote shelf location is exposed.
    shelf = getattr(agent, "carrying_shelf", None)
    if shelf is not None:
        out[15] = float(int(shelf.id) in _requested_shelf_ids_for_observation(env, agent))
        out[16] = float(int(shelf.id) in getattr(env, "_shelves_awaiting_return", set()))
        home = getattr(env, "_shelf_home_positions", {}).get(int(shelf.id))
        if home is not None:
            out[17] = 1.0
            out[18] = float(home[0] - agent.x) / max(width - 1, 1)
            out[19] = float(home[1] - agent.y) / max(height - 1, 1)


def build_semantic_observation(env: Any, agent: Any) -> np.ndarray:
    """Build one agent's egocentric semantic observation as a flat vector."""

    spec = semantic_observation_spec(env)
    obs = np.zeros((spec.obs_dim,), dtype=np.float32)
    _write_ego_features(env, agent, obs)

    height, width = tuple(map(int, getattr(env, "grid_size")))
    sensor_range = int(getattr(env, "sensor_range"))
    requested_ids = _requested_shelf_ids_for_observation(env, agent)
    all_goals = _all_goal_cells(env)
    relevant_goals = _relevant_goal_cells(env, agent)
    spatial = np.zeros(
        (spec.spatial_channels, spec.spatial_size, spec.spatial_size),
        dtype=np.float32,
    )

    for row, dy in enumerate(range(-sensor_range, sensor_range + 1)):
        wy = int(agent.y) + int(dy)
        for col, dx in enumerate(range(-sensor_range, sensor_range + 1)):
            wx = int(agent.x) + int(dx)
            if wx < 0 or wy < 0 or wx >= width or wy >= height:
                spatial[0, row, col] = 1.0
                continue

            is_highway = bool(env._is_highway(wx, wy))
            spatial[1, row, col] = 1.0 if is_highway else 0.0
            spatial[2, row, col] = 0.0 if is_highway else 1.0

            shelf_id = int(env.grid[_LAYER_SHELFS, wy, wx])
            if shelf_id > 0:
                spatial[3, row, col] = 1.0
                spatial[4, row, col] = 1.0 if shelf_id in requested_ids else 0.0

            agent_id = int(env.grid[_LAYER_AGENTS, wy, wx])
            if agent_id > 0:
                visible_agent = env.agents[agent_id - 1]
                spatial[5, row, col] = 1.0
                spatial[6 + int(visible_agent.dir.value), row, col] = 1.0
                spatial[10, row, col] = (
                    1.0
                    if getattr(visible_agent, "carrying_shelf", None) is not None
                    else 0.0
                )

            if (wx, wy) in all_goals:
                spatial[11, row, col] = 1.0
            if (wx, wy) in relevant_goals:
                spatial[12, row, col] = 1.0

    obs[spec.ego_dim :] = spatial.reshape(-1)
    return obs


class RwareSemanticObservationWrapper(gym.Wrapper):
    """Return semantic local observations while preserving RWARE dynamics."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.semantic_observation_spec = semantic_observation_spec(env)
        space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.semantic_observation_spec.obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Tuple(
            tuple(space for _ in range(int(self.unwrapped.n_agents)))
        )

    def _semantic_obs(self) -> Tuple[np.ndarray, ...]:
        return tuple(
            build_semantic_observation(self.unwrapped, agent)
            for agent in self.unwrapped.agents
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._semantic_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._semantic_obs(), reward, terminated, truncated, info

