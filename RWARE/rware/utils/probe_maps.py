"""Canonical RWARE probe maps used by PGCT policy probing.

The training script imports the builder in this module to keep the synthetic
probe bank as a visible, inspectable artifact. Probe states are realized in a
warehouse with the same grid geometry as the training environment when that
geometry is available. Run this file as a module to save PNG renderings of the
current probe scenarios:

    python -m rware.utils.probe_maps --batch-size 48 --output-dir output/probe_maps
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from rware.multi_team_warehouse import MultiTeamWarehouse, TeamRewardMode
from rware.utils.semantic_observation import (
    RwareSemanticObservationWrapper,
    SemanticObservationSpec,
    build_semantic_observation,
    get_semantic_observation_spec,
)
from rware.warehouse import Direction, RewardType


DEFAULT_RWARE_PROBE_LAYOUT = """
...........
.xx...xx...
.xx...xx...
...........
.xx...xx...
.xx...xx...
...........
...g...g...
...........
"""


RWARE_TEAM_SCENARIO_BLOCKS: Tuple[Tuple[str, ...], ...] = (
    (
        "team_home_approach",
        "team_home_pickup",
        "team_goal_approach",
        "team_goal_at_goal",
    ),
    (
        "team_return_approach",
        "team_return_at_home",
        "wrong_team_goal",
        "team_conflict",
    ),
    (
        "dual_requested_home_pair",
        "dual_requested_other_home_pair",
        "dual_requested_goal_pair",
        "dual_requested_goal_zone_pair",
    ),
    (
        "dual_requested_opposite_axis",
        "dual_requested_front_side",
        "dual_requested_near_far",
        "dual_requested_diagonal",
    ),
    (
        "dual_requested_with_decoy",
        "dual_requested_goal_with_decoy",
        "dual_requested_lane_choice",
        "dual_requested_cross_zone_choice",
    ),
    (
        "dual_requested_home_pair",
        "dual_requested_goal_pair",
        "neutral",
        "neutral",
    ),
)

RWARE_TEAM_SCENARIOS: Tuple[str, ...] = tuple(
    scenario
    for block in RWARE_TEAM_SCENARIO_BLOCKS
    for scenario in block
)


RWARE_SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "team_home_approach": "move toward an own-team requested shelf at home",
    "team_home_pickup": "stand on an own-team requested shelf before pickup",
    "team_goal_approach": "carry a shelf toward the own-team goal",
    "team_goal_at_goal": "carry a shelf while standing on the own-team goal",
    "team_return_approach": "carry a delivered shelf toward its home cell",
    "team_return_at_home": "carry a delivered shelf at its home cell",
    "wrong_team_goal": "carry a shelf at another team's goal zone",
    "team_conflict": "own requested shelf plus an unrequested decoy",
    "dual_requested_home_pair": "own and other requested shelves near own home",
    "dual_requested_other_home_pair": "own and other requested shelves near other home",
    "dual_requested_goal_pair": "two requested shelves near own goal route",
    "dual_requested_goal_zone_pair": "two requested cues near different goal zones",
    "dual_requested_opposite_axis": "two requested shelves on opposite axes",
    "dual_requested_front_side": "two requested shelves in front/side choice",
    "dual_requested_near_far": "near requested shelf versus far requested shelf",
    "dual_requested_diagonal": "two diagonal requested shelves",
    "dual_requested_with_decoy": "two requested shelves plus one unrequested shelf",
    "dual_requested_goal_with_decoy": "goal-zone requested pair plus decoy",
    "dual_requested_lane_choice": "left/right requested lane choice",
    "dual_requested_cross_zone_choice": "own home cue versus other goal cue",
    "neutral": "unrequested shelf without a team-specific cue",
}


@dataclass(frozen=True)
class RwareProbeScenario:
    index: int
    name: str
    team_id: int
    variant: int

    @property
    def slug(self) -> str:
        return f"{self.index:03d}_team{self.team_id}_{self.name}_v{self.variant}"


def sensor_location_index(sensor_range: int, dy: int, dx: int) -> Optional[int]:
    if abs(dy) > sensor_range or abs(dx) > sensor_range:
        return None
    width = 1 + 2 * sensor_range
    return int((dy + sensor_range) * width + (dx + sensor_range))


def sample_cardinal_offsets(
    sensor_range: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if sensor_range <= 0:
        return [(0, 0)]
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    rng.shuffle(offsets)
    return offsets


class RwareObjectiveProbeMapBuilder:
    """Build and render canonical local-observation probes for RWARE."""

    REQUIRED_ENV_ATTRS: Tuple[str, ...] = (
        "_obs_bits_for_self",
        "_obs_bits_per_agent",
        "_obs_bits_per_shelf",
        "msg_bits",
        "sensor_range",
        "grid_size",
    )

    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        missing = [
            name for name in self.REQUIRED_ENV_ATTRS if not hasattr(self.unwrapped, name)
        ]
        if missing:
            raise ValueError(f"unsupported RWARE probe env; missing {missing}")

        self.self_bits = int(self.unwrapped._obs_bits_for_self)
        self.per_agent = int(self.unwrapped._obs_bits_per_agent)
        self.per_shelf = int(self.unwrapped._obs_bits_per_shelf)
        self.sensor_range = int(self.unwrapped.sensor_range)
        self.cell_stride = self.per_agent + self.per_shelf
        self.flat_expected_dim = self.self_bits + (
            (1 + 2 * self.sensor_range) ** 2 * self.cell_stride
        )
        self.semantic_spec: Optional[SemanticObservationSpec] = None
        semantic_spec = get_semantic_observation_spec(env)
        if self.flat_expected_dim == int(obs_dim):
            self.expected_dim = self.flat_expected_dim
        elif semantic_spec.obs_dim == int(obs_dim):
            self.semantic_spec = semantic_spec
            self.expected_dim = semantic_spec.obs_dim
        else:
            raise ValueError(
                f"obs_dim={obs_dim} does not match RWARE probe dims "
                f"flat={self.flat_expected_dim} or semantic={semantic_spec.obs_dim}"
            )

        self.obs_dim = int(obs_dim)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_teams = max(int(getattr(self.unwrapped, "n_teams", 1)), 1)
        self.normalised = bool(
            getattr(self.unwrapped, "normalised_coordinates", False)
        )
        self.probe_env = self._make_probe_env()
        self.probe_env.reset(seed=0)
        self.probe_unwrapped = self.probe_env.unwrapped
        probe_obs_dim = int(self.probe_env.observation_space[0].shape[0])
        expected_probe_dim = (
            self.flat_expected_dim if self.semantic_spec is None else self.flat_expected_dim
        )
        if probe_obs_dim != expected_probe_dim:
            raise ValueError(
                f"custom probe env flat obs_dim={probe_obs_dim} does not match "
                f"expected RWARE flat obs_dim={expected_probe_dim}"
            )

        self.goals_by_team = getattr(self.probe_unwrapped, "goals_by_team", None)
        self.shelfs_by_team = getattr(self.probe_unwrapped, "shelfs_by_team", None)
        self.home_positions = getattr(self.probe_unwrapped, "_shelf_home_positions", {})
        self.height, self.width = tuple(map(int, self.probe_unwrapped.grid_size))
        self.sensor_side = 1 + 2 * self.sensor_range
        self._scenario_cycle = tuple(
            (scenario, team_id)
            for block in RWARE_TEAM_SCENARIO_BLOCKS
            for team_id in range(self.n_teams)
            for scenario in block
        )
        self._shelf_by_id = {
            int(getattr(shelf, "id", 0)): shelf
            for shelf in getattr(self.probe_unwrapped, "shelfs", [])
        }
        self._team_shelf_ids = self._snapshot_team_shelf_ids()

    def _spec_kwargs(self) -> Dict[str, Any]:
        spec = getattr(self.env, "spec", None) or getattr(self.unwrapped, "spec", None)
        kwargs = getattr(spec, "kwargs", None)
        if isinstance(kwargs, dict):
            return deepcopy(kwargs)
        return {}

    def _layout_from_training_env(self) -> Optional[str]:
        highways = getattr(self.unwrapped, "highways", None)
        grid_size = getattr(self.unwrapped, "grid_size", None)
        goals = getattr(self.unwrapped, "goals", None)
        if highways is None or grid_size is None or goals is None:
            return None

        height, width = tuple(map(int, grid_size))
        highway_grid = np.asarray(highways)
        if highway_grid.shape[:2] != (height, width):
            return None

        goal_cells = {
            (int(goal[0]), int(goal[1]))
            for goal in goals
            if len(tuple(goal)) >= 2
        }
        lines: List[str] = []
        for y in range(height):
            chars = []
            for x in range(width):
                if (x, y) in goal_cells:
                    chars.append("g")
                elif bool(highway_grid[y, x]):
                    chars.append(".")
                else:
                    chars.append("x")
            lines.append("".join(chars))
        return "\n".join(lines)

    def _make_probe_env(self) -> MultiTeamWarehouse:
        n_agents = max(int(getattr(self.unwrapped, "n_agents", self.n_teams)), self.n_teams)
        team_reward_mode = getattr(
            self.unwrapped,
            "team_reward_mode",
            TeamRewardMode.INDIVIDUAL,
        )
        kwargs: Dict[str, Any] = {
            "shelf_columns": 3,
            "column_height": int(getattr(self.unwrapped, "column_height", 3) or 3),
            "shelf_rows": 1,
            "n_agents": n_agents,
            "msg_bits": int(getattr(self.unwrapped, "msg_bits", 0)),
            "sensor_range": self.sensor_range,
            "request_queue_size": int(
                getattr(self.unwrapped, "request_queue_size", max(self.n_teams, 2))
            ),
            "max_inactivity_steps": getattr(
                self.unwrapped,
                "max_inactivity_steps",
                None,
            ),
            "max_steps": int(getattr(self.unwrapped, "max_steps", 500) or 500),
            "reward_type": getattr(
                self.unwrapped,
                "reward_type",
                RewardType.INDIVIDUAL,
            ),
            "layout": DEFAULT_RWARE_PROBE_LAYOUT,
            "normalised_coordinates": self.normalised,
            "render_mode": getattr(self.unwrapped, "render_mode", "human"),
            "n_teams": self.n_teams,
            "team_assignments": getattr(self.unwrapped, "_base_agent_team_ids", None),
            "request_queue_size_per_team": list(
                getattr(
                    self.unwrapped,
                    "request_queue_size_per_team",
                    [1 for _ in range(self.n_teams)],
                )
            ),
            "team_reward_mode": team_reward_mode,
            "shelf_team_mode": getattr(self.unwrapped, "shelf_team_mode", "zones"),
            "goal_team_mode": getattr(self.unwrapped, "goal_team_mode", "zones"),
            "require_delivered_shelf_return": bool(
                getattr(self.unwrapped, "require_delivered_shelf_return", False)
            ),
            "reveal_team_info": True,
        }
        kwargs.update(self._spec_kwargs())

        layout = self._layout_from_training_env()
        if layout is not None:
            kwargs["layout"] = layout

        kwargs.update(
            {
                "n_agents": n_agents,
                "msg_bits": int(getattr(self.unwrapped, "msg_bits", 0)),
                "sensor_range": self.sensor_range,
                "request_queue_size": int(
                    getattr(
                        self.unwrapped,
                        "request_queue_size",
                        kwargs.get("request_queue_size", max(self.n_teams, 2)),
                    )
                ),
                "max_inactivity_steps": getattr(
                    self.unwrapped,
                    "max_inactivity_steps",
                    kwargs.get("max_inactivity_steps", None),
                ),
                "max_steps": int(
                    getattr(
                        self.unwrapped,
                        "max_steps",
                        kwargs.get("max_steps", 500),
                    )
                    or 500
                ),
                "reward_type": getattr(
                    self.unwrapped,
                    "reward_type",
                    kwargs.get("reward_type", RewardType.INDIVIDUAL),
                ),
                "normalised_coordinates": self.normalised,
                "render_mode": getattr(
                    self.unwrapped,
                    "render_mode",
                    kwargs.get("render_mode", "human"),
                ),
                "n_teams": self.n_teams,
                "team_assignments": getattr(
                    self.unwrapped,
                    "_base_agent_team_ids",
                    kwargs.get("team_assignments", None),
                ),
                "request_queue_size_per_team": list(
                    getattr(
                        self.unwrapped,
                        "request_queue_size_per_team",
                        kwargs.get(
                            "request_queue_size_per_team",
                            [1 for _ in range(self.n_teams)],
                        ),
                    )
                ),
                "team_reward_mode": team_reward_mode,
                "shelf_team_mode": getattr(
                    self.unwrapped,
                    "shelf_team_mode",
                    kwargs.get("shelf_team_mode", "zones"),
                ),
                "goal_team_mode": getattr(
                    self.unwrapped,
                    "goal_team_mode",
                    kwargs.get("goal_team_mode", "zones"),
                ),
                "require_delivered_shelf_return": bool(
                    getattr(
                        self.unwrapped,
                        "require_delivered_shelf_return",
                        kwargs.get("require_delivered_shelf_return", False),
                    )
                ),
                "reveal_team_info": True,
            }
        )
        return MultiTeamWarehouse(**kwargs)

    @property
    def scenario_cycle(self) -> Tuple[Tuple[str, int], ...]:
        return self._scenario_cycle

    def _snapshot_team_shelf_ids(self) -> List[List[int]]:
        if self.shelfs_by_team is not None:
            team_ids: List[List[int]] = []
            for team_id in range(self.n_teams):
                shelves = (
                    list(self.shelfs_by_team[team_id])
                    if 0 <= team_id < len(self.shelfs_by_team)
                    else []
                )
                team_ids.append(
                    [
                        int(getattr(shelf, "id", 0))
                        for shelf in sorted(
                            shelves,
                            key=lambda shelf: (
                                int(
                                    self.home_positions.get(
                                        int(getattr(shelf, "id", 0)),
                                        (int(getattr(shelf, "x", 0)), int(getattr(shelf, "y", 0))),
                                    )[0]
                                ),
                                int(
                                    self.home_positions.get(
                                        int(getattr(shelf, "id", 0)),
                                        (int(getattr(shelf, "x", 0)), int(getattr(shelf, "y", 0))),
                                    )[1]
                                ),
                                int(getattr(shelf, "id", 0)),
                            ),
                        )
                    ]
                )
            return team_ids

        shelves = sorted(
            getattr(self.unwrapped, "shelfs", []),
            key=lambda shelf: (
                int(getattr(shelf, "x", 0)),
                int(getattr(shelf, "y", 0)),
                int(getattr(shelf, "id", 0)),
            ),
        )
        team_ids = [[] for _ in range(self.n_teams)]
        for idx, shelf in enumerate(shelves):
            team_ids[idx % self.n_teams].append(int(getattr(shelf, "id", 0)))
        return team_ids

    def scenario_for(self, probe_id: int) -> RwareProbeScenario:
        scenario, team_id = self._scenario_cycle[int(probe_id) % len(self._scenario_cycle)]
        variant = int(probe_id) // len(self._scenario_cycle)
        return RwareProbeScenario(
            index=int(probe_id),
            name=scenario,
            team_id=int(team_id),
            variant=variant,
        )

    def scenario_for_team(self, probe_id: int, team_id: int) -> RwareProbeScenario:
        scenario = RWARE_TEAM_SCENARIOS[int(probe_id) % len(RWARE_TEAM_SCENARIOS)]
        variant = int(probe_id) // len(RWARE_TEAM_SCENARIOS)
        return RwareProbeScenario(
            index=int(probe_id),
            name=scenario,
            team_id=int(team_id) % self.n_teams,
            variant=variant,
        )

    def _probe_home(self, shelf: Any) -> Tuple[int, int]:
        shelf_id = int(getattr(shelf, "id", 0))
        home = self.home_positions.get(
            shelf_id,
            (int(getattr(shelf, "x", 0)), int(getattr(shelf, "y", 0))),
        )
        return tuple(map(int, home))

    def _scenario_shelf(self, team_id: int, variant: int) -> Any:
        shelves = self.ordered_team_shelves(team_id)
        if not shelves:
            raise RuntimeError(f"custom probe map has no shelves for team {team_id}")
        return shelves[int(variant) % len(shelves)]

    def _all_other_team_ids(self, team_id: int) -> List[int]:
        others = [idx for idx in range(self.n_teams) if idx != int(team_id)]
        return others or [int(team_id)]

    def other_team_id(self, team_id: int, variant: int = 0) -> int:
        others = self._all_other_team_ids(team_id)
        return int(others[int(variant) % len(others)])

    def _goal_for_team(self, team_id: int, variant: int) -> Tuple[int, int]:
        return self.team_goal(team_id, variant)

    def _reset_probe_state(self) -> None:
        for shelf in getattr(self.probe_unwrapped, "shelfs", []):
            shelf.x, shelf.y = self._probe_home(shelf)
        for agent in getattr(self.probe_unwrapped, "agents", []):
            agent.carrying_shelf = None
            agent.has_delivered = False
            agent.dir = Direction.UP
        if hasattr(self.probe_unwrapped, "team_request_queues"):
            self.probe_unwrapped.team_request_queues = [
                [] for _ in range(self.n_teams)
            ]
        self.probe_unwrapped.request_queue = []
        if hasattr(self.probe_unwrapped, "_shelves_awaiting_return"):
            self.probe_unwrapped._shelves_awaiting_return.clear()
        if hasattr(self.probe_unwrapped, "_shelf_return_team_ids"):
            self.probe_unwrapped._shelf_return_team_ids.clear()

    def _set_focal_agent(
        self,
        x: int,
        y: int,
        direction: Direction,
        team_id: int,
    ) -> Any:
        agent = self.probe_unwrapped.agents[0]
        agent.x = int(np.clip(x, 0, self.width - 1))
        agent.y = int(np.clip(y, 0, self.height - 1))
        agent.dir = direction
        agent.carrying_shelf = None
        if hasattr(self.probe_unwrapped, "agent_team_ids"):
            self.probe_unwrapped.agent_team_ids = np.asarray(
                self.probe_unwrapped.agent_team_ids,
                dtype=np.int32,
            )
            self.probe_unwrapped.agent_team_ids[0] = int(team_id)
        return agent

    def _place_background_agents(self, focal_agent: Any) -> None:
        used = {(int(focal_agent.x), int(focal_agent.y))}
        candidates: List[Tuple[int, int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in used:
                    continue
                if not self.is_highway(x, y):
                    continue
                dist = abs(int(focal_agent.x) - x) + abs(int(focal_agent.y) - y)
                candidates.append((-dist, x, y))
        candidates.sort()
        cursor = 0
        for agent in self.probe_unwrapped.agents[1:]:
            while cursor < len(candidates):
                _, x, y = candidates[cursor]
                cursor += 1
                if (x, y) in used:
                    continue
                agent.x = int(x)
                agent.y = int(y)
                agent.dir = Direction.UP
                agent.carrying_shelf = None
                used.add((int(x), int(y)))
                break

    def _request_shelf(self, shelf: Any, team_id: int) -> None:
        if hasattr(self.probe_unwrapped, "shelf_team_ids"):
            self.probe_unwrapped.shelf_team_ids[int(shelf.id)] = int(team_id)
        if hasattr(self.probe_unwrapped, "team_request_queues"):
            self.probe_unwrapped.team_request_queues[int(team_id)].append(shelf)
            self.probe_unwrapped._sync_global_request_queue()
        elif shelf not in self.probe_unwrapped.request_queue:
            self.probe_unwrapped.request_queue.append(shelf)

    def _place_shelf(
        self,
        shelf: Any,
        x: int,
        y: int,
        team_id: int,
        requested: bool,
    ) -> None:
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        for other in getattr(self.probe_unwrapped, "shelfs", []):
            if int(getattr(other, "id", 0)) == int(getattr(shelf, "id", 0)):
                continue
            if int(getattr(other, "x", -1)) == x and int(getattr(other, "y", -1)) == y:
                other.x = int(self.width - 1)
                other.y = int(self.height - 1)
        shelf.x = x
        shelf.y = y
        if hasattr(self.probe_unwrapped, "shelf_team_ids"):
            self.probe_unwrapped.shelf_team_ids[int(shelf.id)] = int(team_id)
        if requested:
            self._request_shelf(shelf, team_id)

    def _view_cell_for_target(
        self,
        target: Tuple[int, int],
        variant: int,
    ) -> Tuple[int, int, Direction]:
        tx, ty = tuple(map(int, target))
        reach = max(int(self.sensor_range), 1)
        offsets = [
            (0, -reach),
            (0, reach),
            (-reach, 0),
            (reach, 0),
            (-reach, -reach),
            (reach, reach),
            (-reach, reach),
            (reach, -reach),
            (0, -max(reach - 1, 1)),
            (0, max(reach - 1, 1)),
            (-max(reach - 1, 1), 0),
            (max(reach - 1, 1), 0),
        ]
        start = int(variant) % len(offsets)
        for idx in range(len(offsets)):
            dx, dy = offsets[(start + idx) % len(offsets)]
            x = int(tx - dx)
            y = int(ty - dy)
            if 0 <= x < self.width and 0 <= y < self.height and self.is_highway(x, y):
                return x, y, self.direction_from_delta(tx - x, ty - y)
        return self.adjacent_to(target, variant)

    def _dual_row(self, variant: int) -> int:
        rows = [2, 4, 1, 5, 3]
        return int(rows[int(variant) % len(rows)])

    def _dual_state(
        self,
        scenario: RwareProbeScenario,
        include_decoy: bool = False,
        near_goals: bool = False,
    ) -> Any:
        team_id = int(scenario.team_id)
        other_id = self.other_team_id(team_id, scenario.variant)
        own_shelf = self._scenario_shelf(team_id, scenario.variant)
        other_shelf = self._scenario_shelf(other_id, scenario.variant)
        row = 7 if near_goals else self._dual_row(scenario.variant)
        focal_x = int(np.clip(4, self.sensor_range, self.width - self.sensor_range - 1))
        focal = self._set_focal_agent(
            focal_x,
            row,
            Direction.LEFT if team_id % 2 == 0 else Direction.RIGHT,
            team_id,
        )
        left_x = focal_x - self.sensor_range
        right_x = focal_x + self.sensor_range
        own_x = left_x if team_id % 2 == 0 else right_x
        other_x = right_x if team_id % 2 == 0 else left_x
        self._place_shelf(own_shelf, own_x, row, team_id, requested=True)
        self._place_shelf(other_shelf, other_x, row, other_id, requested=True)
        if include_decoy:
            decoy_team = self.other_team_id(team_id, scenario.variant + 1)
            decoy = self._scenario_shelf(decoy_team, scenario.variant + 1)
            decoy_y = int(np.clip(row - 2, 0, self.height - 1))
            self._place_shelf(decoy, focal_x, decoy_y, decoy_team, requested=False)
        return focal

    def _build_realized_probe(self, scenario: RwareProbeScenario) -> np.ndarray:
        self._reset_probe_state()
        team_id = int(scenario.team_id)
        other_id = self.other_team_id(team_id, scenario.variant)
        own_shelf = self._scenario_shelf(team_id, scenario.variant)
        other_shelf = self._scenario_shelf(other_id, scenario.variant)
        own_home = self._probe_home(own_shelf)
        other_home = self._probe_home(other_shelf)
        goal = self._goal_for_team(team_id, scenario.variant)
        other_goal = self._goal_for_team(other_id, scenario.variant)

        if scenario.name == "team_home_pickup":
            focal = self._set_focal_agent(*own_home, Direction.UP, team_id)
            self._place_shelf(own_shelf, *own_home, team_id, requested=True)
        elif scenario.name == "team_home_approach":
            x, y, direction = self._view_cell_for_target(own_home, scenario.variant)
            focal = self._set_focal_agent(x, y, direction, team_id)
            self._place_shelf(own_shelf, *own_home, team_id, requested=True)
        elif scenario.name == "team_goal_approach":
            x, y, direction = self._view_cell_for_target(goal, scenario.variant)
            focal = self._set_focal_agent(x, y, direction, team_id)
            self._place_shelf(own_shelf, x, y, team_id, requested=True)
            focal.carrying_shelf = own_shelf
        elif scenario.name == "team_goal_at_goal":
            focal = self._set_focal_agent(*goal, Direction.UP, team_id)
            self._place_shelf(own_shelf, *goal, team_id, requested=True)
            focal.carrying_shelf = own_shelf
        elif scenario.name == "team_return_approach":
            x, y, direction = self._view_cell_for_target(own_home, scenario.variant)
            focal = self._set_focal_agent(x, y, direction, team_id)
            self._place_shelf(own_shelf, x, y, team_id, requested=True)
            focal.carrying_shelf = own_shelf
            focal.has_delivered = True
        elif scenario.name == "team_return_at_home":
            focal = self._set_focal_agent(*own_home, Direction.UP, team_id)
            self._place_shelf(own_shelf, *own_home, team_id, requested=True)
            focal.carrying_shelf = own_shelf
            focal.has_delivered = True
        elif scenario.name == "wrong_team_goal":
            focal = self._set_focal_agent(*other_goal, Direction.UP, team_id)
            self._place_shelf(own_shelf, *other_goal, team_id, requested=True)
            focal.carrying_shelf = own_shelf
        elif scenario.name == "team_conflict":
            x, y, direction = self._view_cell_for_target(own_home, scenario.variant)
            focal = self._set_focal_agent(x, y, direction, team_id)
            self._place_shelf(own_shelf, *own_home, team_id, requested=True)
            decoy_x = int(np.clip(x + 1, 0, self.width - 1))
            decoy_y = int(np.clip(y + 1, 0, self.height - 1))
            self._place_shelf(other_shelf, decoy_x, decoy_y, other_id, requested=False)
        elif "dual_requested" in scenario.name:
            focal = self._dual_state(
                scenario,
                include_decoy="decoy" in scenario.name,
                near_goals=("goal" in scenario.name or "lane" in scenario.name),
            )
        else:
            focal = self._set_focal_agent(4, 3, Direction.UP, team_id)
            self._place_shelf(other_shelf, 4, 5, other_id, requested=False)

        if hasattr(self.probe_unwrapped, "_shelves_awaiting_return"):
            self.probe_unwrapped._shelves_awaiting_return.clear()
            if "return" in scenario.name and focal.carrying_shelf is not None:
                self.probe_unwrapped._shelves_awaiting_return.add(
                    int(focal.carrying_shelf.id)
                )
        if hasattr(self.probe_unwrapped, "_shelf_return_team_ids"):
            self.probe_unwrapped._shelf_return_team_ids.clear()
            if "return" in scenario.name and focal.carrying_shelf is not None:
                self.probe_unwrapped._shelf_return_team_ids[int(focal.carrying_shelf.id)] = team_id

        self._place_background_agents(focal)
        self.probe_unwrapped._recalc_grid()
        if self.semantic_spec is not None:
            return build_semantic_observation(self.probe_unwrapped, focal).astype(
                np.float32
            )
        return self.probe_unwrapped._make_obs(focal).astype(np.float32)

    def build(self, batch_size: int) -> torch.Tensor:
        probes = [self.build_one(idx) for idx in range(max(int(batch_size), 1))]
        return torch.as_tensor(np.stack(probes, axis=0), dtype=torch.float32)

    def build_for_team(self, team_id: int, batch_size: int) -> torch.Tensor:
        probes = [
            self._build_realized_probe(self.scenario_for_team(idx, team_id))
            for idx in range(max(int(batch_size), 1))
        ]
        return torch.as_tensor(np.stack(probes, axis=0), dtype=torch.float32)

    def build_for_agent_teams(
        self,
        agent_team_ids: Sequence[int],
        batch_size: int,
    ) -> torch.Tensor:
        team_probes = [
            self.build_for_team(int(team_id), batch_size)
            for team_id in agent_team_ids
        ]
        return torch.stack(team_probes, dim=0)

    def build_one(self, probe_id: int) -> np.ndarray:
        scenario = self.scenario_for(probe_id)
        return self._build_realized_probe(scenario)

    def write_position(self, obs: np.ndarray, x: int, y: int) -> None:
        if self.normalised:
            obs[0] = float(x) / float(max(self.width - 1, 1))
            obs[1] = float(y) / float(max(self.height - 1, 1))
        else:
            obs[0] = float(x)
            obs[1] = float(y)

    @staticmethod
    def direction_from_delta(dx: int, dy: int) -> Direction:
        if abs(dx) >= abs(dy):
            return Direction.RIGHT if dx >= 0 else Direction.LEFT
        return Direction.DOWN if dy >= 0 else Direction.UP

    def is_highway(self, x: int, y: int) -> bool:
        env = getattr(self, "probe_unwrapped", self.unwrapped)
        if hasattr(env, "_is_highway"):
            return bool(env._is_highway(int(x), int(y)))
        return False

    def sensor_index(self, dy: int, dx: int) -> Optional[int]:
        return sensor_location_index(self.sensor_range, dy, dx)

    def set_agent(
        self,
        obs: np.ndarray,
        dy: int,
        dx: int,
        direction: Direction,
    ) -> None:
        loc = self.sensor_index(dy, dx)
        if loc is None:
            return
        base = self.self_bits + loc * self.cell_stride
        obs[base] = 1.0
        obs[base + 1 : base + 1 + len(Direction)] = 0.0
        obs[base + 1 + int(direction.value)] = 1.0

    def set_shelf(
        self,
        obs: np.ndarray,
        dy: int,
        dx: int,
        requested: bool,
    ) -> None:
        loc = self.sensor_index(dy, dx)
        if loc is None:
            return
        base = self.self_bits + loc * self.cell_stride + self.per_agent
        obs[base] = 1.0
        obs[base + 1] = 1.0 if requested else 0.0

    def make_probe(
        self,
        x: int,
        y: int,
        carrying: bool,
        direction: Direction,
    ) -> np.ndarray:
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.write_position(obs, x, y)
        obs[2] = 1.0 if carrying else 0.0
        obs[3 + int(direction.value)] = 1.0
        obs[7] = 1.0 if self.is_highway(x, y) else 0.0

        # Match Warehouse._get_default_obs: empty cells carry the default absent
        # agent direction, and the center sensor cell contains the observing agent.
        for loc in range(self.sensor_side**2):
            base = self.self_bits + loc * self.cell_stride
            obs[base + 1] = 1.0
        self.set_agent(obs, 0, 0, direction)
        if carrying:
            self.set_shelf(obs, 0, 0, requested=True)
        return obs

    def ordered_team_shelves(self, team_id: int) -> List[Any]:
        if 0 <= team_id < len(self._team_shelf_ids):
            shelves = [
                self._shelf_by_id[shelf_id]
                for shelf_id in self._team_shelf_ids[team_id]
                if shelf_id in self._shelf_by_id
            ]
            if shelves:
                return shelves
        return [
            self._shelf_by_id[shelf_id]
            for shelf_id in sorted(self._shelf_by_id)
        ]

    def team_shelf_home(self, team_id: int, variant: int) -> Tuple[int, int]:
        shelves = self.ordered_team_shelves(team_id)
        if shelves:
            shelf = shelves[variant % len(shelves)]
            home = self.home_positions.get(
                int(getattr(shelf, "id", -1)),
                (
                    int(getattr(shelf, "x", self.width // 2)),
                    int(getattr(shelf, "y", self.height // 2)),
                ),
            )
            return tuple(map(int, home))
        return int(self.width // 2), int(self.height // 2)

    def team_goal(self, team_id: int, variant: int) -> Tuple[int, int]:
        if self.goals_by_team is not None and 0 <= team_id < len(self.goals_by_team):
            goals = list(self.goals_by_team[team_id])
            if goals:
                return tuple(map(int, goals[variant % len(goals)]))
        goals = list(getattr(self.probe_unwrapped, "goals", []))
        if goals:
            return tuple(map(int, goals[variant % len(goals)]))
        return int(self.width // 2), int(self.height - 1)

    def other_team_id(self, team_id: int, variant: int = 0) -> int:
        others = [idx for idx in range(self.n_teams) if idx != int(team_id)]
        if not others:
            return int(team_id)
        return int(others[int(variant) % len(others)])

    def adjacent_to(
        self,
        target: Tuple[int, int],
        variant: int,
    ) -> Tuple[int, int, Direction]:
        tx, ty = tuple(map(int, target))
        candidates = [
            (tx, ty - 1, Direction.DOWN),
            (tx, ty + 1, Direction.UP),
            (tx - 1, ty, Direction.RIGHT),
            (tx + 1, ty, Direction.LEFT),
        ]
        start = int(variant) % len(candidates)
        for idx in range(len(candidates)):
            x, y, direction = candidates[(start + idx) % len(candidates)]
            if 0 <= x < self.width and 0 <= y < self.height:
                return int(x), int(y), direction
        return (
            int(np.clip(tx, 0, self.width - 1)),
            int(np.clip(ty, 0, self.height - 1)),
            Direction.UP,
        )

    def viewpoint_toward(
        self,
        target: Tuple[int, int],
        variant: int,
        min_distance: Optional[int] = None,
    ) -> Tuple[int, int, Direction]:
        tx, ty = tuple(map(int, target))
        distance = int(
            np.clip(
                self.sensor_range if min_distance is None else min_distance,
                1,
                max(self.sensor_range, 1),
            )
        )
        candidates = [
            (tx, ty - distance),
            (tx, ty + distance),
            (tx - distance, ty),
            (tx + distance, ty),
            (tx - distance, ty - distance),
            (tx + distance, ty + distance),
            (tx - distance, ty + distance),
            (tx + distance, ty - distance),
        ]
        start = int(variant) % len(candidates)
        for idx in range(len(candidates)):
            x, y = candidates[(start + idx) % len(candidates)]
            if 0 <= x < self.width and 0 <= y < self.height:
                direction = self.direction_from_delta(tx - x, ty - y)
                return int(x), int(y), direction
        return self.adjacent_to(target, variant)

    def dual_probe_position(
        self,
        anchor: Tuple[int, int],
        team_id: int,
        variant: int,
    ) -> Tuple[int, int, Direction]:
        reach = max(int(self.sensor_range), 1)
        min_x = min(reach, self.width - 1)
        max_x = max(0, self.width - 1 - reach)
        if min_x > max_x:
            min_x, max_x = 0, self.width - 1
        min_y = min(reach, self.height - 1)
        max_y = max(0, self.height - 1 - reach)
        if min_y > max_y:
            min_y, max_y = 0, self.height - 1

        _, anchor_y = tuple(map(int, anchor))
        x = int(np.clip(round((self.width - 1) / 2.0), min_x, max_x))
        y = int(np.clip(anchor_y, min_y, max_y))
        if variant % 3 == 1:
            y = int(np.clip(round((self.height - 1) / 2.0), min_y, max_y))
        elif variant % 3 == 2:
            y = int(np.clip(anchor_y + (1 if anchor_y < self.height // 2 else -1), min_y, max_y))

        own_dx = -reach if int(team_id) % 2 == 0 else reach
        direction = self.direction_from_delta(own_dx, 0)
        return x, y, direction

    def team_separated_offsets(
        self,
        team_id: int,
        variant: int,
    ) -> List[Tuple[int, int]]:
        reach = max(int(self.sensor_range), 1)
        near = max(2, reach - 1) if reach >= 2 else 1
        own_dx = -reach if int(team_id) % 2 == 0 else reach
        other_dx = -own_dx
        patterns = [
            [(0, own_dx), (0, other_dx)],
            [(-near, own_dx), (near, other_dx)],
            [(near, own_dx), (-near, other_dx)],
            [(-reach, own_dx), (0, other_dx)],
            [(0, own_dx), (reach, other_dx)],
            [(-near, own_dx), (0, other_dx)],
            [(0, own_dx), (near, other_dx)],
            [(-reach, 0), (reach, 0)],
            [(-near, own_dx), (near, 0)],
            [(0, own_dx), (near, 0)],
            [(-near, 0), (near, other_dx)],
        ]
        return [
            (
                int(np.clip(dy, -self.sensor_range, self.sensor_range)),
                int(np.clip(dx, -self.sensor_range, self.sensor_range)),
            )
            for dy, dx in patterns[int(variant) % len(patterns)]
        ]

    def cardinal_offset_toward(
        self,
        probe_x: int,
        probe_y: int,
        target: Tuple[int, int],
        used_offsets: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        tx, ty = tuple(map(int, target))
        dx = int(np.clip(tx - int(probe_x), -self.sensor_range, self.sensor_range))
        dy = int(np.clip(ty - int(probe_y), -self.sensor_range, self.sensor_range))
        used = set(used_offsets or set())
        if dx == 0 and dy == 0 and (0, 0) not in used:
            return 0, 0
        if dx != 0 or dy != 0:
            candidate = (dy, dx)
        else:
            candidate = (0, 0)
        if candidate not in used:
            return candidate
        reach = max(int(self.sensor_range), 1)
        for fallback in [
            (0, reach),
            (reach, 0),
            (0, -reach),
            (-reach, 0),
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (0, 0),
        ]:
            if fallback not in used:
                return fallback
        return candidate

    def add_requested_shelf_toward(
        self,
        obs: np.ndarray,
        probe_x: int,
        probe_y: int,
        target: Tuple[int, int],
        requested: bool = True,
        used_offsets: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        dy, dx = self.cardinal_offset_toward(
            probe_x,
            probe_y,
            target,
            used_offsets=used_offsets,
        )
        self.set_shelf(obs, dy, dx, requested=requested)
        return dy, dx

    def add_requested_shelves_toward(
        self,
        obs: np.ndarray,
        probe_x: int,
        probe_y: int,
        targets: Sequence[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        used_offsets: Set[Tuple[int, int]] = set()
        for target in targets:
            used_offsets.add(
                self.add_requested_shelf_toward(
                    obs,
                    probe_x,
                    probe_y,
                    target,
                    requested=True,
                    used_offsets=used_offsets,
                )
            )
        return used_offsets

    def choice_offset_pairs(self, variant: int) -> List[Tuple[int, int]]:
        reach = max(int(self.sensor_range), 1)
        pairs = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(0, 1), (1, 0)],
            [(0, -1), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)],
            [(0, 1), (0, -reach)],
            [(1, 0), (-reach, 0)],
            [(0, reach), (1, 0)],
            [(reach, 0), (0, -1)],
        ]
        pair = pairs[int(variant) % len(pairs)]
        return [
            (
                int(np.clip(dy, -self.sensor_range, self.sensor_range)),
                int(np.clip(dx, -self.sensor_range, self.sensor_range)),
            )
            for dy, dx in pair
        ]

    def add_requested_pair_at_offsets(
        self,
        obs: np.ndarray,
        offsets: Sequence[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        used_offsets: Set[Tuple[int, int]] = set()
        for dy, dx in offsets:
            offset = (int(dy), int(dx))
            if offset in used_offsets:
                continue
            self.set_shelf(obs, offset[0], offset[1], requested=True)
            used_offsets.add(offset)
        return used_offsets

    def add_unrequested_decoy(
        self,
        obs: np.ndarray,
        used_offsets: Set[Tuple[int, int]],
    ) -> None:
        offsets = sample_cardinal_offsets(self.sensor_range, self.rng)
        offsets.extend([(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)])
        for dy, dx in offsets:
            offset = (
                int(np.clip(dy, -self.sensor_range, self.sensor_range)),
                int(np.clip(dx, -self.sensor_range, self.sensor_range)),
            )
            if offset in used_offsets:
                continue
            self.set_shelf(obs, offset[0], offset[1], requested=False)
            used_offsets.add(offset)
            return

    def decode_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        obs = np.asarray(obs, dtype=np.float32)
        if self.semantic_spec is not None:
            spec = self.semantic_spec
            if self.normalised:
                x = int(round(float(obs[0]) * max(self.width - 1, 1)))
                y = int(round(float(obs[1]) * max(self.height - 1, 1)))
            else:
                x = int(round(float(obs[0])))
                y = int(round(float(obs[1])))

            direction_idx = int(np.argmax(obs[3 : 3 + len(Direction)]))
            spatial = obs[spec.ego_dim :].reshape(
                spec.spatial_channels,
                spec.spatial_size,
                spec.spatial_size,
            )
            cells: List[Dict[str, Any]] = []
            requested_count = 0
            shelf_count = 0
            agent_count = 0
            loaded_agent_count = 0
            for row, dy in enumerate(
                range(-self.sensor_range, self.sensor_range + 1)
            ):
                for col, dx in enumerate(
                    range(-self.sensor_range, self.sensor_range + 1)
                ):
                    has_agent = bool(spatial[5, row, col] > 0.5)
                    has_shelf = bool(spatial[3, row, col] > 0.5)
                    requested = bool(spatial[4, row, col] > 0.5)
                    loaded = bool(spatial[10, row, col] > 0.5)
                    if has_agent:
                        agent_count += 1
                    if loaded:
                        loaded_agent_count += 1
                    if has_shelf:
                        shelf_count += 1
                    if requested:
                        requested_count += 1
                    direction = None
                    if has_agent:
                        direction = Direction(
                            int(np.argmax(spatial[6:10, row, col]))
                        ).name
                    cells.append(
                        {
                            "dy": int(dy),
                            "dx": int(dx),
                            "has_agent": has_agent,
                            "agent_direction": direction,
                            "agent_loaded": loaded,
                            "has_shelf": has_shelf,
                            "shelf_requested": requested,
                        }
                    )
            return {
                "self_x": x,
                "self_y": y,
                "carrying": bool(obs[2] > 0.5),
                "direction": Direction(direction_idx).name,
                "on_highway": bool(obs[7] > 0.5),
                "requested_shelf_count": int(requested_count),
                "shelf_count": int(shelf_count),
                "agent_count": int(agent_count),
                "loaded_agent_count": int(loaded_agent_count),
                "cells": cells,
            }

        if self.normalised:
            x = int(round(float(obs[0]) * max(self.width - 1, 1)))
            y = int(round(float(obs[1]) * max(self.height - 1, 1)))
        else:
            x = int(round(float(obs[0])))
            y = int(round(float(obs[1])))

        direction_idx = int(np.argmax(obs[3 : 3 + len(Direction)]))
        cells: List[Dict[str, Any]] = []
        requested_count = 0
        shelf_count = 0
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                loc = sensor_location_index(self.sensor_range, dy, dx)
                assert loc is not None
                agent_base = self.self_bits + loc * self.cell_stride
                shelf_base = agent_base + self.per_agent
                has_agent = bool(obs[agent_base] > 0.5)
                has_shelf = bool(obs[shelf_base] > 0.5)
                requested = bool(obs[shelf_base + 1] > 0.5)
                if has_shelf:
                    shelf_count += 1
                if has_shelf and requested:
                    requested_count += 1
                cells.append(
                    {
                        "dy": int(dy),
                        "dx": int(dx),
                        "has_agent": has_agent,
                        "agent_direction": (
                            Direction(
                                int(
                                    np.argmax(
                                        obs[agent_base + 1 : agent_base + 1 + len(Direction)]
                                    )
                                )
                            ).name
                            if has_agent
                            else None
                        ),
                        "has_shelf": has_shelf,
                        "shelf_requested": requested,
                    }
                )
        return {
            "self_x": x,
            "self_y": y,
            "carrying": bool(obs[2] > 0.5),
            "direction": Direction(direction_idx).name,
            "on_highway": bool(obs[7] > 0.5),
            "requested_shelf_count": int(requested_count),
            "shelf_count": int(shelf_count),
            "cells": cells,
        }

    @staticmethod
    def _direction_from_name(name: str) -> Direction:
        return Direction[str(name)]

    def _shelf_for_render_team(
        self,
        team_id: int,
        used_shelf_ids: Set[int],
        fallback_variant: int = 0,
    ) -> Optional[Any]:
        team_id = int(np.clip(team_id, 0, self.n_teams - 1))
        ordered_ids = []
        if 0 <= team_id < len(self._team_shelf_ids):
            ordered_ids.extend(self._team_shelf_ids[team_id])
        ordered_ids.extend(sorted(self._shelf_by_id))

        if not ordered_ids:
            return None
        start = int(fallback_variant) % len(ordered_ids)
        for idx in range(len(ordered_ids)):
            shelf_id = int(ordered_ids[(start + idx) % len(ordered_ids)])
            if shelf_id in used_shelf_ids:
                continue
            shelf = self._shelf_by_id.get(shelf_id)
            if shelf is not None:
                used_shelf_ids.add(shelf_id)
                return shelf
        return None

    def _team_for_requested_cell(
        self,
        scenario: RwareProbeScenario,
        cell: Dict[str, Any],
        ordinal: int,
    ) -> int:
        if self.n_teams <= 1:
            return 0
        dx = int(cell["dx"])
        dy = int(cell["dy"])
        own_sign = -1 if int(scenario.team_id) % 2 == 0 else 1
        if abs(dx) >= abs(dy) and dx != 0:
            return (
                int(scenario.team_id)
                if int(np.sign(dx)) == own_sign
                else self.other_team_id(scenario.team_id)
            )
        return int(scenario.team_id) if ordinal % 2 == 0 else self.other_team_id(scenario.team_id)

    def _reset_render_shelves(self) -> None:
        for shelf in getattr(self.unwrapped, "shelfs", []):
            shelf_id = int(getattr(shelf, "id", 0))
            home = self.home_positions.get(
                shelf_id,
                (int(getattr(shelf, "x", 0)), int(getattr(shelf, "y", 0))),
            )
            shelf.x, shelf.y = tuple(map(int, home))

    def _place_background_agents(
        self,
        focal_agent: Any,
        used_positions: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        if used_positions is None:
            used_positions = {(int(focal_agent.x), int(focal_agent.y))}
        candidate_positions: List[Tuple[int, int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                pos = (int(x), int(y))
                if pos in used_positions:
                    continue
                if not self.is_highway(x, y):
                    continue
                distance = abs(int(focal_agent.x) - x) + abs(int(focal_agent.y) - y)
                candidate_positions.append((-distance, x, y))
        if not candidate_positions:
            candidate_positions = [
                (
                    -(abs(int(focal_agent.x) - x) + abs(int(focal_agent.y) - y)),
                    x,
                    y,
                )
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) not in used_positions
            ]
        candidate_positions.sort()
        cursor = 0
        for agent in getattr(self.probe_unwrapped, "agents", [])[1:]:
            while cursor < len(candidate_positions):
                _, x, y = candidate_positions[cursor]
                cursor += 1
                if (x, y) in used_positions:
                    continue
                agent.x = int(x)
                agent.y = int(y)
                agent.dir = Direction.UP
                agent.carrying_shelf = None
                used_positions.add((int(x), int(y)))
                break

    def apply_probe_to_render_env(
        self,
        obs: np.ndarray,
        scenario: Optional[RwareProbeScenario] = None,
    ) -> Dict[str, Any]:
        decoded = self.decode_observation(obs)
        scenario = scenario or RwareProbeScenario(0, "probe", 0, 0)
        agents = list(getattr(self.unwrapped, "agents", []))
        if not agents:
            raise RuntimeError("RWARE render env has no agents; call env.reset() first")

        self._reset_render_shelves()
        focal_agent = agents[0]
        focal_agent.x = int(np.clip(decoded["self_x"], 0, self.width - 1))
        focal_agent.y = int(np.clip(decoded["self_y"], 0, self.height - 1))
        focal_agent.dir = self._direction_from_name(decoded["direction"])
        focal_agent.carrying_shelf = None

        if hasattr(self.unwrapped, "agent_team_ids"):
            self.unwrapped.agent_team_ids = np.asarray(
                self.unwrapped.agent_team_ids,
                dtype=np.int32,
            )
            self.unwrapped.agent_team_ids[0] = int(scenario.team_id)

        if hasattr(self.unwrapped, "team_request_queues"):
            self.unwrapped.team_request_queues = [
                [] for _ in range(max(int(self.n_teams), 1))
            ]
        requested_shelves: List[Any] = []
        used_shelf_ids: Set[int] = set()
        used_positions: Set[Tuple[int, int]] = {
            (int(focal_agent.x), int(focal_agent.y))
        }

        shelf_cells = [
            cell
            for cell in decoded["cells"]
            if bool(cell["has_shelf"])
            and 0 <= int(focal_agent.x) + int(cell["dx"]) < self.width
            and 0 <= int(focal_agent.y) + int(cell["dy"]) < self.height
        ]
        requested_cells = [cell for cell in shelf_cells if bool(cell["shelf_requested"])]
        unrequested_cells = [cell for cell in shelf_cells if not bool(cell["shelf_requested"])]

        for ordinal, cell in enumerate(requested_cells):
            team_id = self._team_for_requested_cell(scenario, cell, ordinal)
            shelf = self._shelf_for_render_team(team_id, used_shelf_ids, scenario.variant + ordinal)
            if shelf is None:
                continue
            wx = int(focal_agent.x) + int(cell["dx"])
            wy = int(focal_agent.y) + int(cell["dy"])
            shelf.x = int(wx)
            shelf.y = int(wy)
            if hasattr(self.unwrapped, "shelf_team_ids"):
                self.unwrapped.shelf_team_ids[int(shelf.id)] = int(team_id)
            if hasattr(self.unwrapped, "team_request_queues"):
                self.unwrapped.team_request_queues[int(team_id)].append(shelf)
            requested_shelves.append(shelf)
            used_positions.add((int(wx), int(wy)))
            if int(cell["dx"]) == 0 and int(cell["dy"]) == 0 and decoded["carrying"]:
                focal_agent.carrying_shelf = shelf

        for ordinal, cell in enumerate(unrequested_cells):
            team_id = (
                self.other_team_id(scenario.team_id)
                if self.n_teams > 1 and ordinal % 2 == 0
                else int(scenario.team_id)
            )
            shelf = self._shelf_for_render_team(
                team_id,
                used_shelf_ids,
                scenario.variant + len(requested_cells) + ordinal,
            )
            if shelf is None:
                continue
            wx = int(focal_agent.x) + int(cell["dx"])
            wy = int(focal_agent.y) + int(cell["dy"])
            shelf.x = int(wx)
            shelf.y = int(wy)
            if hasattr(self.unwrapped, "shelf_team_ids"):
                self.unwrapped.shelf_team_ids[int(shelf.id)] = int(team_id)
            used_positions.add((int(wx), int(wy)))
            if int(cell["dx"]) == 0 and int(cell["dy"]) == 0 and decoded["carrying"]:
                focal_agent.carrying_shelf = shelf

        if decoded["carrying"] and focal_agent.carrying_shelf is None:
            shelf = self._shelf_for_render_team(
                scenario.team_id,
                used_shelf_ids,
                scenario.variant + len(requested_cells) + len(unrequested_cells),
            )
            if shelf is not None:
                shelf.x = int(focal_agent.x)
                shelf.y = int(focal_agent.y)
                focal_agent.carrying_shelf = shelf
                if hasattr(self.unwrapped, "shelf_team_ids"):
                    self.unwrapped.shelf_team_ids[int(shelf.id)] = int(scenario.team_id)
                requested_shelves.append(shelf)
                if hasattr(self.unwrapped, "team_request_queues"):
                    self.unwrapped.team_request_queues[int(scenario.team_id)].append(shelf)

        if hasattr(self.unwrapped, "_shelves_awaiting_return"):
            self.unwrapped._shelves_awaiting_return.clear()
            if "return" in scenario.name and focal_agent.carrying_shelf is not None:
                self.unwrapped._shelves_awaiting_return.add(int(focal_agent.carrying_shelf.id))
        if hasattr(self.unwrapped, "_shelf_return_team_ids"):
            self.unwrapped._shelf_return_team_ids.clear()
            if "return" in scenario.name and focal_agent.carrying_shelf is not None:
                self.unwrapped._shelf_return_team_ids[int(focal_agent.carrying_shelf.id)] = int(scenario.team_id)

        if hasattr(self.unwrapped, "_sync_global_request_queue"):
            self.unwrapped._sync_global_request_queue()
        else:
            self.unwrapped.request_queue = list(requested_shelves)

        self._place_background_agents(focal_agent, used_positions)
        if hasattr(self.unwrapped, "_recalc_grid"):
            self.unwrapped._recalc_grid()

        return decoded

    def render_probe_map(
        self,
        obs: np.ndarray,
        scenario: Optional[RwareProbeScenario] = None,
        cell_size: int = 74,
    ) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required only for probe PNG export. Install pillow to render maps."
            ) from exc

        scenario = scenario or RwareProbeScenario(0, "probe", 0, 0)
        self._build_realized_probe(scenario)
        old_mode = getattr(self.probe_unwrapped, "render_mode", None)
        self.probe_unwrapped.render_mode = "rgb_array"
        try:
            rgb = self.probe_env.render()
        finally:
            if old_mode is not None:
                self.probe_unwrapped.render_mode = old_mode
        if not isinstance(rgb, np.ndarray):
            raise RuntimeError(
                "RWARE render did not return an RGB array; create the env with "
                "render_mode='rgb_array' or use the probe_maps CLI."
            )
        return Image.fromarray(rgb.astype(np.uint8))

    def save_probe_map_images(
        self,
        output_dir: Union[Path, str],
        batch_size: int = 48,
        save_individual: bool = True,
        save_contact_sheet: bool = True,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        images = []

        for idx in range(max(int(batch_size), 1)):
            scenario = self.scenario_for(idx)
            obs = self.build_one(idx)
            decoded = self.decode_observation(obs)
            image = self.render_probe_map(obs, scenario=scenario)
            file_path: Optional[Path] = None
            if save_individual:
                file_path = output_path / f"{scenario.slug}.png"
                image.save(file_path)
            images.append((image, scenario))
            records.append(
                {
                    **asdict(scenario),
                    "description": RWARE_SCENARIO_DESCRIPTIONS.get(
                        scenario.name,
                        scenario.name,
                    ),
                    "image": str(file_path) if file_path is not None else None,
                    "self_x": decoded["self_x"],
                    "self_y": decoded["self_y"],
                    "carrying": decoded["carrying"],
                    "direction": decoded["direction"],
                    "requested_shelf_count": decoded["requested_shelf_count"],
                    "shelf_count": decoded["shelf_count"],
                }
            )

        contact_sheet_path = None
        if save_contact_sheet:
            contact_sheet_path = output_path / "probe_contact_sheet.png"
            _save_contact_sheet(images, contact_sheet_path)

        manifest_path = output_path / "probe_manifest.json"
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return {
            "output_dir": str(output_path),
            "individual_count": len(records) if save_individual else 0,
            "manifest": str(manifest_path),
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
        }


def build_rware_objective_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Optional[torch.Tensor]:
    unwrapped = env.unwrapped
    if not all(
        hasattr(unwrapped, name)
        for name in RwareObjectiveProbeMapBuilder.REQUIRED_ENV_ATTRS
    ):
        return None
    builder = RwareObjectiveProbeMapBuilder(env, obs_dim, rng)
    return builder.build(batch_size)


def build_rware_agent_conditioned_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
    agent_team_ids: Sequence[int],
) -> Optional[torch.Tensor]:
    unwrapped = env.unwrapped
    if not all(
        hasattr(unwrapped, name)
        for name in RwareObjectiveProbeMapBuilder.REQUIRED_ENV_ATTRS
    ):
        return None
    builder = RwareObjectiveProbeMapBuilder(env, obs_dim, rng)
    return builder.build_for_agent_teams(agent_team_ids, batch_size)


def save_rware_probe_map_images(
    env: gym.Env,
    obs_dim: int,
    output_dir: Union[Path, str],
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    save_individual: bool = True,
    save_contact_sheet: bool = True,
) -> Dict[str, Any]:
    builder = RwareObjectiveProbeMapBuilder(
        env,
        obs_dim=obs_dim,
        rng=rng if rng is not None else np.random.default_rng(),
    )
    return builder.save_probe_map_images(
        output_dir,
        batch_size=batch_size,
        save_individual=save_individual,
        save_contact_sheet=save_contact_sheet,
    )


def _load_font(image_font: Any, size: int, bold: bool = False) -> Any:
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf"]
    )
    for name in names:
        try:
            return image_font.truetype(name, size=size)
        except OSError:
            continue
    return image_font.load_default()


def _draw_direction_tick(
    draw: Any,
    cx: float,
    cy: float,
    radius: float,
    direction_name: str,
) -> None:
    vectors = {
        "UP": (0.0, -1.0),
        "DOWN": (0.0, 1.0),
        "LEFT": (-1.0, 0.0),
        "RIGHT": (1.0, 0.0),
    }
    vx, vy = vectors.get(direction_name, (0.0, -1.0))
    start = (cx + vx * radius * 0.2, cy + vy * radius * 0.2)
    end = (cx + vx * radius * 0.95, cy + vy * radius * 0.95)
    draw.line((*start, *end), fill=(255, 255, 255), width=3)


def _legend_item(
    draw: Any,
    x: int,
    y: int,
    label: str,
    color: Tuple[int, int, int],
    text: str,
    font: Any,
) -> None:
    draw.rounded_rectangle((x, y, x + 28, y + 24), radius=5, fill=color)
    draw.text((x + 9, y + 3), label, font=font, fill=(255, 255, 255))
    draw.text((x + 38, y + 3), text, font=font, fill=(68, 77, 92))


def _save_contact_sheet(
    images: Sequence[Tuple[Any, RwareProbeScenario]],
    path: Path,
    columns: int = 4,
) -> None:
    if not images:
        return
    first_image = images[0][0]
    thumb_w = 320
    thumb_h = int(first_image.height * thumb_w / first_image.width)
    pad = 18
    rows = int(np.ceil(len(images) / columns))
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = rows * (thumb_h + 28) + (rows + 1) * pad
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save the probe contact sheet") from exc
    sheet = Image.new("RGB", (sheet_w, sheet_h), (246, 248, 251))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(ImageFont, 13)
    for idx, (image, scenario) in enumerate(images):
        row, col = divmod(idx, columns)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + 28 + pad)
        thumb = image.resize((thumb_w, thumb_h))
        sheet.paste(thumb, (x, y))
        draw.text(
            (x, y + thumb_h + 6),
            f"{scenario.index:03d} team{scenario.team_id} {scenario.name}",
            font=font,
            fill=(50, 58, 72),
        )
    sheet.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save PNG renderings of the canonical RWARE PGCT probe maps."
    )
    parser.add_argument(
        "--env-id",
        default="rware-multiteam-tiny-4ag-2teams-v0",
        help="Gymnasium RWARE environment id.",
    )
    parser.add_argument("--env-kwargs-json", default="{}", help="JSON env kwargs.")
    parser.add_argument(
        "--observation-format",
        default="flat",
        choices=["flat", "semantic"],
        help="Observation format used for the generated probe vectors.",
    )
    parser.add_argument("--sensor-range", type=int, default=3)
    parser.add_argument("--request-queue-size", type=int, default=None)
    parser.add_argument("--request-queue-size-per-team", type=int, default=None)
    parser.add_argument("--require-delivered-shelf-return", action="store_true")
    parser.add_argument("--reveal-team-info", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output-dir", default="output/probe_maps")
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_kwargs = json.loads(args.env_kwargs_json)
    if args.sensor_range is not None:
        env_kwargs["sensor_range"] = int(args.sensor_range)
    if args.request_queue_size is not None:
        env_kwargs["request_queue_size"] = int(args.request_queue_size)
    if args.request_queue_size_per_team is not None:
        env_kwargs["request_queue_size_per_team"] = int(args.request_queue_size_per_team)
    if args.require_delivered_shelf_return:
        env_kwargs["require_delivered_shelf_return"] = True
    if args.reveal_team_info:
        env_kwargs["reveal_team_info"] = True

    env = gym.make(args.env_id, render_mode="rgb_array", **env_kwargs)
    if args.observation_format == "semantic":
        env = RwareSemanticObservationWrapper(env)
    try:
        env.reset(seed=args.seed)
        obs_dim = int(env.observation_space[0].shape[0])
        summary = save_rware_probe_map_images(
            env,
            obs_dim=obs_dim,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            rng=np.random.default_rng(args.seed),
            save_individual=not args.no_individual,
            save_contact_sheet=not args.no_contact_sheet,
        )
    finally:
        env.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
