from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from rware.warehouse import (
    _LAYER_AGENTS,
    _LAYER_SHELFS,
    Action,
    ImageLayer,
    ObservationType,
    RewardType,
    Shelf,
    Warehouse,
)


class TeamRewardMode(Enum):
    TEAM = "team"
    INDIVIDUAL = "individual"
    ALL = "all"


def _coerce_team_reward_mode(mode) -> TeamRewardMode:
    if isinstance(mode, TeamRewardMode):
        return mode
    return TeamRewardMode(str(mode).lower())


def _balanced_assignments(n_agents: int, n_teams: int) -> np.ndarray:
    return np.asarray([agent_id % n_teams for agent_id in range(n_agents)], dtype=np.int32)


def _coerce_team_sizes(value, n_teams: int, fallback_total: int) -> List[int]:
    if value is None:
        per_team = max(1, int(np.ceil(fallback_total / n_teams)))
        return [per_team for _ in range(n_teams)]
    if isinstance(value, int):
        return [value for _ in range(n_teams)]
    values = [int(v) for v in value]
    if len(values) != n_teams:
        raise ValueError("request_queue_size_per_team must have one value per team")
    return values


class MultiTeamWarehouse(Warehouse):
    """RWARE variant with hidden agent teams and team-specific shelf requests.

    The observation and action API remains compatible with the base RWARE
    environment. Hidden agent team IDs are used only for rewards and request
    replacement, so algorithms must infer team structure from local experience.
    """

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType = RewardType.INDIVIDUAL,
        layout: Optional[str] = None,
        observation_type: ObservationType = ObservationType.FLATTENED,
        image_observation_layers: List[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        image_observation_directional: bool = True,
        normalised_coordinates: bool = False,
        render_mode: str = "human",
        n_teams: int = 2,
        team_assignments: Optional[Sequence[int]] = None,
        shuffle_team_assignments_on_reset: bool = False,
        request_queue_size_per_team: Optional[Sequence[int]] = None,
        team_reward_mode: TeamRewardMode = TeamRewardMode.TEAM,
        shelf_team_mode: str = "soft_zones",
        shelf_soft_zone_ratio: float = 0.7,
        shelf_soft_zone_axis: str = "x",
        goal_team_mode: str = "soft_zones",
        team_goal_positions: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        goals_per_team: int = 1,
        soft_goal_separation: float = 0.55,
        require_matching_team_goal: bool = True,
        delivery_reward: float = 1.0,
        wrong_team_penalty: float = 0.0,
        step_penalty: float = 0.0,
        failed_forward_penalty: float = 0.0,
        blocking_agent_penalty: float = 0.0,
        deadlock_penalty: float = 0.0,
        deadlock_penalty_after: int = 5,
        forward_movement_reward: float = 0.0,
        stationary_action_penalty: float = 0.0,
        turn_action_penalty: float = 0.0,
        noop_action_penalty: float = 0.0,
        invalid_toggle_load_penalty: float = 0.0,
        repeated_stationary_action_penalty: float = 0.0,
        stationary_streak_penalty_after: int = 2,
        movement_reward_requires_progress: bool = False,
        new_cell_reward: float = 0.0,
        revisit_cell_penalty: float = 0.0,
        requested_shelf_pickup_reward: float = 0.0,
        requested_shelf_progress_reward: float = 0.0,
        goal_progress_reward: float = 0.0,
        return_progress_reward: float = 0.0,
        reward_only_new_best_progress: bool = True,
        delivered_shelf_drop_reward: float = 0.0,
        require_delivered_shelf_return: bool = False,
        shelf_return_reward: float = 0.0,
        wrong_shelf_return_penalty: float = 0.0,
        premature_drop_penalty: float = 0.0,
        wrong_shelf_pickup_penalty: float = 0.0,
        unrequested_shelf_pickup_penalty: float = 0.0,
        normalize_shaping_rewards: bool = False,
        reveal_team_info: bool = False,
        communication_range: Optional[int] = None,
        team_edge_threshold: float = 0.5,
        use_physical_comm_range: bool = True,
        communication_enabled: bool = True,
        communication_topology: str = "physical",
    ):
        if n_teams < 1:
            raise ValueError("n_teams must be positive")
        if n_teams > n_agents:
            raise ValueError("n_teams cannot exceed n_agents")

        super().__init__(
            shelf_columns=shelf_columns,
            column_height=column_height,
            shelf_rows=shelf_rows,
            n_agents=n_agents,
            msg_bits=msg_bits,
            sensor_range=sensor_range,
            request_queue_size=request_queue_size,
            max_inactivity_steps=max_inactivity_steps,
            max_steps=max_steps,
            reward_type=reward_type,
            layout=layout,
            observation_type=observation_type,
            image_observation_layers=image_observation_layers,
            image_observation_directional=image_observation_directional,
            normalised_coordinates=normalised_coordinates,
            render_mode=render_mode,
        )

        self._base_goals = [tuple(map(int, goal)) for goal in self.goals]

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

        self.n_teams = n_teams
        self._base_agent_team_ids = base_assignments
        self.agent_team_ids = base_assignments.copy()
        self.shuffle_team_assignments_on_reset = shuffle_team_assignments_on_reset
        self.request_queue_size_per_team = _coerce_team_sizes(
            request_queue_size_per_team, n_teams, request_queue_size
        )
        self.team_reward_mode = _coerce_team_reward_mode(team_reward_mode)
        self.shelf_team_mode = str(shelf_team_mode).lower()
        self.shelf_soft_zone_ratio = float(shelf_soft_zone_ratio)
        self.shelf_soft_zone_axis = str(shelf_soft_zone_axis).lower()
        self.goal_team_mode = str(goal_team_mode).lower()
        self._manual_team_goal_positions = team_goal_positions
        self.goals_per_team = int(goals_per_team)
        self.soft_goal_separation = float(soft_goal_separation)
        self.require_matching_team_goal = bool(require_matching_team_goal)

        if not 0.0 <= self.shelf_soft_zone_ratio <= 1.0:
            raise ValueError("shelf_soft_zone_ratio must be in [0, 1]")
        if self.shelf_soft_zone_axis not in ["x", "y"]:
            raise ValueError("shelf_soft_zone_axis must be 'x' or 'y'")
        if self.goals_per_team < 1:
            raise ValueError("goals_per_team must be positive")
        if not 0.0 <= self.soft_goal_separation <= 1.0:
            raise ValueError("soft_goal_separation must be in [0, 1]")
        self.delivery_reward = float(delivery_reward)
        self.wrong_team_penalty = float(wrong_team_penalty)
        self.step_penalty = float(step_penalty)
        self.failed_forward_penalty = float(failed_forward_penalty)
        self.blocking_agent_penalty = float(blocking_agent_penalty)
        self.deadlock_penalty = float(deadlock_penalty)
        self.deadlock_penalty_after = int(deadlock_penalty_after)
        self.forward_movement_reward = float(forward_movement_reward)
        self.stationary_action_penalty = float(stationary_action_penalty)
        self.turn_action_penalty = float(turn_action_penalty)
        self.noop_action_penalty = float(noop_action_penalty)
        self.invalid_toggle_load_penalty = float(invalid_toggle_load_penalty)
        self.repeated_stationary_action_penalty = float(repeated_stationary_action_penalty)
        self.stationary_streak_penalty_after = int(stationary_streak_penalty_after)
        self.movement_reward_requires_progress = bool(movement_reward_requires_progress)
        self.new_cell_reward = float(new_cell_reward)
        self.revisit_cell_penalty = float(revisit_cell_penalty)
        self.requested_shelf_pickup_reward = float(requested_shelf_pickup_reward)
        self.requested_shelf_progress_reward = float(requested_shelf_progress_reward)
        self.goal_progress_reward = float(goal_progress_reward)
        self.return_progress_reward = float(return_progress_reward)
        self.reward_only_new_best_progress = bool(reward_only_new_best_progress)
        self.delivered_shelf_drop_reward = float(delivered_shelf_drop_reward)
        self.require_delivered_shelf_return = bool(require_delivered_shelf_return)
        self.shelf_return_reward = float(shelf_return_reward)
        self.wrong_shelf_return_penalty = float(wrong_shelf_return_penalty)
        self.premature_drop_penalty = float(premature_drop_penalty)
        self.wrong_shelf_pickup_penalty = float(wrong_shelf_pickup_penalty)
        self.unrequested_shelf_pickup_penalty = float(unrequested_shelf_pickup_penalty)
        self.normalize_shaping_rewards = bool(normalize_shaping_rewards)
        for name in [
            "requested_shelf_pickup_reward",
            "requested_shelf_progress_reward",
            "goal_progress_reward",
            "return_progress_reward",
            "delivered_shelf_drop_reward",
            "shelf_return_reward",
            "wrong_shelf_return_penalty",
            "premature_drop_penalty",
            "blocking_agent_penalty",
            "deadlock_penalty",
            "forward_movement_reward",
            "stationary_action_penalty",
            "turn_action_penalty",
            "noop_action_penalty",
            "invalid_toggle_load_penalty",
            "repeated_stationary_action_penalty",
            "new_cell_reward",
            "revisit_cell_penalty",
            "wrong_shelf_pickup_penalty",
            "unrequested_shelf_pickup_penalty",
        ]:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.stationary_streak_penalty_after < 0:
            raise ValueError("stationary_streak_penalty_after must be non-negative")
        if self.deadlock_penalty_after < 0:
            raise ValueError("deadlock_penalty_after must be non-negative")
        self.reveal_team_info = reveal_team_info

        self.shelf_team_ids: Dict[int, int] = {}
        self.shelfs_by_team: List[List[Shelf]] = [[] for _ in range(n_teams)]
        self.team_request_queues: List[List[Shelf]] = [[] for _ in range(n_teams)]
        self.goal_team_ids: Dict[Tuple[int, int], int] = {}
        self.goals_by_team: List[List[Tuple[int, int]]] = [[] for _ in range(n_teams)]
        self.team_delivery_counts = np.zeros(n_teams, dtype=np.int64)
        self.team_return_counts = np.zeros(n_teams, dtype=np.int64)
        self._last_delivery_events = []
        self._last_return_events = []
        self._last_wrong_return_events = []
        self._last_pickup_events = []
        self._last_invalid_toggle_events = []
        self._last_failed_forward_events = []
        self._last_blocking_agent_events = []
        self._last_deadlock_events = []
        self._last_reward_shaping = np.zeros(self.n_agents, dtype=np.float32)
        self._last_requested_actions = np.full(
            self.n_agents,
            int(Action.NOOP.value),
            dtype=np.int32,
        )
        self._consecutive_failed_forward_steps = np.zeros(
            self.n_agents,
            dtype=np.int32,
        )
        self._picked_request_shelf_ids_by_agent: List[Set[int]] = [
            set() for _ in range(self.n_agents)
        ]
        self._shelf_home_positions: Dict[int, Tuple[int, int]] = {}
        self._shelves_awaiting_return: Set[int] = set()
        self._shelf_return_team_ids: Dict[int, int] = {}
        self._visited_cells_by_agent: List[Set[Tuple[int, int]]] = [
            set() for _ in range(self.n_agents)
        ]
        self._stationary_action_streak = np.zeros(self.n_agents, dtype=np.int32)
        self._best_requested_shelf_distances: List[Optional[int]] = [
            None for _ in range(self.n_agents)
        ]
        self._best_goal_distances: List[Optional[int]] = [
            None for _ in range(self.n_agents)
        ]
        self._best_return_distances: List[Optional[int]] = [
            None for _ in range(self.n_agents)
        ]
        self._best_delivery_distances_by_agent: List[Dict[int, int]] = [
            {} for _ in range(self.n_agents)
        ]
        self._best_return_distances_by_agent: List[Dict[int, int]] = [
            {} for _ in range(self.n_agents)
        ]

        # Learned / inferred dynamic communication graph.
        # This is for training-time selective sharing or debugging. It is not
        # included in the agents' observations.
        self.communication_range = int(sensor_range if communication_range is None else communication_range)
        if communication_topology not in {"physical", "complete", "none"}:
            raise ValueError("unknown communication_topology")
        if self.communication_range < 0:
            raise ValueError("Communication range must be nonnegative")
        self.communication_enabled = bool(communication_enabled)
        self.communication_topology = communication_topology
        self._training_comm_weights = None
        self.training_comm_directed = False
        self.team_edge_threshold = float(team_edge_threshold)
        self.use_physical_comm_range = bool(use_physical_comm_range)
        self.inferred_team_ids = -np.ones(self.n_agents, dtype=np.int32)
        self.inferred_team_confidence = np.zeros(self.n_agents, dtype=np.float32)
        self.training_comm_adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.training_comm_edges: List[Tuple[int, int]] = []

    def get_oracle_team_assignments(self) -> np.ndarray:
        return self.agent_team_ids.copy()

    def get_team_members(self) -> List[List[int]]:
        return [
            np.flatnonzero(self.agent_team_ids == team_id).astype(np.int32).tolist()
            for team_id in range(self.n_teams)
        ]

    def _home_zone_id(self, x: int, y: int) -> int:
        """Return the spatial home zone id for a grid cell.

        This is used only to create task distributions. The team id is still
        hidden from observations unless the user enables debug info/rendering.
        """
        height, width = self.grid_size
        if self.shelf_soft_zone_axis == "y":
            coord = int(y)
            extent = int(height)
        else:
            coord = int(x)
            extent = int(width)

        zone_edges = np.linspace(0, extent, self.n_teams + 1)
        zone_id = int(np.searchsorted(zone_edges, coord, side="right") - 1)
        return int(np.clip(zone_id, 0, self.n_teams - 1))

    def _assign_shelf_teams(self):
        self.shelf_team_ids = {}
        self.shelfs_by_team = [[] for _ in range(self.n_teams)]

        if self.shelf_team_mode == "balanced_soft_zones":
            if self.n_teams != 2:
                raise ValueError("balanced_soft_zones requires two teams")
            zones = [[shelf for shelf in self.shelfs
                      if self._home_zone_id(shelf.x, shelf.y) == team]
                     for team in range(2)]
            if len(zones[0]) != len(zones[1]):
                raise ValueError("balanced_soft_zones requires equal-sized shelf zones")
            swap_count = int(round(len(zones[0]) * (1.0 - self.shelf_soft_zone_ratio)))
            for team, shelves in enumerate(zones):
                order = self.np_random.permutation(len(shelves))
                swapped = set(order[:swap_count].tolist())
                for idx, shelf in enumerate(shelves):
                    owner = 1 - team if idx in swapped else team
                    self.shelf_team_ids[shelf.id] = owner
                    self.shelfs_by_team[owner].append(shelf)
            return

        if self.shelf_team_mode == "round_robin":
            ordered_shelfs = sorted(self.shelfs, key=lambda shelf: (shelf.x, shelf.y, shelf.id))
            for idx, shelf in enumerate(ordered_shelfs):
                team_id = idx % self.n_teams
                self.shelf_team_ids[shelf.id] = team_id
                self.shelfs_by_team[team_id].append(shelf)
            return

        if self.shelf_team_mode not in ["zones", "soft_zones"]:
            raise ValueError("shelf_team_mode must be 'zones', 'soft_zones', or 'round_robin'")

        for shelf in self.shelfs:
            home_team = self._home_zone_id(shelf.x, shelf.y)

            if self.shelf_team_mode == "soft_zones" and self.n_teams > 1:
                # With probability shelf_soft_zone_ratio, keep the shelf assigned
                # to its spatial home zone. Otherwise assign it to another team.
                # Example: ratio=0.7 gives roughly 70:30 mixing for two teams.
                if float(self.np_random.random()) < self.shelf_soft_zone_ratio:
                    team_id = home_team
                else:
                    other_teams = [team for team in range(self.n_teams) if team != home_team]
                    team_id = int(self.np_random.choice(other_teams))
            else:
                team_id = home_team

            self.shelf_team_ids[shelf.id] = int(team_id)
            self.shelfs_by_team[int(team_id)].append(shelf)

        # Make sure every team owns at least one shelf so every team can have
        # a non-empty request queue. This is especially important for small maps.
        empty_teams = [idx for idx, shelfs in enumerate(self.shelfs_by_team) if not shelfs]
        if empty_teams:
            ordered_shelfs = sorted(self.shelfs, key=lambda shelf: (shelf.x, shelf.y, shelf.id))
            for missing_team in empty_teams:
                donor_team = int(np.argmax([len(shelfs) for shelfs in self.shelfs_by_team]))
                if not self.shelfs_by_team[donor_team]:
                    shelf = ordered_shelfs[missing_team % len(ordered_shelfs)]
                else:
                    shelf = self.shelfs_by_team[donor_team].pop()
                self.shelf_team_ids[shelf.id] = int(missing_team)
                self.shelfs_by_team[missing_team].append(shelf)

    def get_shelf_zone_team_counts(self) -> np.ndarray:
        """Debug helper: counts[home_zone, assigned_team]."""
        counts = np.zeros((self.n_teams, self.n_teams), dtype=np.int64)
        for shelf in self.shelfs:
            home_zone = self._home_zone_id(shelf.x, shelf.y)
            assigned_team = self.shelf_team_ids.get(shelf.id)
            if assigned_team is not None:
                counts[home_zone, int(assigned_team)] += 1
        return counts

    def _nearest_highway_cell(self, target_x: int, target_y: int, used: set) -> Tuple[int, int]:
        height, width = self.grid_size
        candidates = []
        for y in range(height):
            for x in range(width):
                if (x, y) in used:
                    continue
                if self._is_highway(x, y):
                    dist = abs(x - target_x) + abs(y - target_y)
                    candidates.append((dist, x, y))

        if candidates:
            _, x, y = min(candidates, key=lambda item: item[0])
            return int(x), int(y)

        # Fallback: if the layout has no highway cells for some reason.
        x = int(np.clip(target_x, 0, width - 1))
        y = int(np.clip(target_y, 0, height - 1))
        return x, y

    def _make_default_team_goals(self) -> List[List[Tuple[int, int]]]:
        """Create one reachable goal position per team, spread along the x-axis."""
        height, width = self.grid_size
        base_goals = self._base_goals or [(width // 2, height - 1)]
        used = set()
        goals_by_team: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_teams)]

        for team_id in range(self.n_teams):
            target_x = int(round((team_id + 0.5) * width / self.n_teams - 0.5))
            target_y = int(base_goals[team_id % len(base_goals)][1])
            goal = self._nearest_highway_cell(target_x, target_y, used)
            used.add(goal)
            goals_by_team[team_id].append(goal)

        return goals_by_team

    def _make_soft_zone_team_goals(self) -> List[List[Tuple[int, int]]]:
        """Create moderately separated team goals instead of far corner goals.

        soft_goal_separation=0.0 puts all team goals near the global center;
        soft_goal_separation=1.0 puts them near each team's full zone center.
        Values around 0.45--0.65 are useful for latent-team experiments:
        agents share the same warehouse space, but long-term trajectories still
        contain a weak team-specific directional signal.
        """
        height, width = self.grid_size
        base_goals = [tuple(map(int, goal)) for goal in (self._base_goals or [])]
        base_y_values = sorted({int(goal[1]) for goal in base_goals})
        if not base_y_values:
            base_y_values = [height - 1]
        if len(base_y_values) == 1 and self.goals_per_team > 1:
            # Use top/bottom lanes when the base layout exposes only one y-level.
            base_y_values = sorted({0, height - 1})

        used = set()
        goals_by_team: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_teams)]
        global_center_x = (width - 1) / 2.0

        for team_id in range(self.n_teams):
            zone_center_x = (team_id + 0.5) * width / self.n_teams - 0.5
            target_x_base = global_center_x + (zone_center_x - global_center_x) * self.soft_goal_separation

            for k in range(self.goals_per_team):
                y = base_y_values[k % len(base_y_values)]
                # Small alternating offset prevents multiple goals from collapsing
                # to the exact same cell when goals_per_team > 1.
                offset = (k // len(base_y_values)) - ((self.goals_per_team - 1) / 2.0)
                target_x = int(round(target_x_base + offset))
                goal = self._nearest_highway_cell(target_x, int(y), used)
                used.add(goal)
                goals_by_team[team_id].append(goal)

        return goals_by_team

    def _assign_existing_goals_to_teams(self) -> Optional[List[List[Tuple[int, int]]]]:
        """Split existing RWARE goals by zone or round-robin when enough goals exist."""
        goals = [tuple(map(int, goal)) for goal in self._base_goals]
        if len(goals) < self.n_teams:
            return None

        goals_by_team: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_teams)]

        if self.goal_team_mode == "round_robin":
            for idx, goal in enumerate(sorted(goals, key=lambda g: (g[0], g[1]))):
                goals_by_team[idx % self.n_teams].append(goal)
            return goals_by_team

        if self.goal_team_mode not in ["auto", "zones"]:
            raise ValueError("goal_team_mode must be 'auto', 'zones', 'soft_zones', or 'round_robin'")

        width = self.grid_size[1]
        zone_edges = np.linspace(0, width, self.n_teams + 1)
        for goal in goals:
            x, _ = goal
            team_id = int(np.searchsorted(zone_edges, x, side="right") - 1)
            team_id = int(np.clip(team_id, 0, self.n_teams - 1))
            goals_by_team[team_id].append(goal)

        if any(len(team_goals) == 0 for team_goals in goals_by_team):
            goals_by_team = [[] for _ in range(self.n_teams)]
            for idx, goal in enumerate(sorted(goals, key=lambda g: (g[0], g[1]))):
                goals_by_team[idx % self.n_teams].append(goal)

        return goals_by_team

    def _assign_team_goals(self):
        """Set env.goals and create goal -> team mapping.

        If team_goal_positions is provided, use it directly.
        Otherwise, split existing goals when possible; if there are too few
        existing goals, generate one reachable goal position per team.
        """
        if self._manual_team_goal_positions is not None:
            values = list(self._manual_team_goal_positions)
            if len(values) != self.n_teams:
                raise ValueError("team_goal_positions must contain one goal list per team")
            goals_by_team = []
            for team_id, team_goals in enumerate(values):
                coerced = [tuple(map(int, goal)) for goal in team_goals]
                if not coerced:
                    raise ValueError(f"Team {team_id} must have at least one goal")
                goals_by_team.append(coerced)
        else:
            if self.goal_team_mode == "soft_zones":
                goals_by_team = self._make_soft_zone_team_goals()
            else:
                goals_by_team = self._assign_existing_goals_to_teams()
                if goals_by_team is None:
                    goals_by_team = self._make_default_team_goals()

        self.goals_by_team = goals_by_team
        self.goal_team_ids = {}
        merged_goals = []
        for team_id, team_goals in enumerate(self.goals_by_team):
            for goal in team_goals:
                goal = tuple(map(int, goal))
                self.goal_team_ids[goal] = int(team_id)
                merged_goals.append(goal)

        self.goals = merged_goals

    def _make_team_request_queues(self):
        self.team_request_queues = []
        for team_id in range(self.n_teams):
            candidates = self.shelfs_by_team[team_id]
            queue_size = min(self.request_queue_size_per_team[team_id], len(candidates))
            if queue_size <= 0:
                self.team_request_queues.append([])
                continue
            requested = self.np_random.choice(candidates, size=queue_size, replace=False)
            self.team_request_queues.append(list(requested))
        self._sync_global_request_queue()

    def _sync_global_request_queue(self):
        self.request_queue = [
            shelf for team_queue in self.team_request_queues for shelf in team_queue
        ]

    def _remember_shelf_home_positions(self) -> None:
        self._shelf_home_positions = {
            shelf.id: (int(shelf.x), int(shelf.y))
            for shelf in self.shelfs
        }

    def _is_shelf_at_home(self, shelf: Shelf) -> bool:
        home = self._shelf_home_positions.get(shelf.id)
        return home == (int(shelf.x), int(shelf.y))

    def _replace_team_request(self, team_id: int, delivered_shelf: Shelf):
        team_queue = self.team_request_queues[team_id]
        if delivered_shelf not in team_queue:
            return
        candidates = [
            shelf
            for shelf in self.shelfs_by_team[team_id]
            if shelf not in team_queue
        ]
        replacement = (
            self.np_random.choice(candidates)
            if candidates
            else delivered_shelf
        )
        team_queue[team_queue.index(delivered_shelf)] = replacement
        for picked_shelf_ids in self._picked_request_shelf_ids_by_agent:
            picked_shelf_ids.discard(delivered_shelf.id)
        self._sync_global_request_queue()
        self._reset_progress_baselines()

    def _reward_delivery(self, rewards: np.ndarray, team_id: int, carrier_id: int) -> List[int]:
        rewarded_agents: List[int] = []
        if self.team_reward_mode == TeamRewardMode.ALL:
            rewards += self.delivery_reward
            rewarded_agents = list(range(self.n_agents))
        elif self.team_reward_mode == TeamRewardMode.INDIVIDUAL:
            if carrier_id > 0 and self.agent_team_ids[carrier_id - 1] == team_id:
                rewards[carrier_id - 1] += self.delivery_reward
                rewarded_agents = [carrier_id - 1]
        elif self.team_reward_mode == TeamRewardMode.TEAM:
            team_members = np.flatnonzero(self.agent_team_ids == team_id)
            rewards[team_members] += self.delivery_reward
            rewarded_agents = team_members.astype(np.int32).tolist()
        else:
            raise ValueError(f"Unsupported team reward mode: {self.team_reward_mode}")

        if (
            self.wrong_team_penalty
            and carrier_id > 0
            and self.agent_team_ids[carrier_id - 1] != team_id
        ):
            rewards[carrier_id - 1] -= self.wrong_team_penalty

        return rewarded_agents

    def _reward_shelf_return(self, rewards: np.ndarray, team_id: int, carrier_id: int) -> List[int]:
        rewarded_agents: List[int] = []
        if self.team_reward_mode == TeamRewardMode.ALL:
            rewards += self.shelf_return_reward
            rewarded_agents = list(range(self.n_agents))
        elif self.team_reward_mode == TeamRewardMode.INDIVIDUAL:
            if carrier_id > 0 and self.agent_team_ids[carrier_id - 1] == team_id:
                rewards[carrier_id - 1] += self.shelf_return_reward
                rewarded_agents = [carrier_id - 1]
        elif self.team_reward_mode == TeamRewardMode.TEAM:
            team_members = np.flatnonzero(self.agent_team_ids == team_id)
            rewards[team_members] += self.shelf_return_reward
            rewarded_agents = team_members.astype(np.int32).tolist()
        else:
            raise ValueError(f"Unsupported team reward mode: {self.team_reward_mode}")
        return rewarded_agents

    def _agent_team_id(self, agent) -> int:
        return int(self.agent_team_ids[agent.id - 1])

    def _requested_shelves_for_agent(self, agent) -> List[Shelf]:
        team_id = self._agent_team_id(agent)
        if not (0 <= team_id < len(self.team_request_queues)):
            return []
        return list(self.team_request_queues[team_id])

    def _is_agent_requested_shelf(self, agent, shelf: Shelf) -> bool:
        return shelf in self._requested_shelves_for_agent(agent)

    def _is_requested_shelf(self, shelf: Shelf) -> bool:
        return any(shelf in team_queue for team_queue in self.team_request_queues)

    def _requested_shelves_for_observation(self, agent):
        return self._requested_shelves_for_agent(agent)

    def _distance(self, source_x: int, source_y: int, target: Tuple[int, int]) -> int:
        target_x, target_y = target
        return abs(int(source_x) - int(target_x)) + abs(int(source_y) - int(target_y))

    def _distance_scale(self) -> float:
        height, width = self.grid_size
        return float(max(int(height) + int(width) - 2, 1))

    def _nearest_requested_shelf_distance(self, agent) -> Optional[int]:
        shelves = self._available_requested_shelves(agent)
        if not shelves:
            return None
        return min(
            self._distance(agent.x, agent.y, (shelf.x, shelf.y))
            for shelf in shelves
        )

    def _available_requested_shelves(self, agent) -> List[Shelf]:
        carried_ids = {
            other.carrying_shelf.id for other in self.agents
            if other.carrying_shelf is not None
        }
        return [shelf for shelf in self._requested_shelves_for_agent(agent)
                if shelf.id not in carried_ids]

    def _goals_for_shelf_team(self, team_id: int) -> List[Tuple[int, int]]:
        if self.require_matching_team_goal and 0 <= team_id < len(self.goals_by_team):
            return list(self.goals_by_team[team_id])
        return [tuple(map(int, goal)) for goal in self.goals]

    def _carried_requested_shelf_delivery_goal_distance(self, agent) -> Optional[int]:
        shelf = agent.carrying_shelf
        if shelf is None or not self._is_agent_requested_shelf(agent, shelf):
            return None
        team_id = int(self.shelf_team_ids.get(shelf.id, self._agent_team_id(agent)))
        goals = self._goals_for_shelf_team(team_id)
        if not goals:
            return None
        return min(self._distance(agent.x, agent.y, goal) for goal in goals)

    def _carried_requested_shelf_return_distance(self, agent) -> Optional[int]:
        shelf = agent.carrying_shelf
        if shelf is None or not self._is_agent_requested_shelf(agent, shelf):
            return None
        if shelf.id not in self._shelves_awaiting_return:
            return None
        home = self._shelf_home_positions.get(shelf.id)
        if home is None:
            return None
        return self._distance(agent.x, agent.y, home)

    def _carried_requested_shelf_goal_distance(self, agent) -> Optional[int]:
        if (
            agent.carrying_shelf is not None
            and agent.carrying_shelf.id in self._shelves_awaiting_return
        ):
            return self._carried_requested_shelf_return_distance(agent)
        return self._carried_requested_shelf_delivery_goal_distance(agent)

    def _reset_progress_baselines(self) -> None:
        for agent_id, agent in enumerate(self.agents):
            self._best_delivery_distances_by_agent[agent_id].clear()
            self._best_return_distances_by_agent[agent_id].clear()
            self._best_requested_shelf_distances[agent_id] = (
                self._nearest_requested_shelf_distance(agent)
                if agent.carrying_shelf is None
                else None
            )
            if agent.carrying_shelf is None:
                self._best_goal_distances[agent_id] = None
                self._best_return_distances[agent_id] = None
            elif agent.carrying_shelf.id in self._shelves_awaiting_return:
                self._best_goal_distances[agent_id] = None
                self._best_return_distances[agent_id] = (
                    self._carried_requested_shelf_return_distance(agent)
                )
                if self._best_return_distances[agent_id] is not None:
                    self._best_return_distances_by_agent[agent_id][
                        int(agent.carrying_shelf.id)
                    ] = int(self._best_return_distances[agent_id])
            else:
                self._best_goal_distances[agent_id] = (
                    self._carried_requested_shelf_delivery_goal_distance(agent)
                )
                self._best_return_distances[agent_id] = None
                if self._best_goal_distances[agent_id] is not None:
                    self._best_delivery_distances_by_agent[agent_id][
                        int(agent.carrying_shelf.id)
                    ] = int(self._best_goal_distances[agent_id])

    def _progress_reward(
        self,
        before_distance: Optional[int],
        after_distance: Optional[int],
        coefficient: float,
    ) -> float:
        if coefficient <= 0.0 or before_distance is None or after_distance is None:
            return 0.0
        progress = float(before_distance - after_distance)
        if self.normalize_shaping_rewards:
            progress /= self._distance_scale()
        return float(coefficient * progress)

    def _new_best_progress_reward(
        self,
        best_distances: List[Optional[int]],
        agent_id: int,
        after_distance: Optional[int],
        coefficient: float,
    ) -> float:
        if coefficient <= 0.0 or after_distance is None:
            return 0.0

        best_distance = best_distances[agent_id]
        if best_distance is None:
            best_distances[agent_id] = int(after_distance)
            return 0.0
        if after_distance >= best_distance:
            return 0.0

        progress = float(best_distance - after_distance)
        best_distances[agent_id] = int(after_distance)
        if self.normalize_shaping_rewards:
            progress /= self._distance_scale()
        return float(coefficient * progress)

    def _new_best_shelf_progress_reward(
        self,
        best_distances_by_agent: List[Dict[int, int]],
        scalar_best_distances: List[Optional[int]],
        agent_id: int,
        shelf_id: int,
        after_distance: Optional[int],
        coefficient: float,
    ) -> float:
        if coefficient <= 0.0 or after_distance is None:
            return 0.0

        shelf_id = int(shelf_id)
        best_distances = best_distances_by_agent[agent_id]
        best_distance = best_distances.get(shelf_id)
        if best_distance is None:
            best_distances[shelf_id] = int(after_distance)
            scalar_best_distances[agent_id] = int(after_distance)
            return 0.0
        if after_distance >= best_distance:
            scalar_best_distances[agent_id] = int(best_distance)
            return 0.0

        progress = float(best_distance - after_distance)
        best_distances[shelf_id] = int(after_distance)
        scalar_best_distances[agent_id] = int(after_distance)
        if self.normalize_shaping_rewards:
            progress /= self._distance_scale()
        return float(coefficient * progress)

    def _pickup_shaping_reward(self, agent, shelf: Shelf) -> float:
        if self._is_agent_requested_shelf(agent, shelf):
            picked_shelf_ids = self._picked_request_shelf_ids_by_agent[agent.id - 1]
            if shelf.id in picked_shelf_ids:
                return 0.0
            picked_shelf_ids.add(shelf.id)
            return self.requested_shelf_pickup_reward
        if self._is_requested_shelf(shelf):
            return -self.wrong_shelf_pickup_penalty
        return -self.unrequested_shelf_pickup_penalty

    def set_inferred_team_assignments(
        self,
        inferred_team_ids,
        confidence=None,
        update_graph: bool = True,
    ):
        """Set learned team predictions and update the training communication graph.

        Use this from the learning algorithm. Do not use true labels here during
        actual training, except for oracle/debug baselines.
        """
        inferred_team_ids = np.asarray(inferred_team_ids, dtype=np.int32)
        if inferred_team_ids.shape != (self.n_agents,):
            raise ValueError("inferred_team_ids must have shape (n_agents,)")
        self.inferred_team_ids = inferred_team_ids.copy()

        if confidence is None:
            self.inferred_team_confidence = np.ones(self.n_agents, dtype=np.float32)
        else:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != (self.n_agents,):
                raise ValueError("confidence must have shape (n_agents,)")
            self.inferred_team_confidence = confidence.copy()

        if update_graph:
            self.update_training_comm_graph()
        return self.training_comm_adj.copy()

    def set_training_comm_edges(self, edges):
        """Manually set communication edges for debugging or visualization."""
        if not self.communication_enabled:
            return self.set_training_comm_weights(np.zeros((self.n_agents, self.n_agents)))
        self.training_comm_adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.training_comm_edges = []
        for i, j in edges:
            i, j = int(i), int(j)
            if i == j:
                continue
            if not (0 <= i < self.n_agents and 0 <= j < self.n_agents):
                continue
            self.training_comm_adj[i, j] = 1.0
            self.training_comm_adj[j, i] = 1.0
            edge = (min(i, j), max(i, j))
            if edge not in self.training_comm_edges:
                self.training_comm_edges.append(edge)
        return self.training_comm_adj.copy()

    def get_neighbor_adjacency(self):
        """Available links [receiver, sender]. No team labels are consulted."""
        adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        if not self.communication_enabled or self.communication_topology == "none":
            return adj
        for i, receiver in enumerate(self.agents):
            for j, sender in enumerate(self.agents):
                if i == j:
                    continue
                if self.communication_topology == "complete":
                    visible = True
                else:
                    visible = self._agents_within_comm_range(i, j)
                adj[i, j] = float(visible)
        return adj

    def set_training_comm_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (self.n_agents, self.n_agents) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("communication weights must be finite nonnegative NxN")
        self._training_comm_weights = weights.copy()
        np.fill_diagonal(self._training_comm_weights, 0)
        self.training_comm_directed = not np.allclose(weights, weights.T)
        return self.update_training_comm_graph()

    def _agents_within_comm_range(self, i: int, j: int) -> bool:
        ai = self.agents[int(i)]
        aj = self.agents[int(j)]
        dx = abs(int(ai.x) - int(aj.x))
        dy = abs(int(ai.y) - int(aj.y))
        # Match the square local observation window used by sensor_range.
        return max(dx, dy) <= self.communication_range

    def update_training_comm_graph(self):
        """Build a time-varying graph from inferred team labels and distance.

        Edge i--j exists when:
        1. both agents have valid inferred team ids,
        2. the inferred team ids match,
        3. both predictions exceed the confidence threshold,
        4. optionally, the agents are within communication_range.
        """
        if self._training_comm_weights is not None or not self.communication_enabled:
            weights = self._training_comm_weights
            if weights is None:
                weights = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
            self.training_comm_adj = weights * self.get_neighbor_adjacency()
            # Public edge tuples are (sender, receiver), matching arrow direction.
            self.training_comm_edges = [(int(j), int(i)) for i, j in zip(*np.nonzero(self.training_comm_adj))
                                        if self.training_comm_directed or i < j]
            return self.training_comm_adj.copy()
        self.training_comm_adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.training_comm_edges = []

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                same_team = (
                    self.inferred_team_ids[i] >= 0
                    and self.inferred_team_ids[i] == self.inferred_team_ids[j]
                )
                confident = (
                    self.inferred_team_confidence[i] >= self.team_edge_threshold
                    and self.inferred_team_confidence[j] >= self.team_edge_threshold
                )
                physically_connected = (
                    True if not self.use_physical_comm_range else self._agents_within_comm_range(i, j)
                )

                if same_team and confident and physically_connected:
                    self.training_comm_adj[i, j] = 1.0
                    self.training_comm_adj[j, i] = 1.0
                    self.training_comm_edges.append((i, j))

        return self.training_comm_adj.copy()

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)

        if self.shuffle_team_assignments_on_reset:
            self.agent_team_ids = self.np_random.permutation(self._base_agent_team_ids)
        else:
            self.agent_team_ids = self._base_agent_team_ids.copy()

        self._assign_shelf_teams()
        self._assign_team_goals()
        self._make_team_request_queues()
        self._remember_shelf_home_positions()
        self.team_delivery_counts[:] = 0
        self.team_return_counts[:] = 0
        self._last_delivery_events = []
        self._last_return_events = []
        self._last_wrong_return_events = []
        self._last_pickup_events = []
        self._last_invalid_toggle_events = []
        self._last_failed_forward_events = []
        self._last_blocking_agent_events = []
        self._last_deadlock_events = []
        self._last_reward_shaping[:] = 0.0
        self._last_requested_actions[:] = int(Action.NOOP.value)
        self._consecutive_failed_forward_steps[:] = 0
        self._shelves_awaiting_return.clear()
        self._shelf_return_team_ids.clear()
        for picked_shelf_ids in self._picked_request_shelf_ids_by_agent:
            picked_shelf_ids.clear()
        for agent_id, agent in enumerate(self.agents):
            visited_cells = self._visited_cells_by_agent[agent_id]
            visited_cells.clear()
            visited_cells.add((int(agent.x), int(agent.y)))
        self._stationary_action_streak[:] = 0
        self._reset_progress_baselines()
        self.inferred_team_ids[:] = -1
        self.inferred_team_confidence[:] = 0.0
        self.update_training_comm_graph()

        obs = tuple([self._make_obs(agent) for agent in self.agents])
        return obs, self._get_info()

    def _get_info(self):
        info = {
            "deliveries": len(self._last_delivery_events),
            "returns": len(self._last_return_events),
            "wrong_returns": len(self._last_wrong_return_events),
            "pickups": len(self._last_pickup_events),
            "invalid_toggles": len(self._last_invalid_toggle_events),
            "failed_forwards": len(self._last_failed_forward_events),
            "blocking_agents": len(self._last_blocking_agent_events),
            "deadlocked_agents": len(self._last_deadlock_events),
        }
        if self.reveal_team_info:
            info.update(
                {
                    "agent_team_ids": self.agent_team_ids.copy(),
                    "team_members": self.get_team_members(),
                    "team_delivery_counts": self.team_delivery_counts.copy(),
                    "team_return_counts": self.team_return_counts.copy(),
                    "delivery_events": list(self._last_delivery_events),
                    "return_events": list(self._last_return_events),
                    "wrong_return_events": list(self._last_wrong_return_events),
                    "pickup_events": list(self._last_pickup_events),
                    "invalid_toggle_events": list(self._last_invalid_toggle_events),
                    "failed_forward_events": list(self._last_failed_forward_events),
                    "blocking_agent_events": list(self._last_blocking_agent_events),
                    "deadlock_events": list(self._last_deadlock_events),
                    "shelves_awaiting_return": sorted(self._shelves_awaiting_return),
                    "shelf_home_positions": dict(self._shelf_home_positions),
                    "reward_shaping": self._last_reward_shaping.copy(),
                    "team_request_ids": [
                        [shelf.id for shelf in team_queue]
                        for team_queue in self.team_request_queues
                    ],
                    "shelf_team_mode": self.shelf_team_mode,
                    "shelf_soft_zone_ratio": self.shelf_soft_zone_ratio,
                    "shelf_zone_team_counts": self.get_shelf_zone_team_counts(),
                    "team_goal_positions": [
                        list(team_goals) for team_goals in self.goals_by_team
                    ],
                    "goal_team_ids": dict(self.goal_team_ids),
                    "inferred_team_ids": self.inferred_team_ids.copy(),
                    "inferred_team_confidence": self.inferred_team_confidence.copy(),
                    "training_comm_adj": self.training_comm_adj.copy(),
                    "training_comm_edges": list(self.training_comm_edges),
                    "communication_range": self.communication_range,
                }
            )
        return info

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        assert len(actions) == len(self.agents)
        self._last_delivery_events = []
        self._last_return_events = []
        self._last_wrong_return_events = []
        self._last_pickup_events = []
        self._last_invalid_toggle_events = []
        self._last_failed_forward_events = []
        self._last_blocking_agent_events = []
        self._last_deadlock_events = []

        requested_action_values = []
        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)
            requested_action_values.append(int(agent.req_action.value))
        self._last_requested_actions[:] = np.asarray(
            requested_action_values,
            dtype=np.int32,
        )


        commited_agents = set()
        movement_graph = nx.DiGraph()

        for agent in self.agents:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            if (
                agent.carrying_shelf
                and start != target
                and self.grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    self.grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, target[1], target[0]] - 1
                    ].carrying_shelf
                )
            ):
                agent.req_action = Action.NOOP
                movement_graph.add_edge(start, start)
            else:
                movement_graph.add_edge(start, target)

        weak_components = [
            movement_graph.subgraph(c).copy()
            for c in nx.weakly_connected_components(movement_graph)
        ]

        for component in weak_components:
            try:
                cycle = nx.algorithms.find_cycle(component)
                if len(cycle) == 2:
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(component)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        if self.step_penalty:
            rewards += self.step_penalty
        shaping_rewards = np.zeros(self.n_agents, dtype=np.float32)
        progress_shaping_rewards = np.zeros(self.n_agents, dtype=np.float32)
        requested_actions = [agent.req_action for agent in self.agents]
        before_positions = [(agent.x, agent.y) for agent in self.agents]
        before_position_to_agent = {
            position: agent_id for agent_id, position in enumerate(before_positions)
        }
        before_carried_shelves = [agent.carrying_shelf for agent in self.agents]
        # Freeze candidate positions for this transition. A teammate moving,
        # picking up, or replacing a request must not reward an idle agent.
        before_pickup_targets = [
            [(int(shelf.x), int(shelf.y)) for shelf in self._available_requested_shelves(agent)]
            for agent in self.agents
        ]
        before_requested_shelf_distances = [
            (
                None
                if agent.carrying_shelf
                else self._nearest_requested_shelf_distance(agent)
            )
            for agent in self.agents
        ]
        before_goal_distances = [
            self._carried_requested_shelf_delivery_goal_distance(agent)
            if agent.carrying_shelf
            else None
            for agent in self.agents
        ]
        before_return_distances = [
            self._carried_requested_shelf_return_distance(agent)
            if agent.carrying_shelf
            else None
            for agent in self.agents
        ]

        blocking_agent_counts = np.zeros(self.n_agents, dtype=np.float32)
        failed_agent_indices: Set[int] = set()
        failed_agent_blocked_by_agent = np.zeros(self.n_agents, dtype=bool)
        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent_idx = int(agent.id) - 1
            failed_agent_indices.add(agent_idx)
            target = tuple(map(int, agent.req_location(self.grid_size)))
            self._last_failed_forward_events.append(
                {
                    "agent_id": int(agent.id),
                    "position": (int(agent.x), int(agent.y)),
                    "target": target,
                }
            )
            blocker_id = before_position_to_agent.get(target)
            if blocker_id is not None and blocker_id != agent.id - 1:
                blocker = self.agents[blocker_id]
                blocker_is_moving = (
                    blocker in commited_agents
                    and blocker.req_action == Action.FORWARD
                )
                if not blocker_is_moving:
                    blocking_agent_counts[blocker_id] += 1.0
                    failed_agent_blocked_by_agent[agent_idx] = True
                    self._last_blocking_agent_events.append(
                        {
                            "agent_id": int(blocker.id),
                            "position": target,
                            "blocked_agent_id": int(agent.id),
                        }
                    )
            agent.req_action = Action.NOOP
            if self.failed_forward_penalty:
                rewards[agent.id - 1] -= self.failed_forward_penalty

        for agent_idx in range(self.n_agents):
            if agent_idx in failed_agent_indices:
                self._consecutive_failed_forward_steps[agent_idx] += 1
            else:
                self._consecutive_failed_forward_steps[agent_idx] = 0

        if self.blocking_agent_penalty:
            shaping_rewards -= self.blocking_agent_penalty * blocking_agent_counts
        if self.deadlock_penalty:
            for agent_idx in sorted(failed_agent_indices):
                if not failed_agent_blocked_by_agent[agent_idx]:
                    continue
                blocked_steps = int(self._consecutive_failed_forward_steps[agent_idx])
                if blocked_steps <= self.deadlock_penalty_after:
                    continue
                shaping_rewards[agent_idx] -= self.deadlock_penalty
                event = self._last_failed_forward_events[-1]
                for candidate in self._last_failed_forward_events:
                    if int(candidate["agent_id"]) == agent_idx + 1:
                        event = candidate
                        break
                self._last_deadlock_events.append(
                    {
                        "agent_id": int(agent_idx + 1),
                        "blocked_steps": blocked_steps,
                        "position": event["position"],
                        "target": event["target"],
                    }
                )

        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
                    if agent.carrying_shelf.id in self._shelves_awaiting_return:
                        agent.has_delivered = True
                    shaping_rewards[agent.id - 1] += self._pickup_shaping_reward(
                        agent,
                        agent.carrying_shelf,
                    )
                    self._last_pickup_events.append(
                        {
                            "shelf_id": int(agent.carrying_shelf.id),
                            "agent_id": int(agent.id),
                            "position": (int(agent.x), int(agent.y)),
                            "requested": bool(
                                self._is_agent_requested_shelf(
                                    agent,
                                    agent.carrying_shelf,
                                )
                            ),
                            "return_phase": bool(
                                agent.carrying_shelf.id in self._shelves_awaiting_return
                            ),
                        }
                    )
                else:
                    self._last_invalid_toggle_events.append(
                        {
                            "agent_id": int(agent.id),
                            "position": (int(agent.x), int(agent.y)),
                            "reason": "empty_cell",
                        }
                    )
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                delivered_before_drop = bool(agent.has_delivered)
                dropped_shelf = agent.carrying_shelf
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None
                    if (
                        self.require_delivered_shelf_return
                        and delivered_before_drop
                        and dropped_shelf.id in self._shelves_awaiting_return
                    ):
                        team_id = int(
                            self._shelf_return_team_ids.get(
                                dropped_shelf.id,
                                self.shelf_team_ids.get(dropped_shelf.id, self._agent_team_id(agent)),
                            )
                        )
                        if self._is_shelf_at_home(dropped_shelf):
                            rewarded_agents = self._reward_shelf_return(
                                rewards,
                                team_id,
                                agent.id,
                            )
                            self._shelves_awaiting_return.discard(dropped_shelf.id)
                            self._shelf_return_team_ids.pop(dropped_shelf.id, None)
                            self.team_return_counts[team_id] += 1
                            self._replace_team_request(team_id, dropped_shelf)
                            self._last_return_events.append(
                                {
                                    "shelf_id": dropped_shelf.id,
                                    "team_id": team_id,
                                    "home": self._shelf_home_positions.get(dropped_shelf.id),
                                    "carrier_id": int(agent.id),
                                    "rewarded_agents": rewarded_agents,
                                }
                            )
                        else:
                            shaping_rewards[agent.id - 1] -= self.wrong_shelf_return_penalty
                            self._last_wrong_return_events.append(
                                {
                                    "shelf_id": int(dropped_shelf.id),
                                    "agent_id": int(agent.id),
                                    "position": (int(agent.x), int(agent.y)),
                                    "home": self._shelf_home_positions.get(dropped_shelf.id),
                                }
                            )
                    elif delivered_before_drop:
                        shaping_rewards[agent.id - 1] += self.delivered_shelf_drop_reward
                    else:
                        shaping_rewards[agent.id - 1] -= self.premature_drop_penalty
                    agent.has_delivered = False
                else:
                    self._last_invalid_toggle_events.append(
                        {
                            "agent_id": int(agent.id),
                            "position": (int(agent.x), int(agent.y)),
                            "reason": "highway_drop",
                        }
                    )

        self._recalc_grid()

        for agent_id, agent in enumerate(self.agents):
            carried_before = before_carried_shelves[agent_id]
            if carried_before is None and agent.carrying_shelf is None:
                targets = before_pickup_targets[agent_id]
                after_distance = min(
                    (self._distance(agent.x, agent.y, target) for target in targets),
                    default=None,
                )
                if self.reward_only_new_best_progress:
                    progress_reward = self._new_best_progress_reward(
                        self._best_requested_shelf_distances,
                        agent_id,
                        after_distance,
                        self.requested_shelf_progress_reward,
                    )
                else:
                    progress_reward = self._progress_reward(
                        before_requested_shelf_distances[agent_id],
                        after_distance,
                        self.requested_shelf_progress_reward,
                    )
                if before_positions[agent_id] == (agent.x, agent.y):
                    progress_reward = 0.0
                shaping_rewards[agent_id] += progress_reward
                progress_shaping_rewards[agent_id] += progress_reward
            elif carried_before is not None and agent.carrying_shelf is carried_before:
                shelf_id = int(carried_before.id)
                returning_shelf = carried_before.id in self._shelves_awaiting_return
                if returning_shelf:
                    after_distance = self._carried_requested_shelf_return_distance(agent)
                    best_distances_by_agent = self._best_return_distances_by_agent
                    scalar_best_distances = self._best_return_distances
                    before_distance = before_return_distances[agent_id]
                    progress_coefficient = self.return_progress_reward
                else:
                    after_distance = self._carried_requested_shelf_delivery_goal_distance(agent)
                    best_distances_by_agent = self._best_delivery_distances_by_agent
                    scalar_best_distances = self._best_goal_distances
                    before_distance = before_goal_distances[agent_id]
                    progress_coefficient = self.goal_progress_reward

                if self.reward_only_new_best_progress:
                    progress_reward = self._new_best_shelf_progress_reward(
                        best_distances_by_agent,
                        scalar_best_distances,
                        agent_id,
                        shelf_id,
                        after_distance,
                        progress_coefficient,
                    )
                else:
                    progress_reward = self._progress_reward(
                        before_distance,
                        after_distance,
                        progress_coefficient,
                    )
                shaping_rewards[agent_id] += progress_reward
                progress_shaping_rewards[agent_id] += progress_reward

            moved = before_positions[agent_id] != (agent.x, agent.y)
            carrying_changed = carried_before is not agent.carrying_shelf
            if moved:
                self._stationary_action_streak[agent_id] = 0
                movement_reward = self.forward_movement_reward
                if (
                    self.movement_reward_requires_progress
                    and progress_shaping_rewards[agent_id] <= 0.0
                ):
                    movement_reward = 0.0
                shaping_rewards[agent_id] += movement_reward

                current_cell = (int(agent.x), int(agent.y))
                visited_cells = self._visited_cells_by_agent[agent_id]
                if current_cell in visited_cells:
                    shaping_rewards[agent_id] -= self.revisit_cell_penalty
                else:
                    shaping_rewards[agent_id] += self.new_cell_reward
                    visited_cells.add(current_cell)
            elif carrying_changed:
                self._stationary_action_streak[agent_id] = 0
                if agent.carrying_shelf is None:
                    self._best_requested_shelf_distances[agent_id] = (
                        self._nearest_requested_shelf_distance(agent)
                    )
                    self._best_goal_distances[agent_id] = None
                    self._best_return_distances[agent_id] = None
                else:
                    self._best_requested_shelf_distances[agent_id] = None
                    if agent.carrying_shelf.id in self._shelves_awaiting_return:
                        self._best_goal_distances[agent_id] = None
                        shelf_id = int(agent.carrying_shelf.id)
                        if shelf_id not in self._best_return_distances_by_agent[agent_id]:
                            distance = self._carried_requested_shelf_return_distance(agent)
                            if distance is not None:
                                self._best_return_distances_by_agent[agent_id][
                                    shelf_id
                                ] = int(distance)
                        self._best_return_distances[agent_id] = (
                            self._best_return_distances_by_agent[agent_id].get(shelf_id)
                        )
                    else:
                        shelf_id = int(agent.carrying_shelf.id)
                        if shelf_id not in self._best_delivery_distances_by_agent[agent_id]:
                            distance = (
                                self._carried_requested_shelf_delivery_goal_distance(agent)
                            )
                            if distance is not None:
                                self._best_delivery_distances_by_agent[agent_id][
                                    shelf_id
                                ] = int(distance)
                        self._best_goal_distances[agent_id] = (
                            self._best_delivery_distances_by_agent[agent_id].get(shelf_id)
                        )
                        self._best_return_distances[agent_id] = None
            else:
                action = requested_actions[agent_id]
                action_penalty = 0.0
                if action in [Action.LEFT, Action.RIGHT]:
                    action_penalty += self.turn_action_penalty
                elif action == Action.NOOP:
                    action_penalty += self.noop_action_penalty
                elif action == Action.TOGGLE_LOAD:
                    action_penalty += self.invalid_toggle_load_penalty
                elif action != Action.FORWARD:
                    action_penalty += self.stationary_action_penalty

                if action != Action.FORWARD:
                    action_penalty += self.stationary_action_penalty

                self._stationary_action_streak[agent_id] += 1
                if (
                    self._stationary_action_streak[agent_id]
                    > self.stationary_streak_penalty_after
                ):
                    action_penalty += self.repeated_stationary_action_penalty
                shaping_rewards[agent_id] -= action_penalty
        rewards += shaping_rewards
        self._last_reward_shaping = shaping_rewards.copy()

        shelf_delivered = False
        for goal_x, goal_y in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, goal_y, goal_x]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            if shelf.id in self._shelves_awaiting_return:
                continue

            team_id = self.shelf_team_ids[shelf.id]
            goal = (int(goal_x), int(goal_y))
            goal_team_id = self.goal_team_ids.get(goal)

            if self.require_matching_team_goal and goal_team_id != team_id:
                continue

            carrier_id = self.grid[_LAYER_AGENTS, goal_y, goal_x]
            if carrier_id > 0 and self.agents[carrier_id - 1].has_delivered:
                continue

            shelf_delivered = True
            rewarded_agents = self._reward_delivery(rewards, team_id, carrier_id)
            if (
                carrier_id > 0
                and self.agent_team_ids[carrier_id - 1] == team_id
            ):
                self.agents[carrier_id - 1].has_delivered = True
            self.team_delivery_counts[team_id] += 1
            if self.require_delivered_shelf_return:
                self._shelves_awaiting_return.add(shelf.id)
                self._shelf_return_team_ids[shelf.id] = int(team_id)
                if carrier_id > 0:
                    carrier_idx = int(carrier_id) - 1
                    self._best_goal_distances[carrier_idx] = None
                    self._best_return_distances[carrier_idx] = (
                        self._carried_requested_shelf_return_distance(
                            self.agents[carrier_idx]
                        )
                    )
                    if self._best_return_distances[carrier_idx] is not None:
                        self._best_return_distances_by_agent[carrier_idx][
                            int(shelf.id)
                        ] = int(self._best_return_distances[carrier_idx])
            else:
                self._replace_team_request(team_id, shelf)

            event = {
                "shelf_id": shelf.id,
                "team_id": team_id,
                "goal": goal,
                "goal_team_id": int(goal_team_id) if goal_team_id is not None else None,
                "carrier_id": int(carrier_id),
                "rewarded_agents": rewarded_agents,
                "requires_return": bool(self.require_delivered_shelf_return),
                "home": self._shelf_home_positions.get(shelf.id),
            }
            if carrier_id > 0:
                event["carrier_team_id"] = int(self.agent_team_ids[carrier_id - 1])
            self._last_delivery_events.append(event)

        shelf_returned = bool(self._last_return_events)
        if shelf_delivered or shelf_returned:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        done = bool(
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        )
        truncated = bool(self.max_steps and self._cur_steps >= self.max_steps and not done)

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        self.update_training_comm_graph()
        info = self._get_info()
        return new_obs, list(rewards), done, truncated, info
