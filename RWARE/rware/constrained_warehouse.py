"""Opt-in RWARE constraints. Importing this module does not register or alter old envs."""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import networkx as nx
import numpy as np

from rware.multi_team_warehouse import MultiTeamWarehouse
from rware.utils.semantic_observation import RwareSemanticObservationWrapper
from rware.warehouse import Action, Direction

ENV_ID = "rware-constrained-4ag-2teams-v0"
PROFILES = ("control", "detour", "bottleneck", "dock")


def profile_kwargs(profile: str) -> dict:
    """Same 16x16 map and shelves; progressively add a baffle and dock cooldown."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown constrained profile: {profile}")
    walls = ([] if profile == 'control' else [[3,14],[12,14]] if profile == 'detour'
             else [[x, 7] for x in range(16) if x not in (3, 12)])
    return dict(shelf_rows=2, shelf_columns=5, column_height=6,
                wall_cells=walls,
                dock_cooldown_steps=8 if profile == "dock" else 0)


def register_constrained_environment() -> None:
    """Register only the new ID, only in the calling process."""
    if ENV_ID in gym.registry:
        if gym.spec(ENV_ID).entry_point != "rware.constrained_warehouse:ConstrainedWarehouse":
            raise ValueError(f"Environment ID already registered by another implementation: {ENV_ID}")
        return
    gym.register(ENV_ID, entry_point="rware.constrained_warehouse:ConstrainedWarehouse",
                 kwargs=dict(shelf_rows=2, shelf_columns=5, column_height=6,
                             n_agents=4, n_teams=2, msg_bits=0, sensor_range=5,
                             request_queue_size=4, request_queue_size_per_team=2,
                             max_steps=500, max_inactivity_steps=None))


class ConstrainedWarehouse(MultiTeamWarehouse):
    """Static walls and a per-goal delivery cooldown without changing old RWARE.

    Walls occupy existing highway cells, preserving shelf capacity. A blocked
    FORWARD executes NOOP through the inherited simultaneous collision resolver.
    With action masks disabled this incurs the inherited NOOP costs plus the
    explicit constraint_block_penalty. Empty robots can pass a busy dock;
    loaded robots must wait before entering it. The carrier can always leave.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, *args, wall_cells: Sequence[Sequence[int]] = (),
                 dock_cooldown_steps: int = 0, constraint_block_penalty: float = .005,
                 **kwargs):
        if int(dock_cooldown_steps) != dock_cooldown_steps or dock_cooldown_steps < 0:
            raise ValueError("dock_cooldown_steps must be a nonnegative integer")
        if not np.isfinite(constraint_block_penalty) or constraint_block_penalty < 0:
            raise ValueError("constraint_block_penalty must be finite and nonnegative")
        if kwargs.get("msg_bits", 0) != 0:
            raise ValueError("ConstrainedWarehouse currently requires msg_bits=0")
        kwargs.setdefault("render_mode", "rgb_array")
        if kwargs["render_mode"] != "rgb_array":
            raise ValueError("Use render_mode='rgb_array'; previews and videos require no GUI")
        self.wall_cells = frozenset(tuple(map(int, cell)) for cell in wall_cells)
        self.barrier_rows = {cell[1] for cell in self.wall_cells if len(cell) == 2}
        self.dock_cooldown_steps = int(dock_cooldown_steps)
        self.constraint_block_penalty = float(constraint_block_penalty)
        self.dock_timers = {}
        self.last_constraint_blocks = []
        # Lifetime totals, not episode totals, so rollout deltas cross resets safely.
        self.constraint_totals = dict(wall_blocks=0, dock_blocks=0, dock_busy_steps=0,
                                      dock_queue_agent_steps=0, barrier_region_agent_steps=0)
        super().__init__(*args, **kwargs)
        height, width = self.grid_size
        for cell in self.wall_cells:
            if len(cell) != 2:
                raise ValueError("wall_cells must contain (x,y) pairs")
            x, y = cell
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Wall outside the map: {cell}")
            if not self._is_highway(x, y) or cell in self.goals:
                raise ValueError("Walls must occupy existing highway cells, excluding goals")
        self.floor_graph = nx.Graph()
        for y in range(height):
            for x in range(width):
                if self._is_highway(x, y) and (x, y) not in self.wall_cells:
                    self.floor_graph.add_node((x, y))
        for x, y in list(self.floor_graph):
            for neighbor in ((x+1, y), (x, y+1)):
                if neighbor in self.floor_graph:
                    self.floor_graph.add_edge((x, y), neighbor)
        if not self.floor_graph or not nx.is_connected(self.floor_graph):
            raise ValueError("Walls disconnect the loaded-robot highway graph")

    @staticmethod
    def forward_target(agent, grid_size):
        dx, dy = {Direction.UP: (0, -1), Direction.DOWN: (0, 1),
                  Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0)}[agent.dir]
        return (int(np.clip(agent.x + dx, 0, grid_size[1]-1)),
                int(np.clip(agent.y + dy, 0, grid_size[0]-1)))

    def forward_constraint(self, agent):
        target = self.forward_target(agent, self.grid_size)
        if target == (agent.x, agent.y):
            return None  # Existing boundary masks retain their original meaning.
        if target in self.wall_cells:
            return "wall"
        if agent.carrying_shelf is not None and self.dock_timers.get(target, 0) > 0:
            return "dock"
        return None

    def reset(self, seed=None, options=None):
        self.dock_timers = {}
        self.last_constraint_blocks = []
        super().reset(seed=seed, options=options)
        # Original spawn draws are retained; only invalid wall spawns are relocated.
        occupied = {(a.x, a.y) for a in self.agents if (a.x, a.y) not in self.wall_cells}
        for agent in self.agents:
            if (agent.x, agent.y) in self.wall_cells:
                candidates = [(x, y) for y in range(self.grid_size[0]) for x in range(self.grid_size[1])
                              if (x, y) not in self.wall_cells and (x, y) not in occupied]
                agent.x, agent.y = candidates[int(self.np_random.integers(len(candidates)))]
                occupied.add((agent.x, agent.y))
        self._recalc_grid()
        for i, agent in enumerate(self.agents):
            self._visited_cells_by_agent[i] = {(int(agent.x), int(agent.y))}
        self._reset_progress_baselines()
        self.update_training_comm_graph()
        self.dock_timers = {tuple(goal): 0 for goal in self.goals}
        # Validate all shelf homes, including ones not currently requested.
        if any(tuple(goal) not in self.floor_graph for goal in self.goals):
            raise ValueError("A delivery goal is blocked")
        for shelf in self.shelfs:
            if not any(cell in self.floor_graph for cell in
                       ((shelf.x-1, shelf.y), (shelf.x+1, shelf.y),
                        (shelf.x, shelf.y-1), (shelf.x, shelf.y+1))):
                raise ValueError(f"Shelf home cannot reach the loaded highway graph: {shelf.id}")
        return tuple(self._make_obs(a) for a in self.agents), self._get_info()

    def step(self, actions):
        if len(actions) != len(self.agents):
            raise ValueError("Provide one action per agent")
        requested = [int(a) for a in actions]
        executed = requested.copy()
        blocks = []
        streaks = self._consecutive_failed_forward_steps.copy()
        self.constraint_totals["dock_busy_steps"] += sum(t > 0 for t in self.dock_timers.values())
        for i, agent in enumerate(self.agents):
            self.constraint_totals["barrier_region_agent_steps"] += int(any(abs(agent.y-y)<=1 for y in self.barrier_rows))
            if agent.carrying_shelf is not None:
                self.constraint_totals["dock_queue_agent_steps"] += int(any(
                    t > 0 and abs(agent.x-x)+abs(agent.y-y) <= 2
                    for (x, y), t in self.dock_timers.items()))
            reason = self.forward_constraint(agent)
            if requested[i] == Action.FORWARD.value and reason:
                executed[i] = Action.NOOP.value
                blocks.append(dict(agent_id=i+1, position=(int(agent.x), int(agent.y)),
                                   target=self.forward_target(agent, self.grid_size), constraint=reason))
        _, rewards, terminated, truncated, _ = super().step(executed)
        self.dock_timers = {cell: max(0, t-1) for cell, t in self.dock_timers.items()}
        for event in self._last_delivery_events:
            self.dock_timers[tuple(event['goal'])] = self.dock_cooldown_steps
        self.last_constraint_blocks = blocks
        self._last_requested_actions[:] = requested
        for event in blocks:
            i = event['agent_id']-1
            rewards[i] -= self.constraint_block_penalty
            self._last_reward_shaping[i] -= self.constraint_block_penalty
            self._last_failed_forward_events.append(event.copy())
            self._consecutive_failed_forward_steps[i] = streaks[i]+1
            self.constraint_totals[event['constraint']+'_blocks'] += 1
        assert all((a.x, a.y) not in self.wall_cells for a in self.agents)
        return tuple(self._make_obs(a) for a in self.agents), rewards, terminated, truncated, self._get_info()

    def _get_info(self):
        info = super()._get_info()
        info.update(constraint_blocks=list(self.last_constraint_blocks),
                    constraint_totals=dict(self.constraint_totals),
                    dock_timers=[dict(cell=list(cell), remaining=t) for cell, t in self.dock_timers.items()],
                    wall_cells=[list(cell) for cell in sorted(self.wall_cells)])
        return info

    def routing_report(self):
        distances = [nx.multi_source_dijkstra_path_length(self.floor_graph, list(goals), weight=None)
                     for goals in self.goals_by_team]
        routes = []
        for shelf in self.shelfs:
            team = self.shelf_team_ids[shelf.id]
            neighbors = [(shelf.x-1, shelf.y), (shelf.x+1, shelf.y),
                         (shelf.x, shelf.y-1), (shelf.x, shelf.y+1)]
            loaded = 1+min(distances[team][cell] for cell in neighbors if cell in distances[team])
            manhattan = min(abs(shelf.x-x)+abs(shelf.y-y) for x, y in self.goals_by_team[team])
            routes.append(dict(shelf_id=int(shelf.id), home=[int(shelf.x), int(shelf.y)], team=int(team),
                               loaded_distance=int(loaded), manhattan_distance=int(manhattan)))
        return dict(shape=list(map(int,self.grid_size)), shelves=len(self.shelfs), walls=len(self.wall_cells),
                    goals=[list(map(int,g)) for g in self.goals], loaded_floor_connected=True,
                    mean_loaded_distance=float(np.mean([r['loaded_distance'] for r in routes])),
                    max_loaded_distance=max(r['loaded_distance'] for r in routes), routes=routes)

    def render(self):
        # CPU-only renderer; no OpenGL window or change to the original viewer.
        from PIL import Image, ImageDraw
        scale, header = 28, 32
        h, w = self.grid_size
        img = Image.new('RGB', (w*scale, h*scale+header), '#f2f2f2')
        draw = ImageDraw.Draw(img)
        draw.text((6, 8), f'RWARE constraints | walls={len(self.wall_cells)} | dock={self.dock_cooldown_steps}', fill='black')
        colors = ['#377eb8', '#e66101', '#984ea3', '#4daf4a']
        for y in range(h):
            for x in range(w):
                box = (x*scale, y*scale+header, (x+1)*scale-1, (y+1)*scale+header-1)
                fill = '#252525' if (x,y) in self.wall_cells else ('#ffffff' if self._is_highway(x,y) else '#dddddd')
                draw.rectangle(box, fill=fill, outline='#bbbbbb')
        for shelf in self.shelfs:
            x,y=shelf.x*scale,shelf.y*scale+header
            requested=shelf in self.request_queue
            draw.rectangle((x+5,y+5,x+scale-5,y+scale-5),
                           fill=colors[self.shelf_team_ids[shelf.id] % len(colors)] if requested else '#999999',
                           outline='black' if requested else '#777777')
        for goal in self.goals:
            x,y=goal[0]*scale,goal[1]*scale+header
            draw.rectangle((x+1,y+1,x+scale-2,y+scale-2), outline=colors[self.goal_team_ids[goal]], width=3)
            if self.dock_timers.get(goal,0):
                draw.text((x+8,y+8),str(self.dock_timers[goal]),fill='#cc0000')
        for i,j in self.training_comm_edges:
            a,b=self.agents[i],self.agents[j]
            draw.line((a.x*scale+scale//2,a.y*scale+header+scale//2,
                       b.x*scale+scale//2,b.y*scale+header+scale//2),fill='#008833',width=3)
        for i,a in enumerate(self.agents):
            x,y=a.x*scale,a.y*scale+header
            draw.ellipse((x+3,y+3,x+scale-3,y+scale-3),fill=colors[self.agent_team_ids[i]],outline='black',width=2)
            dx,dy={Direction.UP:(0,-8),Direction.DOWN:(0,8),Direction.LEFT:(-8,0),Direction.RIGHT:(8,0)}[a.dir]
            draw.line((x+14,y+14,x+14+dx,y+14+dy),fill='black',width=2)
            draw.text((x+9,y+7),str(i+1),fill='white')
        return np.asarray(img)


class ConstrainedSemanticObservation(RwareSemanticObservationWrapper):
    """Keep 1593-D checkpoint compatibility and expose the new constraints locally.

    Channel 0 now represents outside-map OR a solid internal wall. Wall cells
    have no other content. A busy goal retains its goal marker (channel 11),
    but loses the available-relevant-goal marker (channel 12) until reopening.
    No remote wall, timer, agent reward or team ID is added to observations.
    """
    def _semantic_obs(self):
        observations = super()._semantic_obs()
        base = self.unwrapped
        spec = self.semantic_observation_spec
        radius = base.sensor_range
        for agent, obs in zip(base.agents, observations):
            spatial = obs[spec.ego_dim:].reshape(spec.spatial_channels, spec.spatial_size, spec.spatial_size)
            for x,y in base.wall_cells:
                dx,dy=x-agent.x,y-agent.y
                if abs(dx)<=radius and abs(dy)<=radius:
                    spatial[:,dy+radius,dx+radius]=0
                    spatial[0,dy+radius,dx+radius]=1
            for (x,y),remaining in base.dock_timers.items():
                dx,dy=x-agent.x,y-agent.y
                if remaining and abs(dx)<=radius and abs(dy)<=radius:
                    spatial[12,dy+radius,dx+radius]=0
        return observations
