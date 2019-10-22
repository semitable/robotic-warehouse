import logging

from collections import defaultdict
import gym
from gym import spaces

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

from enum import Enum
import numpy as np

from typing import List, Tuple, Optional

import networkx as nx

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOAD = 4
    UNLOAD = 5


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Warehouse(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, grid_size, n_agents, msg_bits):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.msg_bits = msg_bits

        self._max_steps = None

        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(len(Action)) for _ in range(self.n_agents)]
        )

        self.sensor_range = 1

        self.request_queue_size = 5
        self.request_queue = []

        self.agents: List[Agent] = []
        # self.observation_space = MultiAgentObservationSpace(
        #     [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)]
        # )

    def _is_highway(self, x: int, y: int) -> bool:
        return (
            (x % 3 == 0)  # vertical highways
            or (y % 9 == 0)  # horizontal highways
            or (y == self.grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > self.grid_size[0] - 11)
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
        )

    def _make_obs(self, agent):

        _bits_per_agent = len(Direction) + self.msg_bits
        _bits_per_shelf = 1
        _bits_for_self = 2
        _bits_for_requests = 2

        _sensor_locations = (1 + 2 * self.sensor_range) ** 2
        _request_count = self.request_queue_size

        obs_length = (
            _bits_for_self
            + _bits_for_requests * _request_count
            + (_sensor_locations - 1) * (_bits_per_agent)
            + _sensor_locations * _bits_per_shelf
        )

        obs = _VectorWriter(obs_length)

        obs.write(np.array([agent.x, agent.y]))

        # neighbors
        padded_agents = np.pad(
            self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
        )
        padded_shelfs = np.pad(
            self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
        )

        # + self.sensor_range due to padding
        min_x = agent.x - self.sensor_range + self.sensor_range
        max_x = agent.x + 2 * self.sensor_range + 1

        min_y = agent.y - self.sensor_range + self.sensor_range
        max_y = agent.y + 2 * self.sensor_range + 1

        # ---
        # find neighboring agents

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs.skip(_bits_per_agent)
                continue
            if id_ == agent.id:
                continue
            obs.write(np.eye(len(Direction))[self.agents[id_ - 1].dir])
            obs.write(self.agents[id_ - 1].message)

        # find neighboring shelfs:
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs.skip(_bits_per_shelf)
                continue
            obs.write(np.array([self.shelfs[id_ - 1] in self.request_queue]))

        # writing requests:
        for shelf in self.request_queue:
            obs.write(np.array([shelf.x, shelf.y]))

        assert obs.idx == obs_length
        return obs.vector

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]

        # spawn agents at random locations
        agent_locs = np.random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = np.random.choice(4, size=self.n_agents)

        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]

        self._recalc_grid()

        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )

        return [self._make_obs(agent) for agent in self.agents]
        # for s in self.shelfs:
        #     self.grid[0, s.y, s.x] = 1
        # print(self.grid[0])

    def step(self, actions):
        assert len(actions) == len(self.agents)

        for agent, action in zip(self.agents, actions):
            agent.req_action = action

        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        commited_agents = set()

        G = nx.DiGraph()

        for agent in self.agents:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            G.add_edge(start, target)
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
                # there's a standing shelf at the target location
                # so we add a 'fake' node
                G.add_edge(target, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents
        print(commited_agents)
        print(failed_agents)
        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.LOAD:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
            elif agent.req_action == Action.UNLOAD:
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None

        self._recalc_grid()

    def render(self, mode="human"):
        ...

    def close(self):
        ...

    def seed(self, seed=None):
        ...


if __name__ == "__main__":
    env = Warehouse((29, 10), 20, 2)
    env.reset()
    env.step(18 * [Action.FORWARD] + 2 * [Action.NOOP])
