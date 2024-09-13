from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
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
        self.canceled_action = None
        self.has_delivered = False

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


class Warehouse(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

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
        reward_type: RewardType,
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
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.image_dict_obs = None
        if observation_type == ObservationType.IMAGE:
            self.observation_space = self._use_image_obs(
                image_observation_layers, image_observation_directional
            )
        elif observation_type == ObservationType.IMAGE_DICT:
            self.observation_space = self._use_image_dict_obs(
                image_observation_layers, image_observation_directional
            )

        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self.observation_space = self._use_slow_obs()

            # for performance reasons we
            # can flatten the obs vector
            if observation_type == ObservationType.FLATTENED:
                self.observation_space = self._use_fast_obs()

        self.global_image = None
        self.renderer = None
        self.render_mode = render_mode

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.uint8)

        def highway_func(x, y):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (column_height + 1) == 0
            is_on_delivery_row = y == self.grid_size[0] - 1
            is_on_queue = (y > self.grid_size[0] - (column_height + 3)) and (
                x == self.grid_size[1] // 2 - 1 or x == self.grid_size[1] // 2
            )
            return (
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )

        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = int(highway_func(x, y))

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_dict_obs = True
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max(
                    [d.value + 1 for d in Direction]
                )
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        return gym.spaces.Tuple(
            tuple([gym.spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    def _use_image_dict_obs(self, image_observation_layers, directional=True):
        """
        Get image dictionary observation with image and flattened feature vector
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        image_obs_space = self._use_image_obs(image_observation_layers, directional)[0]
        self.image_obs = False
        self.image_dict_obs = True
        feature_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                }
            )
        )

        feature_flat_dim = gym.spaces.flatdim(feature_space)
        feature_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        {"image": image_obs_space, "features": feature_space}
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        max_grid_val = max(self.grid_size)
        low = np.zeros(2)
        if self.normalised_coordinates:
            high = np.ones(2)
            dtype = np.float32
        else:
            high = np.ones(2) * max_grid_val
            dtype = np.int32
        location_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
        )

        self_observation_dict_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "location": location_space,
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                }
            )
        )
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = gym.spaces.MultiBinary(
                self.msg_bits
            )
        sensor_per_location_dict.update(
            {
                "has_shelf": gym.spaces.MultiBinary(1),
                "shelf_requested": gym.spaces.MultiBinary(1),
            }
        )
        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        OrderedDict(
                            {
                                "self": self_observation_dict_space,
                                "sensors": gym.spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (gym.spaces.Dict(sensor_per_location_dict),)
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return self.observation_space

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        return gym.spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_img_obs(self, agent):
        # write image observations
        if agent.id == 1:
            layers = []
            # first agent's observation --> update global observation layers
            for layer_type in self.image_observation_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                    # print("SHELVES LAYER")
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                    # print("REQUESTS LAYER")
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                    # print("AGENTS LAYER")
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.x, ag.y] = float(agent_direction)
                    # print("AGENT DIRECTIONS LAYER")
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.x, ag.y] = 1.0
                    # print("AGENT LOAD LAYER")
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_x, goal_y] = 1.0
                    # print("GOALS LAYER")
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")

                # pad with 0s for out-of-map cells
                layer = np.pad(layer, self.sensor_range, mode="constant")
                layers.append(layer)
            self.global_layers = np.stack(layers)

        # global information was generated --> get information for agent
        start_x = agent.y
        end_x = agent.y + 2 * self.sensor_range + 1
        start_y = agent.x
        end_y = agent.x + 2 * self.sensor_range + 1
        obs = self.global_layers[:, start_x:end_x, start_y:end_y]

        if self.image_observation_directional:
            # rotate image to be in direction of agent
            if agent.dir == Direction.DOWN:
                # rotate by 180 degrees (clockwise)
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif agent.dir == Direction.LEFT:
                # rotate by 90 degrees (clockwise)
                obs = np.rot90(obs, k=3, axes=(1, 2))
            elif agent.dir == Direction.RIGHT:
                # rotate by 270 degrees (clockwise)
                obs = np.rot90(obs, k=1, axes=(1, 2))
            # no rotation needed for UP direction
        return obs

    def _get_default_obs(self, agent):
        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
            obs = _VectorWriter(flatdim)

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])

            # 'has_agent': MultiBinary(1),
            # 'direction': Discrete(4),
            # 'local_message': MultiBinary(2)
            # 'has_shelf': MultiBinary(1),
            # 'shelf_requested': MultiBinary(1),

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    # no agent, direction, or message
                    obs.write([0.0])  # no agent present
                    obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                    obs.skip(self.msg_bits)  # agent message
                else:
                    obs.write([1.0])  # agent present
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)  # agent direction as onehot
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)  # agent message
                if id_shelf == 0:
                    obs.write([0.0, 0.0])  # no shelf or requested shelf
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )  # shelf presence and request status
            return obs.vector

        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y], dtype=np.int32),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = (
                    self.msg_bits * [0] if self.msg_bits > 0 else None
                )
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = (
                    self.agents[id_ - 1].message if self.msg_bits > 0 else None
                )

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs

    def _make_obs(self, agent):
        if self.image_obs:
            return self._make_img_obs(agent)
        elif self.image_dict_obs:
            image_obs = self._make_img_obs(agent)
            feature_obs = _VectorWriter(
                self.observation_space[agent.id - 1]["features"].shape[0]
            )
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            feature_obs.write(direction)
            feature_obs.write(
                [
                    int(self._is_highway(agent.x, agent.y)),
                    int(agent.carrying_shelf is not None),
                ]
            )
            return {
                "image": image_obs,
                "features": feature_obs.vector,
            }
        else:
            return self._get_default_obs(agent)

    def _get_info(self):
        return {}

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

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
        agent_locs = self.np_random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]

        self._recalc_grid()

        self.request_queue = list(
            self.np_random.choice(
                self.shelfs, size=self.request_queue_size, replace=False
            )
        )

        return tuple([self._make_obs(agent) for agent in self.agents]), self._get_info()

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        assert len(actions) == len(self.agents)

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        commited_agents = set()

        G = nx.DiGraph()

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
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

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

        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)

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
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5

                    agent.has_delivered = False

        self._recalc_grid()

        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            # remove from queue and replace it
            candidates = [s for s in self.shelfs if s not in self.request_queue]
            new_request = self.np_random.choice(candidates)
            self.request_queue[self.request_queue.index(shelf)] = new_request
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            done = True
        else:
            done = False
        truncated = False

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = self._get_info()
        return new_obs, list(rewards), done, truncated, info

    def render(self):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)

        return self.renderer.render(
            self, return_rgb_array=self.render_mode == "rgb_array"
        )

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def get_global_image(
        self,
        image_layers=[
            ImageLayer.SHELVES,
            ImageLayer.GOALS,
        ],
        recompute=False,
        pad_to_shape=None,
    ):
        """
        Get global image observation
        :param image_layers: image layers to include in global image
        :param recompute: bool whether image should be recomputed or taken from last computation
            (for default params, image will be constant for environment so no recomputation needed
             but if agent or request information is included, then should be recomputed)
         :param pad_to_shape: if given than pad environment global image shape into this
             shape (if doesn't fit throw exception)
        """
        if recompute or self.global_image is None:
            layers = []
            for layer_type in image_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.x, ag.y] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.x, ag.y] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_x, goal_y] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layers.append(layer)
            self.global_image = np.stack(layers)
            if pad_to_shape is not None:
                padding_dims = [
                    pad_dim - global_dim
                    for pad_dim, global_dim in zip(
                        pad_to_shape, self.global_image.shape
                    )
                ]
                assert all([dim >= 0 for dim in padding_dims])
                pad_before = [pad_dim // 2 for pad_dim in padding_dims]
                pad_after = [
                    pad_dim // 2 if pad_dim % 2 == 0 else pad_dim // 2 + 1
                    for pad_dim in padding_dims
                ]
                self.global_image = np.pad(
                    self.global_image,
                    pad_width=tuple(zip(pad_before, pad_after)),
                    mode="constant",
                    constant_values=0,
                )
        return self.global_image


if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    from tqdm import tqdm

    # env.render()

    for _ in tqdm(range(1000000)):
        # time.sleep(0.05)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
