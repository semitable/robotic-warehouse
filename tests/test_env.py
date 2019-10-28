import os
import sys
import pytest
import gym

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse import Warehouse, Direction, Action, RewardType


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_0():
    env = Warehouse(3, 8, 3, 1, 0, 1, 10, RewardType.GLOBAL)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelfs[0].x = 4
    env.shelfs[0].y = 27

    env.agents[0].carrying_shelf = env.shelfs[0]

    env.request_queue[0] = env.shelfs[0]
    env._recalc_grid()
    return env


def test_grid_size():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (14, 4)
    env = Warehouse(
        shelf_columns=3,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (14, 10)


def test_action_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == 2 * [gym.spaces.Discrete(len(Action))]
    env.step(env.action_space.sample())


def test_action_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=1,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == 2 * [gym.spaces.MultiDiscrete([len(Action), 2])]
    env.step(env.action_space.sample())


def test_action_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=2,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == 2 * [gym.spaces.MultiDiscrete([len(Action), 2, 2])]
    env.step(env.action_space.sample())


def test_action_space_3():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == 10 * [gym.spaces.MultiDiscrete([len(Action), *5 * (2,)])]
    env.step(env.action_space.sample())


def test_obs_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        max_inactivity=None,
        reward_type=RewardType.GLOBAL,
    )
    obs = env.reset()
    assert env.observation_space.contains(obs)
    nobs, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(nobs)


def test_inactivity_0(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]


def test_inactivity_1(env_0):
    env = env_0
    for i in range(4):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]

    _, reward, _, _, = env.step([Action.FORWARD])
    assert reward[0] == pytest.approx(1.0)
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]

    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]


def test_inactivity_2(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]
    env.reset()
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]
