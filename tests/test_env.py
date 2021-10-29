import os
import sys
import pytest
import gym
import numpy as np
from gym import spaces

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.warehouse import ObserationType, Warehouse, Direction, Action, RewardType


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_0():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, 10, None, RewardType.GLOBAL)
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
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
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
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
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
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.Discrete(len(Action)), ))
    env.step(env.action_space.sample())


def test_action_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=1,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.MultiDiscrete([len(Action), 2]), ))
    env.step(env.action_space.sample())


def test_action_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=2,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.MultiDiscrete([len(Action), 2, 2]), ))
    env.step(env.action_space.sample())


def test_action_space_3():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(10 * (spaces.MultiDiscrete([len(Action), *5 * (2,)]), ))
    env.step(env.action_space.sample())


def test_obs_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        observation_type=ObserationType.DICT,
    )
    obs = env.reset()
    assert env.observation_space[0]["self"].contains(obs[0]["self"])
    assert env.observation_space[0].contains(obs[0])
    assert env.observation_space.contains(obs)
    nobs, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(nobs)


def test_obs_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    obs = env.reset()
    for _ in range(200):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_obs_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    obs = env.reset()
    for s, o in zip(env.observation_space, obs):
        assert len(gym.spaces.flatten(s, o)) == env._obs_length


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


@pytest.mark.parametrize("time_limit,", [1, 100, 200])
def test_time_limit(time_limit):
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=time_limit,
        reward_type=RewardType.GLOBAL,
    )
    _ = env.reset()

    for _ in range(time_limit - 1):
        _, _, done, _ = env.step(env.action_space.sample())
        assert done == 10 * [False]

    _, _, done, _ = env.step(env.action_space.sample())
    assert done == 10 * [True]


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


def test_fast_obs_0():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, 10, None, RewardType.GLOBAL, observation_type=ObserationType.DICT)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 2
        assert len(slow_obs) == 2

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        for i in range(len(fast_obs)):
            print(slow_obs[0])
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())
        
def test_fast_obs_1():
    env = Warehouse(3, 8, 3, 3, 0, 1, 5, 10, None, RewardType.GLOBAL, observation_type=ObserationType.DICT)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        for i in range(len(fast_obs)):
            print(slow_obs[0])
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())
        
def test_fast_obs_2():
    env = Warehouse(3, 8, 3, 3, 2, 1, 5, 10, None, RewardType.GLOBAL, observation_type=ObserationType.DICT)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        for i in range(len(fast_obs)):
            print(slow_obs[0])
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())
        
