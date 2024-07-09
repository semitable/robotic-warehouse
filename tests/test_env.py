import os
import sys

import gymnasium as gym
import numpy as np
import pytest

from rware.warehouse import ObservationType, Warehouse, Direction, Action, RewardType


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)


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
    assert env.action_space == gym.spaces.Tuple(2 * (gym.spaces.Discrete(len(Action)),))
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
    assert env.action_space == gym.spaces.Tuple(
        2 * (gym.spaces.MultiDiscrete([len(Action), 2]),)
    )
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
    assert env.action_space == gym.spaces.Tuple(
        2 * (gym.spaces.MultiDiscrete([len(Action), 2, 2]),)
    )
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
    assert env.action_space == gym.spaces.Tuple(
        10 * (gym.spaces.MultiDiscrete([len(Action), *5 * (2,)]),)
    )
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
        observation_type=ObservationType.DICT,
    )
    obs, _ = env.reset()
    for i in range(env.unwrapped.n_agents):
        for key in env.observation_space[i]["self"].keys():
            if key == "direction":
                # direction is not considered 'contained' by gym if onehot
                continue
            else:
                print(
                    i,
                    "self",
                    key,
                    env.observation_space[i]["self"][key],
                    obs[i]["self"][key],
                )
                assert env.observation_space[
                    i
                ][
                    "self"
                ][
                    key
                ].contains(
                    obs[i]["self"][key]
                ), f"{obs[i]['self'][key]} is not contained in {env.observation_space[i]['self'][key]}"
        for j in range(len(env.observation_space[i]["sensors"])):
            for key in env.observation_space[i]["sensors"][j].keys():
                if key == "direction":
                    # direction is not considered 'contained' by gym if onehot
                    continue
                else:
                    print(
                        i,
                        "sensors",
                        key,
                        env.observation_space[i]["sensors"][j][key],
                        obs[i]["sensors"][j][key],
                    )
                    assert env.observation_space[
                        i
                    ][
                        "sensors"
                    ][
                        j
                    ][
                        key
                    ].contains(
                        obs[i]["sensors"][j][key]
                    ), f"{obs[i]['sensors'][j][key]} is not contained in {env.observation_space[i]['sensors'][j][key]}"
    obs, _, _, _, _ = env.step(env.action_space.sample())
    for i in range(env.unwrapped.n_agents):
        for key in env.observation_space[i]["self"].keys():
            if key == "direction":
                # direction is not considered 'contained' by gym if onehot
                continue
            else:
                print(
                    i,
                    "self",
                    key,
                    env.observation_space[i]["self"][key],
                    obs[i]["self"][key],
                )
                assert env.observation_space[
                    i
                ][
                    "self"
                ][
                    key
                ].contains(
                    obs[i]["self"][key]
                ), f"{obs[i]['self'][key]} is not contained in {env.observation_space[i]['self'][key]}"
        for j in range(len(env.observation_space[i]["sensors"])):
            for key in env.observation_space[i]["sensors"][j].keys():
                if key == "direction":
                    # direction is not considered 'contained' by gym if onehot
                    continue
                else:
                    print(
                        i,
                        "sensors",
                        key,
                        env.observation_space[i]["sensors"][j][key],
                        obs[i]["sensors"][j][key],
                    )
                    assert env.observation_space[
                        i
                    ][
                        "sensors"
                    ][
                        j
                    ][
                        key
                    ].contains(
                        obs[i]["sensors"][j][key]
                    ), f"{obs[i]['sensors'][j][key]} is not contained in {env.observation_space[i]['sensors'][j][key]}"


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
    obs, _ = env.reset()
    for _ in range(200):
        obs, _, _, _, _ = env.step(env.action_space.sample())
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
    obs, _ = env.reset()
    for s, o in zip(env.observation_space, obs):
        assert len(gym.spaces.flatten(s, o)) == env._obs_length


def test_obs_space_3():
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
        observation_type=ObservationType.IMAGE,
        reward_type=RewardType.GLOBAL,
    )
    obs, _ = env.reset()
    for _ in range(200):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_obs_space_4():
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
        observation_type=ObservationType.IMAGE_DICT,
        reward_type=RewardType.GLOBAL,
    )
    obs, _ = env.reset()
    for _ in range(200):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_inactivity_0(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


def test_inactivity_1(env_0):
    env = env_0
    for i in range(4):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done

    (
        _,
        reward,
        _,
        _,
        _,
    ) = env.step([Action.FORWARD])
    assert reward[0] == pytest.approx(1.0)
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done

    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


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
    env.reset()

    for _ in range(time_limit - 1):
        _, _, done, _, _ = env.step(env.action_space.sample())
        assert not done

    _, _, done, _, _ = env.step(env.action_space.sample())
    assert done


def test_inactivity_2(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done
    env.reset()
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


def test_fast_obs_0():
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=10,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        observation_type=ObservationType.DICT,
    )
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 2
        assert len(slow_obs) == 2

        flattened_slow = [
            gym.spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)
        ]
        for i in range(len(fast_obs)):
            assert list(fast_obs[i]) == list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())


def test_fast_obs_1():
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=3,
        n_agents=3,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=10,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        observation_type=ObservationType.DICT,
    )
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [
            gym.spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)
        ]

        for i in range(len(fast_obs)):
            assert list(fast_obs[i]) == list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())


def test_fast_obs_2():
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=3,
        n_agents=3,
        msg_bits=2,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=10,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        observation_type=ObservationType.DICT,
    )
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [
            gym.spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)
        ]

        for i in range(len(fast_obs)):
            assert np.array_equal(
                fast_obs[i].astype(np.int32), flattened_slow[i].astype(np.int32)
            )

        env._use_slow_obs()
        env.step(env.action_space.sample())


def test_reproducibility(env_0):
    env = env_0
    episodes_per_seed = 5
    for seed in range(5):
        obss1 = []
        grid1 = []
        highways1 = []
        request_queue1 = []
        player_x1 = []
        player_y1 = []
        player_carrying1 = []
        player_has_delivered1 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss1.append(np.array(obss).copy())
            grid1.append(env.unwrapped.grid.copy())
            highways1.append(env.unwrapped.highways.copy())
            request_queue1.append(
                np.array([shelf.id for shelf in env.unwrapped.request_queue])
            )
            player_x1.append([p.x for p in env.unwrapped.agents])
            player_y1.append([p.y for p in env.unwrapped.agents])
            player_carrying1.append([p.carrying_shelf for p in env.unwrapped.agents])
            player_has_delivered1.append(
                [p.has_delivered for p in env.unwrapped.agents]
            )

        obss2 = []
        grid2 = []
        highways2 = []
        request_queue2 = []
        player_x2 = []
        player_y2 = []
        player_carrying2 = []
        player_has_delivered2 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss2.append(np.array(obss).copy())
            grid2.append(env.unwrapped.grid.copy())
            highways2.append(env.unwrapped.highways.copy())
            request_queue2.append(
                np.array([shelf.id for shelf in env.unwrapped.request_queue])
            )
            player_x2.append([p.x for p in env.unwrapped.agents])
            player_y2.append([p.y for p in env.unwrapped.agents])
            player_carrying2.append([p.carrying_shelf for p in env.unwrapped.agents])
            player_has_delivered2.append(
                [p.has_delivered for p in env.unwrapped.agents]
            )

        for i, (obs1, obs2) in enumerate(zip(obss1, obss2)):
            assert np.array_equal(
                obs1, obs2
            ), f"Observations of env not identical for episode {i} with seed {seed}"
        for i, (g1, g2) in enumerate(zip(grid1, grid2)):
            assert np.array_equal(
                g1, g2
            ), f"Grid of env not identical for episode {i} with seed {seed}"
        for i, (h1, h2) in enumerate(zip(highways1, highways2)):
            assert np.array_equal(
                h1, h2
            ), f"Highways of env not identical for episode {i} with seed {seed}"
        for i, (rq1, rq2) in enumerate(zip(request_queue1, request_queue2)):
            assert np.array_equal(
                rq1, rq2
            ), f"Request queue of env not identical for episode {i} with seed {seed}"
        for i, (px1, px2) in enumerate(zip(player_x1, player_x2)):
            assert np.array_equal(
                px1, px2
            ), f"Player x of env not identical for episode {i} with seed {seed}"
        for i, (py1, py2) in enumerate(zip(player_y1, player_y2)):
            assert np.array_equal(
                py1, py2
            ), f"Player y of env not identical for episode {i} with seed {seed}"
        for i, (pc1, pc2) in enumerate(zip(player_carrying1, player_carrying2)):
            assert np.array_equal(
                pc1, pc2
            ), f"Player carrying of env not identical for episode {i} with seed {seed}"
        for i, (pd1, pd2) in enumerate(
            zip(player_has_delivered1, player_has_delivered2)
        ):
            assert np.array_equal(
                pd1, pd2
            ), f"Player has delivered of env not identical for episode {i} with seed {seed}"
