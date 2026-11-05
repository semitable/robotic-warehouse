import os
import sys

import pytest

from rware.warehouse import Warehouse, Direction, Action, RewardType


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)


@pytest.fixture
def env_0():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
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


@pytest.fixture
def env_1():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelfs[0].x = 4
    env.shelfs[0].y = 27

    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = 3
    env.agents[1].y = 3

    env.request_queue[0] = env.shelfs[0]
    env._recalc_grid()
    return env


@pytest.fixture
def env_2():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.INDIVIDUAL)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelfs[0].x = 4
    env.shelfs[0].y = 27

    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = 3
    env.agents[1].y = 3

    env.request_queue[0] = env.shelfs[0]
    env._recalc_grid()
    return env


@pytest.fixture
def env_3():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.TWO_STAGE)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelfs[0].x = 4
    env.shelfs[0].y = 27

    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = 3
    env.agents[1].y = 3

    env.request_queue[0] = env.shelfs[0]
    env._recalc_grid()
    return env


def test_goal_location(env_0: Warehouse):
    assert env_0.goals[0] == (4, 28)
    assert env_0.goals[1] == (5, 28)


def test_goal_1(env_0: Warehouse):
    assert env_0.request_queue[0] == env_0.shelfs[0]

    _, rewards, _, _, _ = env_0.step([Action.FORWARD])
    assert env_0.agents[0].x == 4
    assert env_0.agents[0].y == 28

    assert env_0.request_queue[0] != env_0.shelfs[0]
    assert rewards[0] == pytest.approx(1.0)


def test_goal_2(env_1: Warehouse):
    assert env_1.request_queue[0] == env_1.shelfs[0]

    _, rewards, _, _, _ = env_1.step([Action.FORWARD, Action.NOOP])
    assert env_1.agents[0].x == 4
    assert env_1.agents[0].y == 28

    assert env_1.request_queue[0] != env_1.shelfs[0]
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(1.0)


def test_goal_3(env_2: Warehouse):
    env = env_2
    assert env.request_queue[0] == env.shelfs[0]

    _, rewards, _, _, _ = env.step([Action.FORWARD, Action.NOOP])
    assert env.agents[0].x == 4
    assert env.agents[0].y == 28

    assert env.request_queue[0] != env.shelfs[0]
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.0)


def test_goal_4(env_0: Warehouse):
    assert env_0.request_queue[0] == env_0.shelfs[0]

    _, rewards, _, _, _ = env_0.step([Action.LEFT])
    assert rewards[0] == pytest.approx(0.0)
    _, rewards, _, _, _ = env_0.step([Action.LEFT])
    assert rewards[0] == pytest.approx(0.0)
    _, rewards, _, _, _ = env_0.step([Action.FORWARD])
    assert env_0.agents[0].x == 4
    assert env_0.agents[0].y == 26

    assert env_0.request_queue[0] == env_0.shelfs[0]

    assert rewards[0] == pytest.approx(0.0)


def test_goal_5(env_3: Warehouse):
    env = env_3
    assert env.request_queue[0] == env.shelfs[0]

    _, rewards, _, _, _ = env.step([Action.FORWARD, Action.NOOP])
    assert env.agents[0].x == 4
    assert env.agents[0].y == 28

    assert env.request_queue[0] != env.shelfs[0]
    assert rewards[0] == pytest.approx(0.5)
    assert rewards[1] == pytest.approx(0.0)

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.shelfs[0].x = 1
    env.shelfs[0].y = 1
    env._recalc_grid()
    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD, Action.NOOP])

    assert rewards[0] == pytest.approx(0.5)
    assert rewards[1] == pytest.approx(0.0)
    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD, Action.NOOP])

    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)
    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD, Action.NOOP])

    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)
