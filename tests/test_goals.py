import os
import sys
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse import Warehouse, Direction, Action


@pytest.fixture
def env_0():
    grid_size = (29, 10)
    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
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


def test_goal_location(env_0: Warehouse):
    assert env_0.goals[0] == (4, 28)
    assert env_0.goals[1] == (5, 28)


def test_goal_1(env_0: Warehouse):
    assert env_0.request_queue[0] == env_0.shelfs[0]

    _, rewards, _, _ = env_0.step([Action.FORWARD])
    assert env_0.agents[0].x == 4
    assert env_0.agents[0].y == 28

    assert env_0.request_queue[0] != env_0.shelfs[0]
    assert rewards[0] == pytest.approx(1.0)


def test_goal_2(env_0: Warehouse):
    assert env_0.request_queue[0] == env_0.shelfs[0]

    _, rewards, _, _ = env_0.step([Action.LEFT])
    assert rewards[0] == pytest.approx(0.0)
    _, rewards, _, _ = env_0.step([Action.LEFT])
    assert rewards[0] == pytest.approx(0.0)
    _, rewards, _, _ = env_0.step([Action.FORWARD])
    assert env_0.agents[0].x == 4
    assert env_0.agents[0].y == 26

    assert env_0.request_queue[0] == env_0.shelfs[0]

    assert rewards[0] == pytest.approx(0.0)
