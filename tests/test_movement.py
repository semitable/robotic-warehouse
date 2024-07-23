import os
import sys

import pytest

from rware.warehouse import Warehouse, Direction, Action, RewardType


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()

    return env


@pytest.fixture
def env_two_agents():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_three_agents():
    env = Warehouse(3, 8, 3, 3, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_four_agents():
    env = Warehouse(3, 8, 3, 4, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_five_agents():
    env = Warehouse(3, 8, 3, 5, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


def test_simple_movement_down(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 26


def test_simple_movement_up(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 24


def test_simple_movement_left(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 25


def test_simple_movement_right(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 5
    assert env.agents[0].y == 25


def test_movement_under_shelf(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 0  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    for i in range(10):
        env.step([Action.FORWARD])

        assert env.agents[0].x == min(i + 1, 9)
        assert env.agents[0].y == 25


def test_simple_wall_collision_up(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 0
    env.agents[0].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 0


def test_simple_wall_collision_down(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 28
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 28


def test_simple_wall_collision_right(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 9  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 9
    assert env.agents[0].y == 25


def test_simple_wall_collision_left(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 0  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 0
    assert env.agents[0].y == 25


def test_head_collision_0(env_two_agents):
    env = env_two_agents

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 5  # should place it next to the other
    env.agents[1].y = 25
    env.agents[1].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 5
    assert env.agents[1].y == 25


def test_head_collision_1(env_two_agents):
    env = env_two_agents

    env.agents[0].x = env.shelfs[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = 5  # should place it next to the other
    env.agents[1].y = 25
    env.agents[1].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 5
    assert env.agents[1].y == 25


def test_head_collision_2(env_two_agents):
    env = env_two_agents

    env.agents[0].x = env.shelfs[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = env.shelfs[1].x = 5  # should place it next to the other
    env.agents[1].y = env.shelfs[1].y = 25
    env.agents[1].dir = Direction.LEFT
    env.agents[1].carrying_shelf = env.shelfs[1]
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 5
    assert env.agents[1].y == 25


def test_head_collision_3(env_two_agents):
    env = env_two_agents

    env.agents[0].x = env.shelfs[0].x = 3  # should place it in the middle (empty space)
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env.agents[0].carrying_shelf = env.shelfs[0]

    env.agents[1].x = 2  # should place it next to the other
    env.agents[1].y = 25
    env.agents[1].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 25
    assert env.agents[1].x == 2
    assert env.agents[1].y == 25


def test_chain_movement_1(env_two_agents):
    env = env_two_agents

    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 25
    env.agents[1].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 5
    assert env.agents[1].y == 25


def test_chain_movement_2(env_two_agents):
    env = env_two_agents

    env.agents[0].x = 8
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 9
    env.agents[1].y = 25
    env.agents[1].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 8
    assert env.agents[0].y == 25
    assert env.agents[1].x == 9
    assert env.agents[1].y == 25


def test_chain_movement_3(env_three_agents):
    env = env_three_agents

    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 25
    env.agents[1].dir = Direction.RIGHT

    env.agents[2].x = 5
    env.agents[2].y = 26
    env.agents[2].dir = Direction.UP

    env._recalc_grid()
    env.step(3 * [Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 5
    assert env.agents[1].y == 25
    assert env.agents[2].x == 5
    assert env.agents[2].y == 26


def test_circle_chain_movement_0(env_four_agents):
    env = env_four_agents
    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 25
    env.agents[1].dir = Direction.UP

    env.agents[2].x = 4
    env.agents[2].y = 24
    env.agents[2].dir = Direction.LEFT

    env.agents[3].x = 3
    env.agents[3].y = 24
    env.agents[3].dir = Direction.DOWN

    env._recalc_grid()
    env.step(4 * [Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25

    assert env.agents[1].x == 4
    assert env.agents[1].y == 24

    assert env.agents[2].x == 3
    assert env.agents[2].y == 24

    assert env.agents[3].x == 3
    assert env.agents[3].y == 25


def test_circle_chain_movement_1(env_five_agents):
    env = env_five_agents

    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 25
    env.agents[1].dir = Direction.UP

    env.agents[2].x = 4
    env.agents[2].y = 24
    env.agents[2].dir = Direction.LEFT

    env.agents[3].x = 3
    env.agents[3].y = 24
    env.agents[3].dir = Direction.DOWN

    env.agents[4].x = 5
    env.agents[4].y = 24
    env.agents[4].dir = Direction.LEFT

    env._recalc_grid()
    env.step(5 * [Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25

    assert env.agents[1].x == 4
    assert env.agents[1].y == 24

    assert env.agents[2].x == 3
    assert env.agents[2].y == 24

    assert env.agents[3].x == 3
    assert env.agents[3].y == 25

    # this stayed still:
    assert env.agents[4].x == 5
    assert env.agents[4].y == 24


def test_turn_right_0(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.RIGHT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.RIGHT


def test_turn_right_1(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.RIGHT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.DOWN


def test_turn_right_2(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.RIGHT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.LEFT


def test_turn_right_3(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.RIGHT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.UP


def test_turn_left_0(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.LEFT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.LEFT


def test_turn_left_1(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.LEFT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.UP


def test_turn_left_2(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.LEFT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.RIGHT


def test_turn_left_3(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.LEFT])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[0].dir == Direction.DOWN


def test_simple_carrying(env_single_agent):
    env = env_single_agent

    env.agents[0].x = env.shelfs[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.DOWN

    env.agents[0].carrying_shelf = env.shelfs[0]

    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 26
    assert env.shelfs[0].x == 4
    assert env.shelfs[0].y == 26


def test_simple_carrying_collision(env_single_agent):
    env = env_single_agent

    env.agents[0].x = env.shelfs[0].x = 3
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.LEFT

    env.agents[0].carrying_shelf = env.shelfs[0]

    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 25
    assert env.shelfs[0].x == 3
    assert env.shelfs[0].y == 25


def test_simple_carrying_chain(env_two_agents):
    env = env_two_agents

    env.agents[0].x = env.shelfs[0].x = 3
    env.agents[0].y = env.shelfs[0].y = 25
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = env.shelfs[1].x = 4
    env.agents[1].y = env.shelfs[1].y = 25
    env.agents[1].dir = Direction.RIGHT

    env.agents[0].carrying_shelf = env.shelfs[0]
    env.agents[1].carrying_shelf = env.shelfs[1]

    env._recalc_grid()
    env.step(2 * [Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.shelfs[0].x == 4
    assert env.shelfs[0].y == 25

    assert env.agents[1].x == 5
    assert env.agents[1].y == 25
    assert env.shelfs[1].x == 5
    assert env.shelfs[1].y == 25


def test_pickup_and_carry_0(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD])

    env.step([Action.TOGGLE_LOAD])
    assert env.agents[0].carrying_shelf is not None
    shelf = env.agents[0].carrying_shelf
    assert shelf.x == 2
    assert shelf.y == 25
    env.step([Action.LEFT])
    env.step([Action.LEFT])
    env.step([Action.FORWARD])
    assert env.agents[0].x == 3
    assert env.agents[0].y == 25
    assert shelf.x == 3
    assert shelf.y == 25

    env.step([Action.FORWARD])
    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert shelf.x == 4
    assert shelf.y == 25
    env.step([Action.TOGGLE_LOAD])  # cannot unload on highway
    env.step([Action.FORWARD])
    assert env.agents[0].x == 5
    assert env.agents[0].y == 25
    assert shelf.x == 5
    assert shelf.y == 25


def test_pickup_and_carry_1(env_single_agent):
    env = env_single_agent

    env.agents[0].x = 3
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT
    env._recalc_grid()
    env.step([Action.FORWARD])
    env.step([Action.TOGGLE_LOAD])
    assert env.agents[0].carrying_shelf is not None
    shelf = env.agents[0].carrying_shelf
    assert shelf.x == 2
    assert shelf.y == 25
    env.step([Action.LEFT])
    env.step([Action.LEFT])
    env.step([Action.TOGGLE_LOAD])  # can unload here
    env.step([Action.FORWARD])
    assert env.agents[0].x == 3
    assert env.agents[0].y == 25
    assert shelf.x == 2
    assert shelf.y == 25
