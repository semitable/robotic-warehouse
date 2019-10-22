import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse import Warehouse, Direction, Action


def test_simple_movement_down():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 26


def test_simple_movement_up():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.UP.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 24


def test_simple_movement_left():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 25


def test_simple_movement_right():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 5
    assert env.agents[0].y == 25


def test_simple_wall_collision_up():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 0
    env.agents[0].dir = Direction.UP.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 0


def test_simple_wall_collision_down():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 28
    env.agents[0].dir = Direction.DOWN.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 28


def test_simple_wall_collision_right():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 9  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.RIGHT.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 9
    assert env.agents[0].y == 25


def test_simple_wall_collision_left():
    grid_size = (29, 10)

    env = Warehouse(grid_size=grid_size, n_agents=1, msg_bits=0)
    env.reset()
    env.agents[0].x = 0  # should place it in the middle (empty space)
    env.agents[0].y = 25
    env.agents[0].dir = Direction.LEFT.value
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 0
    assert env.agents[0].y == 25
