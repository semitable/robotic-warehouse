import os
import sys

import gymnasium as gym
import numpy as np
import pytest


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.multi_team_grid import GridAction, MultiTeamGrid
from rware.multi_team_warehouse import MultiTeamWarehouse, TeamRewardMode
from rware.warehouse import Action, Direction, ObservationType, RewardType


def test_multiteam_rware_hides_team_info_by_default():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=3,
        n_agents=4,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=4,
        request_queue_size_per_team=2,
        max_inactivity_steps=None,
        max_steps=50,
        reward_type=RewardType.INDIVIDUAL,
        n_teams=2,
        team_assignments=[0, 1, 0, 1],
    )
    obs, info = env.reset(seed=7)

    assert env.observation_space.contains(obs)
    assert "agent_team_ids" not in info
    assert env.get_oracle_team_assignments().tolist() == [0, 1, 0, 1]
    assert len(env.request_queue) == 4
    for team_id, team_queue in enumerate(env.team_request_queues):
        assert len(team_queue) == 2
        assert all(env.shelf_team_ids[shelf.id] == team_id for shelf in team_queue)


def test_multiteam_rware_observes_only_own_team_requested_shelves():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=2,
        msg_bits=0,
        sensor_range=2,
        request_queue_size=2,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.xg\n.....",
        observation_type=ObservationType.DICT,
        n_teams=2,
        team_assignments=[0, 1],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
    )
    env.reset(seed=3)

    team0_shelf, team1_shelf = env.shelfs[:2]
    env.shelf_team_ids = {team0_shelf.id: 0, team1_shelf.id: 1}
    env.shelfs_by_team = [[team0_shelf], [team1_shelf]]
    env.team_request_queues = [[team0_shelf], [team1_shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 2
    env.agents[0].y = 0
    env.agents[0].dir = Direction.DOWN
    env.agents[1].x = 2
    env.agents[1].y = 2
    env.agents[1].dir = Direction.UP
    env._recalc_grid()

    def sensor_index(agent, shelf):
        side = 2 * env.sensor_range + 1
        dx = int(shelf.x) - int(agent.x)
        dy = int(shelf.y) - int(agent.y)
        return (dy + env.sensor_range) * side + (dx + env.sensor_range)

    obs0 = env._make_obs(env.agents[0])
    obs1 = env._make_obs(env.agents[1])
    team0_idx_for_agent0 = sensor_index(env.agents[0], team0_shelf)
    team1_idx_for_agent0 = sensor_index(env.agents[0], team1_shelf)
    team0_idx_for_agent1 = sensor_index(env.agents[1], team0_shelf)
    team1_idx_for_agent1 = sensor_index(env.agents[1], team1_shelf)

    assert obs0["sensors"][team0_idx_for_agent0]["shelf_requested"] == [1]
    assert obs0["sensors"][team1_idx_for_agent0]["shelf_requested"] == [0]
    assert obs1["sensors"][team0_idx_for_agent1]["shelf_requested"] == [0]
    assert obs1["sensors"][team1_idx_for_agent1]["shelf_requested"] == [1]

    slow_obs_space = env.observation_space[0]
    flat_obs0 = gym.spaces.flatten(slow_obs_space, obs0)
    flat_obs1 = gym.spaces.flatten(slow_obs_space, obs1)
    env._use_fast_obs()
    fast_obs0 = env._make_obs(env.agents[0])
    fast_obs1 = env._make_obs(env.agents[1])

    assert np.array_equal(fast_obs0, flat_obs0)
    assert np.array_equal(fast_obs1, flat_obs1)

    bits_per_sensor_cell = 7
    self_feature_count = 8
    requested_offset = 6

    def flat_requested_index(sensor_idx):
        return self_feature_count + sensor_idx * bits_per_sensor_cell + requested_offset

    assert fast_obs0[flat_requested_index(team0_idx_for_agent0)] == 1.0
    assert fast_obs0[flat_requested_index(team1_idx_for_agent0)] == 0.0
    assert fast_obs1[flat_requested_index(team0_idx_for_agent1)] == 0.0
    assert fast_obs1[flat_requested_index(team1_idx_for_agent1)] == 1.0


def test_multiteam_rware_rewards_only_latent_shelf_team():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=2,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=50,
        reward_type=RewardType.INDIVIDUAL,
        n_teams=2,
        team_assignments=[0, 1],
        team_reward_mode=TeamRewardMode.TEAM,
    )
    env.reset(seed=11)

    goal_x, goal_y = env.goals[0]
    shelf = env.shelfs[0]
    env.shelf_team_ids[shelf.id] = 0
    env.shelfs_by_team[0] = [shelf]
    env.team_request_queues = [[shelf], [env.shelfs[1]]]
    env._sync_global_request_queue()

    env.agents[0].x = shelf.x = goal_x
    env.agents[0].y = shelf.y = goal_y - 1
    env.agents[0].dir = Direction.DOWN
    env.agents[0].carrying_shelf = shelf
    env.agents[1].x = 0
    env.agents[1].y = 0
    env.agents[1].dir = Direction.RIGHT
    env._recalc_grid()

    _, rewards, _, _, info = env.step([Action.FORWARD, Action.NOOP])

    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.0)
    assert info["deliveries"] == 1


def test_multiteam_rware_reward_shaping_tracks_local_task_progress():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        requested_shelf_pickup_reward=0.2,
        requested_shelf_progress_reward=0.05,
        goal_progress_reward=0.1,
        premature_drop_penalty=0.15,
        reveal_team_info=True,
    )
    env.reset(seed=13)

    shelf = env.shelfs[0]
    shelf.x = 1
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()
    env.goals = [(3, 1)]
    env.goals_by_team = [[(3, 1)]]
    env.goal_team_ids = {(3, 1): 0}

    env.agents[0].x = 0
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env.agents[0].has_delivered = False
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)

    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.2)

    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(-0.15)

    _, rewards, _, _, _ = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.0)

    _, rewards, _, _, _ = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.1)

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(1.1)
    assert info["deliveries"] == 1
    assert env.agents[0].has_delivered

    _, rewards, _, _, info = env.step([Action.NOOP])
    assert rewards[0] == pytest.approx(0.0)
    assert info["deliveries"] == 0


def test_multiteam_rware_can_require_delivered_shelf_return_home():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=30,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        delivery_reward=1.0,
        require_delivered_shelf_return=True,
        shelf_return_reward=0.7,
        reveal_team_info=True,
    )
    env.reset(seed=19)

    shelf = env.shelfs[0]
    shelf.x = 1
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()
    env._shelf_home_positions = {shelf.id: (1, 1)}
    env.goals = [(3, 1)]
    env.goals_by_team = [[(3, 1)]]
    env.goal_team_ids = {(3, 1): 0}

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env.agents[0].has_delivered = False
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["deliveries"] == 0

    env.step([Action.FORWARD])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(1.0)
    assert info["deliveries"] == 1
    assert info["returns"] == 0
    assert shelf in env.request_queue
    assert shelf.id in env._shelves_awaiting_return
    assert env.agents[0].has_delivered
    assert env._carried_requested_shelf_goal_distance(env.agents[0]) == 2

    _, rewards, _, _, info = env.step([Action.NOOP])
    assert rewards[0] == pytest.approx(0.0)
    assert info["deliveries"] == 0

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    env.step([Action.FORWARD])
    env.step([Action.FORWARD])
    _, rewards, _, _, info = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.7)
    assert info["deliveries"] == 0
    assert info["returns"] == 1
    assert env.team_return_counts[0] == 1
    assert shelf.id not in env._shelves_awaiting_return
    assert env.agents[0].carrying_shelf is None
    assert (shelf.x, shelf.y) == (1, 1)


def test_multiteam_rware_return_progress_uses_return_distance_baseline():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=40,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        delivery_reward=1.0,
        require_delivered_shelf_return=True,
        shelf_return_reward=0.7,
        wrong_shelf_return_penalty=0.05,
        return_progress_reward=0.05,
        reveal_team_info=True,
    )
    env.reset(seed=23)

    shelf = env.shelfs[0]
    shelf.x = 1
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()
    env._shelf_home_positions = {shelf.id: (1, 1)}
    env.goals = [(3, 1)]
    env.goals_by_team = [[(3, 1)]]
    env.goal_team_ids = {(3, 1): 0}

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env.agents[0].has_delivered = False
    env._recalc_grid()
    env._reset_progress_baselines()

    _, _, _, _, info = env.step([Action.TOGGLE_LOAD])
    assert info["pickups"] == 1
    assert info["pickup_events"][0]["requested"]

    env.step([Action.FORWARD])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(1.0)
    assert info["deliveries"] == 1
    assert env._best_return_distances[0] == 2

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)
    assert env._best_return_distances[0] == 1

    _, rewards, _, _, info = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["invalid_toggles"] == 1
    assert shelf.id in env._shelves_awaiting_return
    assert env.agents[0].carrying_shelf is shelf

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)

    _, rewards, _, _, info = env.step([Action.TOGGLE_LOAD])
    assert rewards[0] == pytest.approx(0.7)
    assert info["returns"] == 1
    assert shelf.id not in env._shelves_awaiting_return


def test_multiteam_rware_movement_shaping_discourages_idle_actions():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        forward_movement_reward=0.03,
        stationary_action_penalty=0.04,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    shelf = env.shelfs[0]
    shelf.x = 3
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.LEFT])
    assert rewards[0] == pytest.approx(-0.04)
    assert info["reward_shaping"][0] == pytest.approx(-0.04)

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.03)
    assert info["reward_shaping"][0] == pytest.approx(0.03)


def test_multiteam_rware_movement_reward_can_require_task_progress():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        forward_movement_reward=0.03,
        movement_reward_requires_progress=True,
        requested_shelf_progress_reward=0.05,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    shelf = env.shelfs[0]
    shelf.x = 3
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.UP
    env.agents[0].carrying_shelf = None
    env._visited_cells_by_agent[0] = {(1, 1)}
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env._visited_cells_by_agent[0] = {(1, 1)}
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.08)
    assert info["reward_shaping"][0] == pytest.approx(0.08)


def test_multiteam_rware_progress_reward_only_pays_for_new_best_distance():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        requested_shelf_progress_reward=0.05,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    shelf = env.shelfs[0]
    shelf.x = 3
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)


def test_multiteam_rware_delivery_progress_does_not_reset_after_repickup():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=30,
        reward_type=RewardType.INDIVIDUAL,
        layout="......\n.x..g.\n......",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        goal_progress_reward=0.05,
        reveal_team_info=True,
    )
    env.reset(seed=29)

    shelf = env.shelfs[0]
    shelf.x = 1
    shelf.y = 1
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()
    env.goals = [(4, 1)]
    env.goals_by_team = [[(4, 1)]]
    env.goal_team_ids = {(4, 1): 0}

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env._recalc_grid()
    env._reset_progress_baselines()

    env.step([Action.TOGGLE_LOAD])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.05)
    assert info["reward_shaping"][0] == pytest.approx(0.05)

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)

    env.step([Action.TOGGLE_LOAD])
    env.step([Action.TOGGLE_LOAD])
    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)


def test_multiteam_rware_repeated_stationary_actions_are_penalized():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        turn_action_penalty=0.02,
        repeated_stationary_action_penalty=0.03,
        stationary_streak_penalty_after=1,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    shelf = env.shelfs[0]
    shelf.x = 4
    shelf.y = 2
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.UP
    env.agents[0].carrying_shelf = None
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.LEFT])
    assert rewards[0] == pytest.approx(-0.02)
    assert info["reward_shaping"][0] == pytest.approx(-0.02)

    _, rewards, _, _, info = env.step([Action.LEFT])
    assert rewards[0] == pytest.approx(-0.05)
    assert info["reward_shaping"][0] == pytest.approx(-0.05)

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.0)
    assert info["reward_shaping"][0] == pytest.approx(0.0)
    assert env._stationary_action_streak[0] == 0


def test_multiteam_rware_stationary_blocker_is_penalized():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0, 0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        failed_forward_penalty=0.01,
        blocking_agent_penalty=0.07,
        deadlock_penalty=0.05,
        deadlock_penalty_after=1,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    env.agents[0].x = 0
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env.agents[1].x = 1
    env.agents[1].y = 1
    env.agents[1].dir = Direction.UP
    env.agents[1].carrying_shelf = None
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD, Action.NOOP])

    assert rewards[0] == pytest.approx(-0.01)
    assert rewards[1] == pytest.approx(-0.07)
    assert info["failed_forwards"] == 1
    assert info["blocking_agents"] == 1
    assert info["deadlocked_agents"] == 0
    assert info["reward_shaping"][1] == pytest.approx(-0.07)
    assert info["blocking_agent_events"] == [
        {"agent_id": 2, "position": (1, 1), "blocked_agent_id": 1}
    ]

    _, rewards, _, _, info = env.step([Action.FORWARD, Action.NOOP])

    assert rewards[0] == pytest.approx(-0.06)
    assert rewards[1] == pytest.approx(-0.07)
    assert info["failed_forwards"] == 1
    assert info["blocking_agents"] == 1
    assert info["deadlocked_agents"] == 1
    assert info["reward_shaping"][0] == pytest.approx(-0.05)
    assert info["reward_shaping"][1] == pytest.approx(-0.07)
    assert info["deadlock_events"] == [
        {
            "agent_id": 1,
            "blocked_steps": 2,
            "position": (0, 1),
            "target": (1, 1),
        }
    ]


def test_multiteam_rware_new_cell_reward_is_not_repeatable_loop_reward():
    env = MultiTeamWarehouse(
        shelf_columns=3,
        column_height=1,
        shelf_rows=1,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        request_queue_size_per_team=1,
        max_inactivity_steps=None,
        max_steps=20,
        reward_type=RewardType.INDIVIDUAL,
        layout=".....\n.x.g.\n.....",
        n_teams=1,
        team_assignments=[0],
        team_reward_mode=TeamRewardMode.INDIVIDUAL,
        new_cell_reward=0.02,
        revisit_cell_penalty=0.03,
        reveal_team_info=True,
    )
    env.reset(seed=17)

    shelf = env.shelfs[0]
    shelf.x = 4
    shelf.y = 2
    env.shelf_team_ids = {shelf.id: 0}
    env.shelfs_by_team = [[shelf]]
    env.team_request_queues = [[shelf]]
    env._sync_global_request_queue()

    env.agents[0].x = 1
    env.agents[0].y = 1
    env.agents[0].dir = Direction.RIGHT
    env.agents[0].carrying_shelf = None
    env._visited_cells_by_agent[0] = {(1, 1)}
    env._recalc_grid()
    env._reset_progress_baselines()

    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(0.02)
    assert info["reward_shaping"][0] == pytest.approx(0.02)

    env.step([Action.LEFT])
    env.step([Action.LEFT])
    _, rewards, _, _, info = env.step([Action.FORWARD])
    assert rewards[0] == pytest.approx(-0.03)
    assert info["reward_shaping"][0] == pytest.approx(-0.03)


def test_multiteam_grid_collection_and_hidden_oracle():
    env = MultiTeamGrid(
        grid_size=(6, 6),
        n_agents=2,
        n_teams=2,
        sensor_range=1,
        max_steps=10,
        team_assignments=[0, 1],
        targets_per_team=1,
        obstacle_density=0.0,
        team_reward_mode=TeamRewardMode.TEAM,
    )
    obs, info = env.reset(seed=3)

    assert env.observation_space.contains(obs)
    assert "agent_team_ids" not in info
    assert env.get_oracle_team_assignments().tolist() == [0, 1]

    env.agent_positions = np.asarray([[1, 1], [4, 4]], dtype=np.int32)
    env.target_positions = {(2, 1): 0}

    obs, rewards, done, truncated, info = env.step([GridAction.DOWN, GridAction.NOOP])

    assert env.observation_space.contains(obs)
    assert rewards == pytest.approx([1.0, 0.0])
    assert not done
    assert not truncated
    assert info["collections"] == 1
    assert len(env.target_positions) == 1


def test_registered_multiteam_envs_can_step():
    import rware  # noqa: F401

    for env_id in [
        "rware-multiteam-tiny-4ag-2teams-v0",
        "mtgrid-small-4ag-2teams-v0",
    ]:
        env = gym.make(env_id)
        obs, info = env.reset(seed=5)
        assert env.observation_space.contains(obs)
        assert "agent_team_ids" not in info
        obs, rewards, done, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert len(rewards) == env.unwrapped.n_agents
