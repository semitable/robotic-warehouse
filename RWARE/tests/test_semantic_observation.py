import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rware  # noqa: F401
from examples.train_recurrent_ippo_consensus import RecurrentActorCritic
from rware.utils.semantic_observation import (
    RwareSemanticObservationWrapper,
    build_semantic_observation,
    get_semantic_observation_spec,
)
from rware.multi_team_warehouse import MultiTeamWarehouse, TeamRewardMode
from rware.warehouse import Action, Direction, RewardType


def _cell_value(obs, spec, channel, sensor_range, dy, dx):
    spatial = obs[spec.ego_dim :].reshape(
        spec.spatial_channels,
        spec.spatial_size,
        spec.spatial_size,
    )
    return float(spatial[channel, dy + sensor_range, dx + sensor_range])


def test_semantic_observation_distinguishes_loaded_neighbor_from_shelf_overlap():
    env = gym.make(
        "rware-multiteam-tiny-4ag-2teams-v0",
        sensor_range=3,
        request_queue_size=2,
        request_queue_size_per_team=1,
        reveal_team_info=True,
    )
    try:
        env.reset(seed=3)
        unwrapped = env.unwrapped
        focal = unwrapped.agents[0]
        neighbor = unwrapped.agents[1]
        shelf = unwrapped.shelfs[0]

        focal.x, focal.y, focal.dir = 3, 3, Direction.RIGHT
        neighbor.x, neighbor.y, neighbor.dir = 4, 3, Direction.LEFT
        shelf.x, shelf.y = neighbor.x, neighbor.y
        neighbor.carrying_shelf = None
        unwrapped._recalc_grid()

        spec = get_semantic_observation_spec(env)
        obs = build_semantic_observation(unwrapped, focal)
        assert _cell_value(obs, spec, 5, unwrapped.sensor_range, 0, 1) == 1.0
        assert _cell_value(obs, spec, 3, unwrapped.sensor_range, 0, 1) == 1.0
        assert _cell_value(obs, spec, 10, unwrapped.sensor_range, 0, 1) == 0.0

        neighbor.carrying_shelf = shelf
        shelf.x, shelf.y = neighbor.x, neighbor.y
        unwrapped._recalc_grid()
        obs = build_semantic_observation(unwrapped, focal)
        assert _cell_value(obs, spec, 5, unwrapped.sensor_range, 0, 1) == 1.0
        assert _cell_value(obs, spec, 3, unwrapped.sensor_range, 0, 1) == 1.0
        assert _cell_value(obs, spec, 10, unwrapped.sensor_range, 0, 1) == 1.0
    finally:
        env.close()


def test_semantic_wrapper_and_cnn_actor_critic_forward():
    env = RwareSemanticObservationWrapper(
        gym.make(
            "rware-multiteam-tiny-4ag-2teams-v0",
            sensor_range=3,
            request_queue_size=2,
            request_queue_size_per_team=1,
        )
    )
    try:
        obs, _ = env.reset(seed=5)
        spec = get_semantic_observation_spec(env)
        obs_dim = env.observation_space[0].shape[0]
        assert obs_dim == spec.obs_dim
        assert np.asarray(obs[0]).shape == (spec.obs_dim,)

        net = RecurrentActorCritic(
            obs_dim=obs_dim,
            action_dim=env.action_space[0].n,
            mlp_hidden_dim=32,
            recurrent_hidden_dim=32,
            encoder_type="cnn",
            spatial_channels=spec.spatial_channels,
            spatial_size=spec.spatial_size,
            ego_dim=spec.ego_dim,
        )
        actor_h, critic_h = net.initial_hidden(1, torch.device("cpu"))
        action, log_prob, value, next_actor_h, next_critic_h = net.act(
            torch.as_tensor(obs[0], dtype=torch.float32),
            actor_h.squeeze(0),
            critic_h.squeeze(0),
        )
        assert 0 <= int(action.item()) < env.action_space[0].n
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        assert next_actor_h.shape == (1, 32)
        assert next_critic_h.shape == (1, 32)
    finally:
        env.close()


def test_semantic_observation_tracks_action_and_blocked_history():
    env = RwareSemanticObservationWrapper(
        MultiTeamWarehouse(
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
            reveal_team_info=True,
        )
    )
    try:
        obs, _ = env.reset(seed=17)
        spec = get_semantic_observation_spec(env)
        feature_index = {
            name: idx for idx, name in enumerate(spec.ego_features)
        }
        assert obs[0][feature_index["self_prev_action_noop"]] == 1.0

        unwrapped = env.unwrapped
        unwrapped.agents[0].x = 0
        unwrapped.agents[0].y = 1
        unwrapped.agents[0].dir = Direction.RIGHT
        unwrapped.agents[0].carrying_shelf = None
        unwrapped.agents[1].x = 1
        unwrapped.agents[1].y = 1
        unwrapped.agents[1].dir = Direction.UP
        unwrapped.agents[1].carrying_shelf = None
        unwrapped._recalc_grid()
        unwrapped._reset_progress_baselines()

        obs, _, _, _, info = env.step([Action.FORWARD, Action.NOOP])

        assert info["failed_forwards"] == 1
        assert obs[0][feature_index["self_prev_action_forward"]] == 1.0
        assert obs[0][feature_index["self_blocked_steps"]] == pytest.approx(0.1)
        assert obs[1][feature_index["self_prev_action_noop"]] == 1.0
        assert obs[1][feature_index["self_waiting_steps"]] == pytest.approx(0.1)
    finally:
        env.close()

