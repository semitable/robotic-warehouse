"""Regressions for idle teammates, return-state aliasing and CNN training."""
import gymnasium as gym
import numpy as np
import pytest
import torch

import rware
from rware.warehouse import Action, Direction
from rware.utils.semantic_observation import (
    RwareSemanticObservationWrapper, build_semantic_observation,
    get_semantic_observation_spec,
)
from rware.utils.probe_maps import RwareObjectiveProbeMapBuilder
from examples.train_recurrent_ippo_consensus import (
    RecurrentActorCritic, RunnerState, TeamGraphEstimator, TrainConfig,
    build_arg_parser, collect_rollout, compute_gae, local_rware_action_masks,
    update_ippo,
)


def make_env(n=4, ratio=1.0, **kwargs):
    return RwareSemanticObservationWrapper(gym.make(
        "rware-multiteam-tiny-4ag-2teams-v0", n_agents=n, sensor_range=5,
        shelf_team_mode="balanced_soft_zones", shelf_soft_zone_ratio=ratio,
        goal_team_mode="zones", request_queue_size=n,
        request_queue_size_per_team=n // 2, team_reward_mode="individual",
        require_delivered_shelf_return=True, normalised_coordinates=True,
        **kwargs,
    ))


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("ratio", [1.0, 0.875, 0.75, 0.5])
def test_balanced_shelf_mixing(n, ratio):
    with make_env(n, ratio) as env:
        obs, _ = env.reset(seed=9)
        counts = env.unwrapped.get_shelf_zone_team_counts()
        np.testing.assert_array_equal(counts.sum(axis=0), [16, 16])
        assert counts[0, 0] == round(16 * ratio)
        assert counts[1, 1] == round(16 * ratio)
        assert len(obs) == n
        assert [len(q) for q in env.unwrapped.team_request_queues] == [n // 2] * 2


def test_return_phase_is_observable_and_probe_matches_environment():
    with make_env() as env:
        env.reset(seed=9)
        base = env.unwrapped
        agent = base.agents[0]
        shelf = base.team_request_queues[int(base.agent_team_ids[0])][0]
        agent.carrying_shelf = shelf
        shelf.x, shelf.y = agent.x, agent.y
        base._recalc_grid()
        before = build_semantic_observation(base, agent)
        base._shelves_awaiting_return.add(shelf.id)
        after = build_semantic_observation(base, agent)
        spec = get_semantic_observation_spec(env)
        idx = spec.ego_features.index("self_return_phase")
        assert before[idx] == 0 and after[idx] == 1
        assert after[spec.ego_features.index("self_shelf_home_known")] == 1
        builder = RwareObjectiveProbeMapBuilder(env, spec.obs_dim)
        try:
            probes = builder.build_for_team(0, 24)
            for i in (4, 5):
                assert probes[i, idx] == 1
                assert probes[i, spec.ego_features.index("self_carrying_requested_shelf")] == 1
                assert builder.scenario_for_team(i, 0).name.startswith("team_return")
        finally:
            builder.probe_env.close()


def test_idle_agent_does_not_get_progress_from_loaded_teammate():
    with make_env(requested_shelf_progress_reward=1.0,
                  reward_only_new_best_progress=False) as env:
        env.reset(seed=2)
        base = env.unwrapped
        idle, carrier = base.agents[0], base.agents[2]
        assert base.agent_team_ids[0] == base.agent_team_ids[2]
        shelf = base.team_request_queues[0][0]
        base.team_request_queues[0] = [shelf]
        base._sync_global_request_queue()
        for i, agent in enumerate(base.agents):
            agent.x, agent.y = i * 2, 0
        idle.x, idle.y = 1, 0
        carrier.x, carrier.y, carrier.dir = 4, 0, Direction.LEFT
        carrier.carrying_shelf = shelf
        shelf.x, shelf.y = carrier.x, carrier.y
        base._recalc_grid()
        base._reset_progress_baselines()
        assert base._nearest_requested_shelf_distance(idle) is None
        _, rewards, _, _, _ = env.step([Action.LEFT, Action.NOOP, Action.FORWARD, Action.NOOP])
        assert carrier.x == 3
        assert rewards[0] == pytest.approx(0.0)


def test_loaded_convoy_is_not_masked():
    with make_env() as env:
        env.reset(seed=1)
        base = env.unwrapped
        for i, agent in enumerate(base.agents):
            agent.x, agent.y = 6 + i, 0
        for i in (0, 1):
            agent = base.agents[i]
            agent.x, agent.y, agent.dir = 1 + i, 0, Direction.RIGHT
            agent.carrying_shelf = base.shelfs[i]
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
        base._recalc_grid()
        masks = local_rware_action_masks(env, 4, 5, True)
        assert masks[0, Action.FORWARD.value] == 1
        env.step([Action.FORWARD, Action.FORWARD, Action.NOOP, Action.NOOP])
        assert [a.x for a in base.agents[:2]] == [2, 3]


def test_timeout_bootstrap_does_not_leak_across_reset():
    rewards = torch.tensor([[1.0], [100.0]])
    _, returns = compute_gae(rewards, torch.ones(2), torch.zeros_like(rewards),
                             torch.zeros(1), 0.9, 0.95, torch.tensor([[2.0], [0.0]]))
    assert returns[:, 0].tolist() == pytest.approx([2.8, 100.0])


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("transfer", ["none", "pgct"])
def test_every_cnn_actor_updates_and_timeouts_bootstrap(n, transfer):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    with make_env(n, max_steps=4, repeated_stationary_action_penalty=0.006,
                  stationary_streak_penalty_after=1) as env:
        obs, _ = env.reset(seed=7)
        spec = get_semantic_observation_spec(env)
        nets = [RecurrentActorCritic(spec.obs_dim, 5, 16, 16, "cnn",
                                    spec.spatial_channels, spec.spatial_size, spec.ego_dim)
                for _ in range(n)]
        state = RunnerState(obs, torch.zeros(n, 16), torch.zeros(n, 16),
                            np.zeros(n), np.zeros(n), np.zeros(n))
        rollout, _, _ = collect_rollout(env, nets, state, 8, spec.obs_dim, 5, 16,
                                        torch.device("cpu"), True)
        assert rollout.dones.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
        assert torch.any(rollout.timeout_values[3] != 0)
        cfg = TrainConfig(**vars(build_arg_parser().parse_args([])))
        cfg.sequence_length, cfg.minibatch_chunks, cfg.ppo_epochs = 4, 2, 1
        cfg.peer_transfer_mode = transfer
        graph = TeamGraphEstimator(n, 0.05, 1, 0.0, 0.95)
        before = [net.actor_head.weight.detach().clone() for net in nets]
        builder = RwareObjectiveProbeMapBuilder(env, spec.obs_dim)
        try:
            probes = builder.build(48).unsqueeze(1)
        finally:
            builder.probe_env.close()
        allocation = (np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)) / (n - 1)
        metrics = update_ippo(nets, [torch.optim.Adam(net.parameters(), lr=0.001) for net in nets],
                              rollout, cfg, graph, torch.device("cpu"), np.random.default_rng(7),
                              allocation, probes)
        assert np.isfinite(metrics["policy_loss"])
        assert all(not torch.equal(old, net.actor_head.weight) for old, net in zip(before, nets))
