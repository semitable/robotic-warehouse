import gymnasium as gym
import numpy as np
import pytest
import torch
import rware
from rware.multi_team_warehouse import MultiTeamWarehouse
from rware.utils.semantic_observation import RwareSemanticObservationWrapper
from examples.train_recurrent_ippo_consensus import RecurrentActorCritic, evaluate
from examples.train_recurrent_ippo_consensus import (
    TeamGraphEstimator, TrainConfig, build_arg_parser, baseline_weight_matrix,
    sample_policy_probe_sequences, enforce_communication_config, communication_weights,
)
from rware.warehouse import Direction
from rware.utils.semantic_observation import build_semantic_observation


def test_ex5_threshold_is_not_an_uncertainty_detector():
    graph = TeamGraphEstimator(3, .05, 1, 0, .95, gate_style="threshold")
    distance = np.array([[0, 0, .12], [0, 0, .2], [.12, .2, 0]])
    graph.update_from_policy_distance(distance, np.ones((3, 3)), beta=1)
    assert not graph.pgct_gate_matrix(False, .12, .25, 1).any()
    gate = graph.pgct_gate_matrix(True, .12, .25, 1)
    # Equal random policies may pass after warm-up, as explicitly stated in ex5.
    assert gate[0, 1] == gate[1, 0] == 1
    assert gate[0, 2] == gate[2, 0] == 0
    assert np.count_nonzero(gate) == 2


def test_fixed_common_bank_is_side_effect_free_and_reused():
    import copy
    cfg = TrainConfig(**vars(build_arg_parser().parse_args([
        "--fixed-canonical-probes", "--probe-source", "objective",
        "--policy-probe-team-conditioning", "shared", "--policy-probe-batch-size", "48",
    ])))
    with RwareSemanticObservationWrapper(gym.make(
        "rware-multiteam-tiny-4ag-2teams-v0", sensor_range=5,
    )) as env:
        env.reset(seed=7)
        base = env.unwrapped
        grid = base.grid.copy()
        state = copy.deepcopy(base.np_random.bit_generator.state)
        rng = np.random.default_rng(9)
        rng_state = copy.deepcopy(rng.bit_generator.state)
        # Objective probes must not need a rollout at all.
        bank, meta = sample_policy_probe_sequences(cfg, env, None, rng)
        bank_copy = bank.clone()
        bank.zero_()  # caller mutation cannot corrupt the canonical bank
        again, _ = sample_policy_probe_sequences(cfg, env, None, rng)
        assert torch.equal(again, bank_copy)
        assert again.ndim == 3 and again.shape[0] == 48
        assert meta["probe_team_conditioned"] == 0
        np.testing.assert_array_equal(base.grid, grid)
        assert base.np_random.bit_generator.state == state
        assert rng.bit_generator.state == rng_state


def test_baselines_and_independent_override():
    with gym.make("rware-multiteam-tiny-4ag-2teams-v0") as env:
        env.reset(seed=1)
        same = baseline_weight_matrix(env, 4, "oracle")
        wrong = baseline_weight_matrix(env, 4, "wrong")
        full = baseline_weight_matrix(env, 4, "unrestricted")
        assert not (same * wrong).any()
        np.testing.assert_array_equal(same + wrong, full)
        assert same.sum() == 4 and wrong.sum() == 8
        cfg = TrainConfig(**vars(build_arg_parser().parse_args([
            "--no-communication", "--graph-mode", "unrestricted", "--peer-transfer-mode", "pgct",
        ])))
        enforce_communication_config(cfg)
        assert cfg.graph_mode == cfg.peer_transfer_mode == cfg.comm_graph_mode == "none"
        assert cfg.probe_interval == cfg.consensus_interval == 0
        assert not communication_weights(cfg, env, None, 999).any()


def test_rear_agent_is_observed_and_heading_does_not_change_graph():
    with RwareSemanticObservationWrapper(gym.make(
        "rware-multiteam-tiny-4ag-2teams-v0", sensor_range=5,
        communication_topology="complete",
    )) as env:
        env.reset(seed=1)
        base = env.unwrapped
        focal, rear = base.agents[:2]
        focal.x, focal.y, focal.dir = 3, 3, Direction.RIGHT
        rear.x, rear.y = 2, 3
        base._recalc_grid()
        spec = env.semantic_observation_spec
        spatial = build_semantic_observation(base, focal)[spec.ego_dim:].reshape(spec.spatial_channels, 11, 11)
        assert spatial[5, 5, 4] == 1
        before = base.get_neighbor_adjacency()
        focal.dir = Direction.LEFT
        np.testing.assert_array_equal(before, base.get_neighbor_adjacency())
        base.set_training_comm_weights(before)
        assert not base.training_comm_directed
        assert len(base.training_comm_edges) == 6

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("enabled", [True, False])
def test_evaluation_video_uses_current_geometry_and_gate(monkeypatch, n, enabled):
    torch.set_num_threads(1)
    snapshots = []
    gate = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(gate, 0)

    def capture(base):
        np.testing.assert_array_equal(base.training_comm_adj, gate * base.get_neighbor_adjacency())
        snapshots.append(base.training_comm_adj.copy())
        return np.zeros((24, 24, 3), dtype=np.uint8)

    monkeypatch.setattr(MultiTeamWarehouse, "render", capture)
    kwargs = dict(n_agents=n, sensor_range=5, communication_topology="complete",
                  communication_enabled=enabled)
    with RwareSemanticObservationWrapper(gym.make("rware-multiteam-tiny-4ag-2teams-v0", **kwargs)) as env:
        spec = env.semantic_observation_spec
        nets = [RecurrentActorCritic(spec.obs_dim, 5, 8, 8, "cnn", spec.spatial_channels,
                                    spec.spatial_size, spec.ego_dim) for _ in range(n)]
    _, video = evaluate("rware-multiteam-tiny-4ag-2teams-v0", kwargs, "semantic", nets,
                        1, 3, .99, spec.obs_dim, 8, torch.device("cpu"), 9,
                        collect_video=True, communication_gate=gate)
    assert len(snapshots) == len(video) == 4
    assert any(frame.any() for frame in snapshots) if enabled else not any(frame.any() for frame in snapshots)
