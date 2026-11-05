import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import train_recurrent_ippo_consensus as T
from examples.train_pgct_spatial_experiments import (
    MonitoredEnv, SpatialMonitor, build_jobs, geometric_adjacency,
)


def source_checkpoint():
    cfg = vars(T.build_arg_parser().parse_args([
        "--env-id", "rware-multiteam-tiny-4ag-2teams-v0",
        "--observation-format", "semantic", "--obs-encoder", "cnn", "--sensor-range", "5",
    ]))
    return dict(config=cfg, agents=[None] * 4,
                effective_env_kwargs=dict(sensor_range=5, n_teams=2, reveal_team_info=True))


def test_matched_branches_do_not_chain_weights_or_lose_workload(tmp_path):
    source = tmp_path / "source.pt"
    jobs = build_jobs(source_checkpoint(), source, tmp_path / "runs", maps=["e1", "e2", "e3"],
                      modes=["pgct", "ippo"], seeds=[1, 2], timesteps=4000000, requests=8)
    assert len(jobs) == 12
    for on, off in zip(jobs[::2], jobs[1::2]):
        assert on["config"]["init_checkpoint"] == off["config"]["init_checkpoint"] == str(source.resolve())
        assert on["effective_timesteps"] == off["effective_timesteps"] == 4000768
        assert on["config"]["seed"] == off["config"]["seed"]
        assert off["config"]["peer_transfer_mode"] == "none"
        assert off["config"]["pgct_peer_loss_coef"] == 0
        assert not off["config"]["communication_enabled"]
        assert on["config"]["peer_transfer_mode"] == "pgct"
        for job in (on, off):
            kw = T.apply_agent_overrides(job["config"]["env_id"],
                                        json.loads(job["config"]["env_kwargs_json"]), 4, 2, True)
            with T.make_env(job["config"]["env_id"], kw, "semantic") as env:
                env.reset(seed=job["seed"])
                assert list(env.unwrapped.grid_size) == job["grid_shape"]
                assert len(env.unwrapped.request_queue) == 8
                assert env.observation_space[0].shape == (1593,)


def test_independent_geometry_counts_neighbors_when_ippo_is_off():
    with T.make_env("rware-multiteam-tiny-4ag-2teams-v0",
                    dict(sensor_range=5, communication_enabled=False), "semantic") as env:
        env.reset(seed=1)
        env.unwrapped.agents[0].x, env.unwrapped.agents[0].y = 0, 0
        env.unwrapped.agents[1].x, env.unwrapped.agents[1].y = 5, 5
        assert not env.unwrapped.get_neighbor_adjacency().any()
        assert geometric_adjacency(env, 5)[0, 1]
        env.unwrapped.agents[1].x = 6
        assert not geometric_adjacency(env, 5)[0, 1]


def test_monitor_preserves_dynamics_observations_and_reset_rng():
    kwargs = dict(sensor_range=5, reveal_team_info=True)
    with T.make_env("rware-multiteam-tiny-4ag-2teams-v0", kwargs, "semantic") as control:
        monitor = SpatialMonitor(5)
        with MonitoredEnv(T.make_env("rware-multiteam-tiny-4ag-2teams-v0", kwargs, "semantic"),
                          monitor) as observed:
            control.reset(seed=17)
            observed.reset(seed=17)
            monitor.active = True
            for actions in np.random.default_rng(42).integers(0, 5, (12, 4)):
                expected = control.step(actions.tolist())
                actual = observed.step(actions.tolist())
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                assert actual[2:4] == expected[2:4]
            assert monitor.steps == 12
            np.testing.assert_array_equal(monitor.occupancy.sum((1, 2)), [12] * 4)
            np.testing.assert_array_equal(observed.reset()[0], control.reset()[0])


@pytest.mark.parametrize("kwargs", [dict(requests=3), dict(requests=34),
                                   dict(maps=["e1", "e1"]), dict(seeds=[-1])])
def test_invalid_experiment_inputs_fail_before_training(tmp_path, kwargs):
    params = dict(maps=["e1"], modes=["pgct", "ippo"], seeds=[1], timesteps=1024)
    params.update(kwargs)
    with pytest.raises(ValueError):
        build_jobs(source_checkpoint(), "source.pt", tmp_path, **params)
