import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import train_recurrent_ippo_consensus as T
from examples.train_pgct_physical_mixing_curriculum import build_stages


def checkpoint():
    cfg = vars(T.build_arg_parser().parse_args([
        "--env-id", "rware-multiteam-tiny-4ag-2teams-v0",
        "--observation-format", "semantic", "--obs-encoder", "cnn", "--sensor-range", "5",
    ]))
    return dict(config=cfg, agents=[None] * 4,
                effective_env_kwargs=dict(sensor_range=5, n_teams=2,
                                          communication_topology="complete", communication_range=99))


def test_curriculum_overrides_complete_graph_and_chains_all_weights(tmp_path):
    stages = build_stages(checkpoint(), tmp_path/'09700.pt', tmp_path/'output',
                          home_ratios=[1, .875, .75, .5], timesteps=[1024]*4)
    predecessor = str((tmp_path/'09700.pt').resolve())
    for stage in stages:
        cfg = T.TrainConfig(**stage['config'])
        assert cfg.init_checkpoint == predecessor
        assert cfg.transfer_components == 'all'
        assert cfg.comm_graph_mode == 'physical' and cfg.peer_transfer_mode == 'pgct'
        assert cfg.communication_enabled
        env_kwargs = json.loads(cfg.env_kwargs_json)
        assert env_kwargs['communication_topology'] == 'physical'
        assert env_kwargs['communication_range'] == env_kwargs['sensor_range'] == 5
        assert env_kwargs['shelf_soft_zone_ratio'] == stage['home_ratio']
        predecessor = stage['output_checkpoint']


@pytest.mark.parametrize('ratio', [1, .875, .75, .5])
def test_stage_environment_has_exact_balanced_mixing_and_local_transfer(tmp_path, ratio):
    ratios = [1] if ratio == 1 else [1, ratio]
    cfg = build_stages(checkpoint(), tmp_path/'09700.pt', tmp_path/'output',
                       home_ratios=ratios, timesteps=[1024]*len(ratios))[-1]['config']
    with T.make_env(cfg['env_id'], json.loads(cfg['env_kwargs_json']), 'semantic') as env:
        env.reset(seed=1)
        base = env.unwrapped
        owned_home = [base.shelf_team_ids[s.id] == base._home_zone_id(s.x, s.y) for s in base.shelfs]
        assert np.mean(owned_home) == ratio
        assert len(base.shelfs_by_team[0]) == len(base.shelfs_by_team[1])
        focal, other = base.agents[:2]
        focal.x, focal.y = 0, 0
        other.x, other.y = 5, 5  # Boundary of square sensor footprint.
        gate = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(gate, 0)
        base.set_training_comm_weights(gate)
        assert base.training_comm_adj[0, 1] == 1
        other.x = 6  # Same PGCT eligibility, now outside sensor range.
        base.update_training_comm_graph()
        assert base.training_comm_adj[0, 1] == 0
        neighbors = T.communication_adjacency(env, 4, 'physical')
        allocation_gate = T.mask_gate_by_current_neighbors(gate, neighbors)
        assert allocation_gate[0, 1] == allocation_gate[1, 0] == 0


@pytest.mark.parametrize('ratios,budgets', [([.75], [1024]), ([1,.5,.75], [1024]*3),
                                           ([1,.75], [1024]), ([1], [0])])
def test_invalid_stage_schedule_is_rejected(tmp_path, ratios, budgets):
    with pytest.raises(ValueError):
        build_stages(checkpoint(), '09700.pt', tmp_path, home_ratios=ratios, timesteps=budgets)
