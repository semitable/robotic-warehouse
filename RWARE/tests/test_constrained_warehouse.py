"""Constraint dynamics, local visibility, reachability and opt-in isolation."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rware.constrained_warehouse import ConstrainedWarehouse, profile_kwargs
from rware.warehouse import Action, Direction
from examples.train_hard_rware_experiments import (
    T, ENV_ID, build_hard_jobs, constrained_runtime, preserved_hashes,
)

BASE = dict(n_agents=4, n_teams=2, msg_bits=0, sensor_range=5,
            request_queue_size=4, request_queue_size_per_team=2, max_steps=500,
            max_inactivity_steps=None, shelf_team_mode='balanced_soft_zones',
            shelf_soft_zone_ratio=.5, team_reward_mode='individual', goal_team_mode='zones',
            require_delivered_shelf_return=True, normalised_coordinates=True, reveal_team_info=True)


def make(profile):
    return T.make_env(ENV_ID, {**BASE, **profile_kwargs(profile)}, 'semantic')


def test_control_preserves_original_dynamics_and_future_reset_rng():
    kwargs = {**BASE, **profile_kwargs('control')}
    kwargs.pop('wall_cells');kwargs.pop('dock_cooldown_steps')
    original_mask = T.local_rware_action_masks
    with T.make_env('rware-multiteam-tiny-4ag-2teams-v0', kwargs, 'semantic') as old:
        with constrained_runtime(), make('control') as control:
            for seed in (17, None):
                a,_ = old.reset(seed=seed);b,_ = control.reset(seed=seed)
                np.testing.assert_array_equal(a,b)
                for actions in np.random.default_rng(52).integers(0,5,(40,4)):
                    expected=old.step(actions.tolist());actual=control.step(actions.tolist())
                    np.testing.assert_array_equal(expected[0],actual[0])
                    np.testing.assert_array_equal(expected[1],actual[1])
                    assert expected[2:4] == actual[2:4]
                    np.testing.assert_array_equal(original_mask(old,4,5,True),T.local_rware_action_masks(control,4,5,True))
    assert T.local_rware_action_masks is original_mask


def test_obstacle_spawns_local_observations_masks_and_collision_resolver():
    with constrained_runtime(), make('bottleneck') as env:
        base=env.unwrapped
        for seed in range(50):
            obs,_=env.reset(seed=seed)
            assert np.asarray(obs).shape == (4,1593)
            assert len({(a.x,a.y) for a in base.agents}) == 4
            assert not any((a.x,a.y) in base.wall_cells for a in base.agents)
            assert len(base.shelfs)==108
            assert nx.is_connected(base.floor_graph)
        base.agents[0].x,base.agents[0].y,base.agents[0].dir=0,6,Direction.DOWN
        base.agents[1].x,base.agents[1].y,base.agents[1].dir=0,5,Direction.DOWN
        base.agents[2].x,base.agents[2].y=15,0
        base.agents[3].x,base.agents[3].y=15,15
        base._recalc_grid()
        mask=T.local_rware_action_masks(env,4,5,True)
        assert mask[0,Action.FORWARD.value]==0
        assert T.local_rware_action_masks(env,4,5,False)[0,Action.FORWARD.value]==1
        spatial=env._semantic_obs()[0][20:].reshape(13,11,11)
        assert spatial[0,6,5]==1 and spatial[1:,6,5].sum()==0
        _,_,_,_,info=env.step([Action.FORWARD.value,Action.FORWARD.value,0,0])
        assert (base.agents[0].x,base.agents[0].y)==(0,6)
        assert (base.agents[1].x,base.agents[1].y)==(0,5)
        assert info['constraint_blocks'][0]['constraint']=='wall'
        assert len(info['failed_forward_events'])==2
        assert len(info['blocking_agent_events'])==1


def test_busy_dock_visible_masked_and_reopens_after_exact_cooldown():
    with constrained_runtime(), make('dock') as env:
        env.reset(seed=2);base=env.unwrapped
        goal=base.goals_by_team[0][0]
        a=base.agents[0];s=base.team_request_queues[0][0]
        a.x,a.y,a.dir=goal[0],goal[1]-1,Direction.DOWN
        a.carrying_shelf=s;s.x,s.y=a.x,a.y
        for i,cell in enumerate([(0,0),(15,0),(15,15)],start=1):
            base.agents[i].x,base.agents[i].y=cell
        base._recalc_grid();base.dock_timers[goal]=2
        obs=env._semantic_obs()[0][20:].reshape(13,11,11)
        assert obs[11,6,5]==1 and obs[12,6,5]==0
        assert T.local_rware_action_masks(env,4,5,True)[0,1]==0
        env.step([1,0,0,0])
        assert (a.x,a.y)!=(goal[0],goal[1]) and base.dock_timers[goal]==1
        env.step([0,0,0,0])
        assert base.dock_timers[goal]==0
        assert T.local_rware_action_masks(env,4,5,True)[0,1]==1
        obs=env._semantic_obs()[0][20:].reshape(13,11,11)
        assert obs[12,6,5]==1
        _,_,_,_,info=env.step([1,0,0,0])
        assert info['deliveries']==1 and base.dock_timers[goal]==8
        assert all(a not in base.wall_cells for a in base.goals)
        # Empty robots are allowed through a busy dock; carriers can leave it.
        a.dir=Direction.UP
        assert base.forward_constraint(a) is None
        empty=base.agents[1]
        empty.x,empty.y,empty.dir=goal[0],goal[1]-1,Direction.DOWN
        assert empty.carrying_shelf is None and base.forward_constraint(empty) is None


@pytest.mark.parametrize('profile',['control','detour','bottleneck','dock'])
def test_actual_pickup_delivery_and_return_cycle(profile):
    with constrained_runtime(), make(profile) as env:
        env.reset(seed=3);base=env.unwrapped;a=base.agents[0]
        shelf=next(s for s in base.shelfs_by_team[0] if s.y<7)
        base.team_request_queues[0][0]=shelf;base._sync_global_request_queue()
        home=(shelf.x,shelf.y);goal=base.goals_by_team[0][0]
        a.x,a.y=home
        for i,cell in enumerate([(0,15),(15,0),(15,15)],start=1):
            base.agents[i].x,base.agents[i].y=cell
        base._recalc_grid();base._reset_progress_baselines()
        _,_,_,_,info=env.step([4,0,0,0]);assert info['pickups']==1
        graph=base.floor_graph.copy();graph.add_node(home)
        for cell in [(home[0]-1,home[1]),(home[0]+1,home[1]),(home[0],home[1]-1),(home[0],home[1]+1)]:
            if cell in base.floor_graph:graph.add_edge(home,cell)
        path=nx.shortest_path(graph,home,goal)
        def walk(cells):
            delivered=0
            for x,y in cells:
                direction={(0,-1):Direction.UP,(0,1):Direction.DOWN,(-1,0):Direction.LEFT,(1,0):Direction.RIGHT}[(x-a.x,y-a.y)]
                while a.dir!=direction:env.step([3,0,0,0])
                assert T.local_rware_action_masks(env,4,5,True)[0,1]==1
                _,_,_,_,info=env.step([1,0,0,0]);delivered+=info['deliveries']
                assert (a.x,a.y)==(x,y)
            return delivered
        assert walk(path[1:])==1
        walk(list(reversed(path[:-1])))
        _,_,_,_,info=env.step([4,0,0,0])
        assert info['returns']==1 and base.team_return_counts[0]==1
        assert (shelf.x,shelf.y)==home and a.carrying_shelf is None


def test_routing_detours_capacity_and_cpu_rendering():
    with constrained_runtime():
        values={}
        for profile in ['control','detour','bottleneck','dock']:
            with make(profile) as env:
                env.reset(seed=9);base=env.unwrapped
                values[profile]=base.routing_report()
                json.dumps(values[profile])
                frame=env.render()
                assert frame.dtype==np.uint8 and frame.shape==(480,448,3)
        assert {v['shelves'] for v in values.values()}=={108}
        assert values['bottleneck']['mean_loaded_distance'] > values['control']['mean_loaded_distance']
        assert values['control']['mean_loaded_distance'] < values['detour']['mean_loaded_distance'] < values['bottleneck']['mean_loaded_distance']
        assert values['bottleneck']==values['dock']


@pytest.mark.parametrize('extra',[
    dict(wall_cells=[[1,1]]),dict(wall_cells=[[0,99]]),dict(dock_cooldown_steps=-1),
    dict(wall_cells=[[x,7] for x in range(16)]),dict(wall_cells=[[7,15]]),
    dict(constraint_block_penalty=float('nan')),
])
def test_invalid_constraints_rejected(extra):
    kwargs={**BASE,**profile_kwargs('control'),**extra}
    with pytest.raises(ValueError):ConstrainedWarehouse(**kwargs)


def test_runtime_restores_helpers_and_existing_registration_after_exception():
    import gymnasium as gym
    before_make,before_mask=T.make_env,T.local_rware_action_masks
    before_probe=T._build_rware_probe_map_bank
    old_spec=copy.deepcopy(gym.spec('rware-multiteam-tiny-4ag-2teams-v0').kwargs)
    before_hashes=preserved_hashes()
    with pytest.raises(RuntimeError):
        with constrained_runtime():raise RuntimeError('simulated failure')
    assert T.make_env is before_make and T.local_rware_action_masks is before_mask
    assert T._build_rware_probe_map_bank is before_probe
    assert gym.spec('rware-multiteam-tiny-4ag-2teams-v0').kwargs==old_spec
    assert preserved_hashes()==before_hashes


def test_canonical_probe_bank_unchanged_from_original_map_across_profiles():
    import torch
    cfg=SimpleNamespace(fixed_canonical_probes=True,probe_source='objective',
        policy_probe_team_conditioning='shared',seed=1,pgct_probe_sequence_length=1,policy_probe_batch_size=48)
    kwargs={**BASE,**profile_kwargs('control')}
    kwargs.pop('wall_cells');kwargs.pop('dock_cooldown_steps')
    with constrained_runtime():
        with T.make_env('rware-multiteam-tiny-4ag-2teams-v0',kwargs,'semantic') as old:
            old.reset(seed=1)
            reference,_=T.sample_policy_probe_sequences(cfg,old,None,np.random.default_rng(7))
        for profile in ['control','detour','bottleneck','dock']:
            with make(profile) as env:
                env.reset(seed=1)
                actual,_=T.sample_policy_probe_sequences(cfg,env,None,np.random.default_rng(7))
                assert torch.equal(reference,actual)
                assert actual.shape==(48,1,1593)


def test_hard_job_branches_match_and_do_not_chain_weights(tmp_path):
    cfg=vars(T.build_arg_parser().parse_args([
        '--observation-format','semantic','--obs-encoder','cnn','--sensor-range','5']))
    cp=dict(config=cfg,agents=[None]*4,effective_env_kwargs=dict(sensor_range=5,n_teams=2))
    jobs=build_hard_jobs(cp,tmp_path/'source.pt',tmp_path/'run',profiles=['control','detour','bottleneck','dock'],
                        modes=['pgct','ippo'],seeds=[1,2],timesteps=4000000)
    assert len(jobs)==16
    assert len({j['config']['init_checkpoint'] for j in jobs})==1
    for job in jobs:
        assert job['effective_timesteps']==4000768
        assert job['config']['env_id']==ENV_ID
        assert job['grid_shape']==[16,16]
        assert len(json.loads(job['config']['env_kwargs_json'])['wall_cells'])=={'control':0,'detour':2,'bottleneck':14,'dock':14}[job['profile']]
        assert job['config']['peer_transfer_mode']==('pgct' if job['mode']=='pgct' else 'none')
    with pytest.raises(ValueError):
        build_hard_jobs(cp,'x',tmp_path,profiles=['dock','dock'],modes=['ippo'],seeds=[1],timesteps=64)
