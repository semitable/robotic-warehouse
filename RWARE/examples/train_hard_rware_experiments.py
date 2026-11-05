"""Opt-in constrained warehouse branches. Existing runners and files are untouched."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch
from rware.constrained_warehouse import (
    ENV_ID, PROFILES, ConstrainedWarehouse, ConstrainedSemanticObservation,
    profile_kwargs, register_constrained_environment,
)
from rware.warehouse import Action

try:
    from . import train_recurrent_ippo_consensus as T
    from .train_pgct_spatial_experiments import build_jobs, experiment_logging, summarize_run, DEFAULT_CHECKPOINT
except ImportError:
    import train_recurrent_ippo_consensus as T
    from train_pgct_spatial_experiments import build_jobs, experiment_logging, summarize_run, DEFAULT_CHECKPOINT

PRESERVED = ['rware/warehouse.py', 'rware/multi_team_warehouse.py', 'rware/__init__.py',
             'rware/utils/semantic_observation.py', 'rware/utils/probe_maps.py', 'rware/rendering.py',
             'examples/train_recurrent_ippo_consensus.py', 'examples/train_pgct_spatial_experiments.py',
             'examples/train_pgct_transfer_experiments.py', 'examples/run_pgct_spatial_experiments.ps1']


def preserved_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PRESERVED}


@contextmanager
def constrained_runtime():
    """Only this process and only the new env ID use the extended wrapper/mask."""
    register_constrained_environment()
    make, masks, probe = T.make_env, T.local_rware_action_masks, T._build_rware_probe_map_bank

    def constrained_make(env_id, env_kwargs, observation_format):
        if env_id != ENV_ID:
            return make(env_id, env_kwargs, observation_format)
        if observation_format != 'semantic':
            raise ValueError('The constrained runner requires semantic observations')
        return ConstrainedSemanticObservation(gym.make(env_id, **env_kwargs))

    def constrained_masks(env, n_agents, action_dim, enabled):
        result = masks(env, n_agents, action_dim, enabled)
        if enabled and isinstance(env.unwrapped, ConstrainedWarehouse):
            for i, agent in enumerate(env.unwrapped.agents[:n_agents]):
                if env.unwrapped.forward_constraint(agent):
                    result[i, Action.FORWARD.value] = 0
        return result

    def canonical_probe(env, *args, **kwargs):
        if not isinstance(env.unwrapped, ConstrainedWarehouse):
            return probe(env, *args, **kwargs)
        # The original builder constructs an ordinary MultiTeamWarehouse from
        # source spec kwargs. Remove only extension-only constructor fields on
        # this temporary view, retaining the original canonical probe contents.
        class ProbeSource:
            def __init__(self):
                self.unwrapped = env.unwrapped
                self.spec = copy.deepcopy(env.spec)
                for name in ('wall_cells','dock_cooldown_steps','constraint_block_penalty'):
                    self.spec.kwargs.pop(name,None)
            def __getattr__(self, name):
                return getattr(env,name)
        return probe(ProbeSource(), *args, **kwargs)

    T.make_env, T.local_rware_action_masks, T._build_rware_probe_map_bank = constrained_make, constrained_masks, canonical_probe
    try:
        yield
    finally:
        T.make_env, T.local_rware_action_masks, T._build_rware_probe_map_bank = make, masks, probe


def build_hard_jobs(checkpoint, source, output, *, profiles, modes, seeds, timesteps,
                    requests=4, track=False, save_video=False, smoke=False):
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in PROFILES for p in profiles):
        raise ValueError('Choose unique, nonempty constrained profiles')
    jobs = []
    template = build_jobs(checkpoint, source, output, maps=['e2'], modes=modes, seeds=seeds,
                          timesteps=timesteps, requests=requests, track=track,
                          save_video=save_video, smoke=smoke)
    for profile in profiles:
        for original in template:
            job = json.loads(json.dumps(original))
            cfg = job['config']
            name = f'{profile}_{job["mode"]}_s{job["seed"]}'
            kwargs = json.loads(cfg['env_kwargs_json'])
            kwargs.update(profile_kwargs(profile), render_mode='rgb_array', constraint_block_penalty=.005)
            cfg.update(env_id=ENV_ID, exp_name=name, wandb_name=name, env_kwargs_json=json.dumps(kwargs))
            job.update(name=name, profile=profile, map=profile,
                       output_checkpoint=str(Path(output)/name/'final.pt'))
            jobs.append(job)
    return jobs


def save_previews(output, job_configs=None):
    from PIL import Image, ImageDraw
    reports, pictures = {}, []
    with constrained_runtime():
        for profile in PROFILES:
            if job_configs:
                source = next(iter(job_configs))['config']
                kwargs = json.loads(source['env_kwargs_json'])
            else:
                kwargs = json.loads((ROOT/'examples/env_kwargs_stage2_pgct_success_unblock_zones_individual.json').read_text())
                kwargs.update(sensor_range=5, n_agents=4, n_teams=2, request_queue_size=4,
                              request_queue_size_per_team=2, shelf_team_mode='balanced_soft_zones',
                              shelf_soft_zone_ratio=.5, normalised_coordinates=True)
            kwargs.update(profile_kwargs(profile), render_mode='rgb_array')
            with T.make_env(ENV_ID, kwargs, 'semantic') as env:
                env.reset(seed=5000001)
                reports[profile] = env.unwrapped.routing_report()
                reports[profile]['dock_cooldown_steps'] = env.unwrapped.dock_cooldown_steps
                picture = Image.fromarray(env.render())
                pictures.append((profile, picture))
                picture.save(output/f'{profile}.png')
    width = sum(p.width for _,p in pictures)
    canvas = Image.new('RGB', (width, max(p.height for _,p in pictures)+30), 'white')
    draw = ImageDraw.Draw(canvas)
    x = 0
    for profile, picture in pictures:
        draw.text((x+8, 8), profile, fill='black')
        canvas.paste(picture, (x,30));x += picture.width
    canvas.save(output/'maps.png')
    (output/'layouts.json').write_text(json.dumps(reports, indent=2), encoding='utf8')
    return reports


@contextmanager
def hard_logging(cfg, run_dir, eval_seed):
    """Reuse existing spatial/evaluation logs; add initial evaluation and constraint deltas."""
    with experiment_logging(cfg, run_dir, eval_seed):
        collect = T.collect_rollout
        iteration = 0
        with (run_dir/'constraints.jsonl').open('w', encoding='utf8', buffering=1) as stream:
            def collect_with_constraints(env, agents, *args, **kwargs):
                nonlocal iteration
                if iteration == 0:
                    initial, _ = T.evaluate(cfg.env_id, json.loads(cfg.env_kwargs_json), cfg.observation_format,
                        agents, cfg.eval_episodes, cfg.eval_horizon, cfg.gamma,
                        env.observation_space[0].shape[0], cfg.recurrent_hidden_dim,
                        next(agents[0].parameters()).device, seed=eval_seed, use_action_mask=cfg.action_mask)
                    (run_dir/'initial_eval_metrics.json').write_text(json.dumps(initial, indent=2), encoding='utf8')
                    T.log_to_wandb(None, T.eval_payload_from_metrics(initial), step=0)
                before = dict(env.unwrapped.constraint_totals)
                result = collect(env, agents, *args, **kwargs)
                iteration += 1
                after = env.unwrapped.constraint_totals
                stream.write(json.dumps(dict(env_steps=iteration*cfg.rollout_steps,
                    counts={k: after[k]-before[k] for k in before}))+'\n')
                return result
            T.collect_rollout = collect_with_constraints
            try:
                yield
            finally:
                T.collect_rollout = collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--profiles', choices=PROFILES, nargs='+', default=['detour'])
    parser.add_argument('--modes', choices=['pgct','ippo'], nargs='+', default=['pgct','ippo'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--timesteps', type=int, default=4000000)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--eval-seed', type=int, default=5000001)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('WANDB_DISABLE_GIT', 'true')
    torch.set_num_threads(1)
    prefix = 'output/hard_preview_' if args.preview_only else 'runs/hard_'
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        parser.error('Output already exists; choose a new directory to preserve prior results')
    hashes = preserved_hashes()
    if args.preview_only:
        output.mkdir(parents=True)
        save_previews(output)
        assert hashes == preserved_hashes()
        print(f'Preview: {output / "maps.png"}', flush=True)
        return
    checkpoint = T._load_checkpoint(args.init_checkpoint, torch.device('cpu'))
    try:
        jobs = build_hard_jobs(checkpoint, args.init_checkpoint, output, profiles=args.profiles,
                              modes=args.modes, seeds=args.seeds, timesteps=args.timesteps,
                              requests=args.requests, track=args.track, save_video=args.save_video,
                              smoke=args.smoke_test)
    except ValueError as error:
        parser.error(str(error))
    del checkpoint
    output.mkdir(parents=True)
    layouts = save_previews(output, jobs)
    manifest = dict(initial_checkpoint=str(Path(args.init_checkpoint).resolve()),
                    note='Matched pretrained branches. Original PPO, PGCT probes/gates and reward coefficients retained. '
                         'Solid walls and dock entry constraints are opt-in; this is not evidence that PGCT helps.',
                    eval_seed=args.eval_seed, smoke_test=args.smoke_test, layouts=layouts,
                    preserved_source_hashes=hashes, jobs=jobs)
    def save():
        (output/'experiment.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    save()
    if args.dry_run:
        print(f'Plan only: {output / "experiment.json"}', flush=True)
        return
    try:
        with constrained_runtime():
            for job in jobs:
                cfg = T.TrainConfig(**job['config'])
                run_dir = output/job['name']
                job['status'] = 'running';save()
                print(f'Starting {job["name"]}: {job["effective_timesteps"]} additional steps', flush=True)
                try:
                    with hard_logging(cfg, run_dir, args.eval_seed):
                        T.train(cfg)
                    if not Path(job['output_checkpoint']).is_file():
                        raise RuntimeError('Training did not produce final.pt')
                    job['summary'] = summarize_run(run_dir)
                    totals = {}
                    for line in (run_dir/'constraints.jsonl').read_text().splitlines():
                        for k,v in json.loads(line)['counts'].items():totals[k]=totals.get(k,0)+v
                    job['constraint_totals'] = totals
                    job['status'] = 'complete';save()
                except BaseException as error:
                    job.update(status='failed', error=str(error));save();raise
    finally:
        manifest['preserved_sources_unchanged'] = hashes == preserved_hashes()
        save()
    assert manifest['preserved_sources_unchanged']
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
