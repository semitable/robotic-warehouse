"""A+C gate v2 for RWARE: personal reward predictors queried on shared reward-event probes.

Since the 2026-11 repository split, the environment-independent algorithm (predictor, probe set
Q_ij = E_i u E_j, pooled / action_max response distance, EMA, maturity, approval, rollout-start snapshot)
lives in ``pgct_core.reward_gate`` (<repo>/PGCT/pgct_core). This module keeps the RWARE-specific parts:

- the public RWARE probe view (reward_probe_ac.public_view: loaded/direction/highway ego bits and the
  public physical channels of the semantic observation within ``radius``), 5 actions, uint8 storage;
- the left-right mirror of RWARE views/actions (used only when ``mirror`` is enabled);
- ``ACV2Config``: identical fields to the core config; for backward compatibility its default keeps
  ``mirror=True`` as before the split. The PGCT-D2 main runs pass ``mirror: false``
  (latent_team/acv2_nomirror.json), i.e. Q_ij = E_i u E_j;
- the coordinate-free trainer wrapper and the scoped trainer adapter ``install``.

Names, signatures, defaults, RNG use and checkpoint payloads are unchanged; tests/test_pgct_core_equivalence.py
compares this module with a frozen copy of the pre-split implementation (tests/legacy/).
Team labels never enter a decision. The install() logger audits same/cross decisions afterwards from the
environment info, for W&B only.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import _pgct_path  # noqa: F401  (puts <repo>/PGCT on sys.path if pgct_core is not installed)
    from . import reward_probe_ac as AC
except ImportError:
    import _pgct_path  # noqa: F401
    import reward_probe_ac as AC

from pgct_core import reward_gate as _core
from pgct_core.reward_gate import DISTANCE_MODES, Ring, action_max_distance, audit, response_distance  # noqa: F401

EGO = 6          # public ego: carrying, up, down, left, right, highway
N_ACTIONS = 5    # NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD
LEFT, RIGHT = 2, 3
DIR_LEFT, DIR_RIGHT = 7, 8  # public channel positions of the visible agents' left/right heading


@dataclass(frozen=True)
class ACV2Config(_core.RewardGateConfig):
    """RWARE gate options (same fields and order as pgct_core.RewardGateConfig); legacy default mirror=True."""
    mirror: bool = True


def view_size(radius):
    return EGO + len(AC.PUBLIC_CHANNELS) * (2 * radius + 1) ** 2


def mirror_view(views, radius):
    """Left-right mirror of public views [..., V]: a feasible variant of the same local event."""
    views = np.array(views, copy=True)
    side = 2 * radius + 1
    ego = views[..., :EGO].copy()
    ego[..., [3, 4]] = ego[..., [4, 3]]
    grid = views[..., EGO:].reshape(views.shape[:-1] + (len(AC.PUBLIC_CHANNELS), side, side))[..., ::-1].copy()
    grid[..., [DIR_LEFT, DIR_RIGHT], :, :] = grid[..., [DIR_RIGHT, DIR_LEFT], :, :]
    views[..., :EGO] = ego
    views[..., EGO:] = grid.reshape(views.shape[:-1] + (-1,))
    return views


def mirror_action(actions):
    actions = np.asarray(actions)
    return np.where(actions == LEFT, RIGHT, np.where(actions == RIGHT, LEFT, actions))


def features(before, actions, after):
    return _core.features(before, actions, after, N_ACTIONS)


class RwareProbeProjection:
    """pgct_core.interfaces.ProbeProjection for RWARE semantic observations (coordinates already masked)."""
    view_dtype = np.uint8
    n_actions = N_ACTIONS

    def __init__(self, radius):
        self.radius = int(radius)
        self.view_size = view_size(self.radius)

    def public_view(self, observation):
        return AC.public_view(observation, self.radius)

    def mirror(self, before, actions, after):
        return mirror_view(before, self.radius), mirror_action(actions), mirror_view(after, self.radius)


class RewardPredictor(_core.RewardPredictor):
    """Legacy signature RewardPredictor(config, seed) for RWARE views."""
    def __init__(self, config, seed):
        super().__init__(config, seed, view_size(config.radius), N_ACTIONS, np.uint8)


class ACV2Controller(_core.RewardProbeGate):
    """Legacy signature ACV2Controller(n_agents, config=None) bound to the RWARE projection."""
    def __init__(self, n_agents, config=None):
        config = config or ACV2Config()
        super().__init__(n_agents, config, RwareProbeProjection(config.radius))


class AuditedObservationEnv(AC.LocalObservationEnv):
    """v1's coordinate-free wrapper; also keeps the latest team ids from info for the audit log only."""
    def __init__(self, env, controller=None):
        super().__init__(env, controller)
        self.audit_team_ids = None

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        if isinstance(info, dict) and info.get("agent_team_ids") is not None:
            self.audit_team_ids = np.asarray(info["agent_team_ids"]).copy()
        after = AC.strip_coordinates(obs)
        if self.controller is not None and self.controller.active:
            self.controller.observe(self.previous, actions, after, rewards)
        self.previous = after
        return tuple(after.copy()), rewards, terminated, truncated, info


@contextmanager
def install(cfg, run_dir=None, config=None, *, learn=True):
    """Scoped adapter like reward_probe_ac.install; adds update/ac_v2_* metrics for W&B."""
    from examples import train_recurrent_ippo_consensus as T

    config = config or ACV2Config()
    env_kwargs = json.loads(cfg.env_kwargs_json)
    if env_kwargs.get("team_reward_mode") != "individual":
        raise ValueError("A+C requires explicit individual rewards; team/ALL is forbidden")
    if cfg.observation_format != "semantic" or cfg.sensor_range < config.radius:
        raise ValueError("A+C v2 needs semantic observations covering the probe radius")
    originals = {name: getattr(T, name) for name in
                 ("make_env", "collect_rollout", "sample_policy_probe_sequences", "save_checkpoint", "update_ippo")}
    controller = ACV2Controller(cfg.agent_count or env_kwargs["n_agents"], config) if learn else None
    envs = []

    def make_env(*args, **kwargs):
        env = AuditedObservationEnv(originals["make_env"](*args, **kwargs), controller)
        envs.append(env)
        return env

    def collect(*args, **kwargs):
        if controller is not None:
            if controller.config.contact_snapshot:
                controller.snapshot()  # before this rollout's first step (and before any contact in it)
            controller.active = True
        try:
            return originals["collect_rollout"](*args, **kwargs)
        finally:
            if controller is not None:
                controller.active = False

    def sample(*args, **kwargs):
        probes, metadata = originals["sample_policy_probe_sequences"](*args, **kwargs)
        probes = probes.clone()
        probes[..., :2] = 0
        return probes, metadata

    def update(*args, **kwargs):
        metrics = originals["update_ippo"](*args, **kwargs)
        if controller is not None and isinstance(metrics, dict):
            metrics.update(controller.last_metrics)
            teams = next((e.audit_team_ids for e in reversed(envs) if e.audit_team_ids is not None), None)
            if teams is not None:
                metrics.update(audit(controller, teams))
        return metrics

    def save(path, *args, **kwargs):
        originals["save_checkpoint"](path, *args, **kwargs)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["ac_observation"] = "self_x/self_y masked to zero; input dimension retained"
        if controller is not None:
            payload["reward_probe_ac_v2"] = controller.state_dict()
        torch.save(payload, path)

    T.make_env, T.collect_rollout, T.sample_policy_probe_sequences = make_env, collect, sample
    T.save_checkpoint, T.update_ippo = save, update
    try:
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "reward_probe_ac_v2.json").write_text(json.dumps(dict(
                config=asdict(config), learning=learn, reward="individual", coordinates="masked",
                predictor="mlp_own_reward", probes="own reward events + mirror",
                comparison=("rollout-start snapshot, compared before refit" if config.contact_snapshot
                            else "refit, then compare"),
                distance="sum|a-b|/sum(|a|+|b|)", unsupported="abstain"), indent=2), encoding="utf-8")
        yield controller
    finally:
        for name, original in originals.items():
            setattr(T, name, original)
