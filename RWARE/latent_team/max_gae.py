"""Max-GAE (D; D91-D99) for the RWARE trainer. Actor only: critics keep their own returns.

Since the 2026-11 repository split, the environment-independent Max-GAE math (donor rules, donor critics on
the receiver's history, uplift b = clip(max_j V_j(h_i) - V_i(h_i), 0, cap), A_max, the D2 'shrink' blend,
warm-up mask, explained variance) lives in ``pgct_core.max_gae`` (<repo>/PGCT/pgct_core). This module keeps
the RWARE trainer adapter, unchanged:

- ``ContactEnv``: physical contacts within the communication range (Chebyshev) at any step of a rollout,
  and the timeout observations the trainer bootstraps;
- ``install``: scoped monkeypatch of the RWARE trainer (make_env, collect_rollout, update_ippo). With no
  donors the trainer's IPPO update runs untouched; with donors the actor advantages are injected at its
  single compute_gae call. The single v2 gate call per update happens here (rollout contacts).

Donor rules: 'oracle' same team, 'all' anyone, 'allk' as many random agents as the oracle set would give,
'gate' contact & v2 reward approval (no labels), 'oracle_mature' true teammates with both reward predictors
mature. MAMBA-like but without MAMBA's guarantees (D94/D95). See docs/MAX_GAE.md.
Names and behaviour are unchanged; tests/test_pgct_core_equivalence.py compares this module with a frozen
copy of the pre-split implementation (tests/legacy/).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

try:
    from . import _pgct_path  # noqa: F401  (puts <repo>/PGCT on sys.path if pgct_core is not installed)
except ImportError:
    import _pgct_path  # noqa: F401

from examples import train_recurrent_ippo_consensus as T
from pgct_core import max_gae as _core
from pgct_core.max_gae import (DONOR_MODES, SCALES, MaxGAEConfig, donor_values, explained_variance,  # noqa: F401
                               max_targets, select_donors, uplift, warm_mask)


def blended_advantages(*args, gae=None, **kwargs):
    """pgct_core.max_gae.blended_advantages with the RWARE trainer's own compute_gae as the default GAE."""
    return _core.blended_advantages(*args, gae=gae or T.compute_gae, **kwargs)


class ContactEnv(gym.Wrapper):
    """Records who met whom within the communication range during a rollout, and timeout observations."""
    def __init__(self, env, comm_range):
        super().__init__(env)
        self.comm_range = int(comm_range)
        self.begin()

    def begin(self):
        self.contact, self.step_index, self.timeout_obs = None, 0, {}

    def step(self, actions):
        obs, rewards, done, truncated, info = self.env.step(actions)
        pos = np.asarray([(a.x, a.y) for a in self.env.unwrapped.agents])
        met = np.abs(pos[:, None] - pos[None]).max(-1) <= self.comm_range
        self.contact = met if self.contact is None else self.contact | met
        if truncated and not done:  # the trainer bootstraps exactly these steps
            self.timeout_obs[self.step_index] = torch.as_tensor(T.flatten_multi_agent_obs(self, obs),
                                                               dtype=torch.float32)
        self.step_index += 1
        return obs, rewards, done, truncated, info


@contextmanager
def install(cfg, run_dir=None, config=None, *, controller=None):
    """Scoped adapter; the trainer source is unchanged. Wraps make_env (contact log), collect_rollout (last
    observation) and update_ippo (actor advantages injected at its single compute_gae call)."""
    config = config or MaxGAEConfig()
    if config.donors in ("gate", "oracle_mature") and controller is None:
        raise ValueError("gate/oracle_mature Max-GAE requires a learning A+C v2 controller")
    if json.loads(cfg.env_kwargs_json).get("team_reward_mode") != "individual":
        raise ValueError("Max-GAE keeps each agent's individual reward (D82); team/ALL reward is refused")
    originals = {name: getattr(T, name) for name in ("make_env", "collect_rollout", "update_ippo")}
    originals_gae = T.compute_gae
    state = dict(env=None, last_obs=None, update=0, previous=None)
    rng = np.random.default_rng(config.seed)
    lam_f = cfg.gae_lambda if config.gae_lambda is None else config.gae_lambda
    log = None
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "max_gae.json").write_text(json.dumps(dict(
            config=asdict(config), lambda_f=lam_f, value_targets="own returns", contact="any step of the rollout",
            note="MAMBA-like max aggregation without MAMBA's guarantees (D94/D95)"), indent=2), encoding="utf-8")
        log = (Path(run_dir) / "max_gae.jsonl").open("w", encoding="utf-8", buffering=1)

    def make_env(*args, **kwargs):
        env = originals["make_env"](*args, **kwargs)
        return ContactEnv(env, getattr(env.unwrapped, "communication_range", cfg.sensor_range))

    def collect(env, *args, **kwargs):
        if isinstance(env, ContactEnv):
            env.begin()
        rollout, runner_state, metrics = originals["collect_rollout"](env, *args, **kwargs)
        state["env"] = env
        state["last_obs"] = torch.as_tensor(T.flatten_multi_agent_obs(env, runner_state.obs), dtype=torch.float32)
        return rollout, runner_state, metrics

    def update(agents, optimizers, rollout, run_cfg, graph_estimator, device, rng_ippo,
               peer_allocation=None, peer_probe_sequences=None):
        state["update"] += 1
        env, n = state["env"], len(agents)
        held = state["update"] <= config.warmup_updates
        contact = (env.contact if isinstance(env, ContactEnv) and env.contact is not None
                   else np.zeros((n, n), dtype=bool))
        team_ids = np.asarray(env.unwrapped.get_oracle_team_assignments())
        # Refit and compare on the completed rollout's contacts, before PPO changes any weights.
        # The IPPO trainer never activates a peer gate; Max-GAE owns this single controller call.
        approved, mature = None, None
        if controller is not None:
            controller(state["update"], contact, active=not held)
            mature, approved = controller.maturity(), controller.approved()
        _, own_returns = originals_gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values,
                                       run_cfg.gamma, run_cfg.gae_lambda, rollout.timeout_values)
        ev = explained_variance(rollout.values, own_returns)
        critic_mature = (np.ones(n, dtype=bool) if config.min_critic_ev is None else
                         np.isfinite(ev.numpy()) & (ev.numpy() >= config.min_critic_ev))
        donors = np.zeros((n, n), dtype=bool) if held else select_donors(
            config.donors, team_ids, contact, rng, approved=approved, mature=mature, critic_mature=critic_mature)
        call = lambda: originals["update_ippo"](agents, optimizers, rollout, run_cfg, graph_estimator, device,
                                                 rng_ippo, peer_allocation=(np.zeros((n, n), dtype=np.float32)
                                                                           if controller is not None else peer_allocation),
                                                 peer_probe_sequences=peer_probe_sequences)
        same = team_ids[:, None] == team_ids[None, :]
        extra = dict(max_gae_held=float(held), max_gae_contact_arcs=float(np.sum(contact & ~np.eye(n, dtype=bool))),
                     max_gae_donor_arcs=float(donors.sum()), max_gae_same_team_arcs=float(np.sum(donors & same)),
                     max_gae_cross_team_arcs=float(np.sum(donors & ~same)),
                     max_gae_receivers=float(donors.any(1).sum()),
                     max_gae_arc_changes=float(np.sum(donors ^ state["previous"]))
                     if state["previous"] is not None else 0.0)
        state["previous"] = donors
        extra.update({f"max_gae_critic_ev_agent_{k}": float(v) for k, v in enumerate(ev)})
        extra.update({f"max_gae_critic_mature_agent_{k}": float(v) for k, v in enumerate(critic_mature)})
        row = dict(update=state["update"], env_steps=state["update"] * cfg.rollout_steps,
                   held=held, contact=contact.tolist(), donors=donors.tolist(),
                   matrix_convention="receiver_donor", critic_ev=[float(v) for v in ev],
                   critic_mature=critic_mature.tolist(), critic_ev_filter_enabled=config.min_critic_ev is not None)
        if controller is not None:
            eligible_same = same & ~np.eye(n, dtype=bool)
            extra.update(max_gae_reward_mature_agents=float(mature.sum()),
                         max_gae_approved_arcs=float(approved.sum()),
                         max_gae_approved_same_team_arcs=float(np.sum(approved & eligible_same)),
                         max_gae_approved_cross_team_arcs=float(np.sum(approved & ~same)),
                         max_gae_same_team_approval_coverage=float(np.sum(approved & eligible_same)
                                                                 / max(1, eligible_same.sum())))
            extra.update({f"max_gae_reward_mature_agent_{k}": float(v) for k, v in enumerate(mature)})
            row.update(reward_mature=mature.tolist(), approved=approved.tolist(),
                       same_team_possible_arcs=int(eligible_same.sum()),
                       ac_probe_update=controller.last_probe_update, ac_pairs=controller.last_log.get("ac_pairs", []))
        if not donors.any():
            metrics = call()  # bit-identical IPPO update
        else:
            f_values, f_last, f_timeout, active, b, raw = max_targets(
                agents, rollout, donors, env.timeout_obs, state["last_obs"], config, device)
            valid, _ = warm_mask(rollout.dones, config.window)
            actor, returns, adv_self, adv_max = blended_advantages(
                rollout, run_cfg.gamma, run_cfg.gae_lambda, lam_f, config.eta, f_values, f_last, f_timeout, active,
                gae=originals_gae, scale=config.scale, valid=valid)

            def injected(rewards, *args, **kwargs):
                if rewards is not rollout.rewards:
                    raise RuntimeError("Max-GAE advantages were prepared for a different rollout")
                return actor, returns

            T.compute_gae = injected
            try:
                metrics = call()
            finally:
                T.compute_gae = originals_gae
            cols = np.flatnonzero(active)
            scale = adv_self[:, cols].std(0, unbiased=False).clamp_min(1e-8)
            shift = ((actor[:, cols] - adv_self[:, cols]).abs().mean(0) / scale)
            # Effective eta: weight actually given to the (masked) max term, eta * s for 'shrink'.
            delta = (adv_max[:, cols] - adv_self[:, cols]) * valid[:, None]
            eta_eff = ((actor[:, cols] - adv_self[:, cols]).std(0, unbiased=False)
                       / delta.std(0, unbiased=False).clamp_min(1e-8))
            capped = (raw[:, cols] > b[:, cols] + 1e-9).float().mean() if config.cap is not None else torch.tensor(0.)
            extra.update(max_gae_select_frac=float((b[:, cols] > 0).float().mean()),
                         max_gae_mean_b=float(b[:, cols].mean()), max_gae_max_b=float(b[:, cols].max()),
                         max_gae_capped_frac=float(capped), max_gae_adv_shift=float(shift.mean()),
                         max_gae_eta_eff=float(eta_eff.mean()))
            row.update(select_frac=(b > 0).float().mean(0).tolist(), mean_b=b.mean(0).tolist(),
                       adv_shift={int(c): float(s) for c, s in zip(cols, shift)},
                       eta_eff={int(c): float(e) for c, e in zip(cols, eta_eff)})
        metrics = dict(metrics)
        metrics.update(extra)
        if log is not None:
            log.write(json.dumps(dict(row, **{k: v for k, v in extra.items()})) + "\n")
        return metrics

    T.make_env, T.collect_rollout, T.update_ippo = make_env, collect, update
    try:
        yield state
    finally:
        T.make_env, T.collect_rollout, T.update_ippo = (originals["make_env"], originals["collect_rollout"],
                                                        originals["update_ippo"])
        T.compute_gae = originals_gae
        if log is not None:
            log.close()
