"""Max-aggregated advantage from teammates' critics (D; D91-D95). Actor only: critics keep their own returns.

For receiver i with donor set G_i (fixed for one update), each donor critic j is run recurrently over i's own
rollout observations (critic weights are shared on contact; no probe bank). With
    b = clip(max_j V_j(h_i) - V_i(h_i), 0, cap),   f = V_i + b,
A_max is GAE(lambda_f) of r + gamma f(s') - f(s), and the actor advantage is A_self + eta (A_max - A_self)
('raw'). Since GAE is linear in the value function, with lambda_f = lambda this equals GAE(V + eta min(b, cap)).
Value targets stay the receiver's own GAE returns. eta = 1 with 'raw' is D1 (MAMBA-style).
'shrink' (D98) multiplies the difference per receiver by s = min(1, std(A_self) / std(A_max - A_self)), i.e.
f = V + eta s min(b, cap) with eta s <= eta: in a 1M branch the raw difference was 3-12x std(A_self) for
laggards, which made raw D2 and D1 nearly identical after PPO's per-agent normalisation, while a small b stays
small and b -> 0 returns to IPPO. 'relative' (s without the min, kept only for the record) inflates a near-zero
b to the receiver's own advantage scale (E66).
Donor hidden states start cold at the rollout boundary; on the first `window - 1` steps (before any reset) the
uplift is unknown, so the actor advantage there is A_self (zeroing only b would put a spurious positive step
into the GAE of those steps; E66).
A receiver without donors gets the untouched IPPO update; with no donors at all the trainer runs unchanged.
Every update also logs each critic's explained variance on its own rollout (donor maturity is not yet gated).

Donors: 'oracle' same team, 'all' anyone, 'allk' as many random agents as the oracle set would give (control
for the max over more noisy critics). Candidates are agents met inside the communication range (Chebyshev)
at any step of the rollout: weights are fixed during a rollout, so one contact suffices to share them.
MAMBA-like but without MAMBA's guarantees (D94/D95): donor critics are extrapolated to the receiver's
histories instead of being re-estimated by oracle roll-outs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from examples import train_recurrent_ippo_consensus as T

DONOR_MODES = ("oracle", "all", "allk")
SCALES = ("raw", "shrink", "relative")


@dataclass(frozen=True)
class MaxGAEConfig:
    donors: str = "oracle"
    eta: float = .5                      # 1.0 with scale 'raw' = D1
    scale: str = "raw"                   # raw | shrink (eta_eff <= eta) | relative (record only; amplifies small b)
    gae_lambda: Optional[float] = None   # None: the trainer's lambda
    cap: Optional[float] = None          # upper bound on b; None = no cap
    window: int = 8                      # donor warm-up steps after a cold start at the rollout boundary
    warmup_updates: int = 0              # first updates without donors (matched_transfer's rule: update <= warmup)
    seed: int = 0                        # 'allk' sampling stream

    def __post_init__(self):
        if self.donors not in DONOR_MODES:
            raise ValueError(f"donors must be one of {DONOR_MODES}")
        if not 0 < self.eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if self.gae_lambda is not None and not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.window < 1 or self.warmup_updates < 0:
            raise ValueError("window must be >= 1 and warmup_updates >= 0")


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


def select_donors(mode, team_ids, contact, rng):
    """receiver x donor; met agents only, never oneself."""
    team_ids = np.asarray(team_ids)
    met = np.asarray(contact, dtype=bool).copy()
    np.fill_diagonal(met, False)
    oracle = met & (team_ids[:, None] == team_ids[None, :])
    if mode == "oracle":
        return oracle
    if mode == "all":
        return met
    donors = np.zeros_like(met)
    for i in range(len(team_ids)):
        k = int(oracle[i].sum())
        if k:
            donors[i, rng.choice(np.flatnonzero(met[i]), size=k, replace=False)] = True
    return donors


@torch.no_grad()
def donor_values(net, obs, dones, timeout_obs, last_obs, device):
    """Critic `net` over receivers' own streams obs [T, B, D] from a cold start, reset after each done like the
    trainer's own critic. timeout_obs {t: [B, D]} are scored with the post-step hidden state before the reset."""
    steps, batch = obs.shape[:2]
    _, h = net.initial_hidden(batch, device)
    obs = obs.to(device)
    values, timeout = torch.zeros(steps, batch), torch.zeros(steps, batch)
    for t in range(steps):
        v, h = net._critic_step(obs[t], h)
        values[t] = v.reshape(batch).cpu()
        if t in timeout_obs:
            v_timeout, _ = net._critic_step(timeout_obs[t].to(device), h)
            timeout[t] = v_timeout.reshape(batch).cpu()
        if dones[t] > 0:
            h = torch.zeros_like(h)
    v_last, _ = net._critic_step(last_obs.to(device), h)
    return values, timeout, v_last.reshape(batch).cpu()


def warm_mask(dones, window):
    """Steps where a cold-started donor hidden state is usable: `window` observations seen, or after a reset
    (both the receiver's and the donor's hidden states are zeroed there)."""
    steps = len(dones)
    valid = torch.arange(steps) >= window - 1
    done_at = torch.nonzero(torch.as_tensor(dones) > 0).flatten()
    if len(done_at):
        valid |= torch.arange(steps) > int(done_at[0])
    last_valid = bool(valid[-1]) or bool(dones[-1] > 0)
    return valid.float(), last_valid


def uplift(own, best, cap):
    b = torch.clamp(best - own, min=0.0)
    return b if cap is None else torch.clamp(b, max=float(cap))


def blended_advantages(rollout, gamma, lam, lam_f, eta, f_values, f_last, f_timeout, active, gae=None,
                       scale="raw", valid=None):
    """(actor advantages, own returns, A_self, A_max). Inactive receivers keep the plain IPPO advantages.
    valid [T]: steps where the uplift is known; elsewhere the actor advantage stays A_self."""
    gae = gae or T.compute_gae
    adv_self, returns = gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values, gamma, lam,
                            rollout.timeout_values)
    adv_max, _ = gae(rollout.rewards, rollout.dones, f_values, f_last, gamma, lam_f, f_timeout)
    actor = adv_self.clone()
    cols = torch.as_tensor(np.flatnonzero(active), dtype=torch.long)
    delta = adv_max[:, cols] - adv_self[:, cols]
    if valid is not None:
        delta = delta * torch.as_tensor(valid, dtype=delta.dtype)[:, None]
    if scale in ("shrink", "relative"):
        ratio = adv_self[:, cols].std(0, unbiased=False) / delta.std(0, unbiased=False).clamp_min(1e-8)
        delta = delta * (ratio.clamp(max=1.0) if scale == "shrink" else ratio)
    actor[:, cols] = adv_self[:, cols] + eta * delta
    return actor, returns, adv_self, adv_max


def explained_variance(values, returns):
    """Per column 1 - Var(R - V) / Var(R) on one rollout; nan where the returns are constant."""
    var = returns.var(0, unbiased=False)
    ev = 1.0 - (returns - values).var(0, unbiased=False) / var.clamp_min(1e-12)
    return torch.where(var > 1e-12, ev, torch.full_like(ev, float("nan")))


def max_targets(agents, rollout, donors, timeout_obs, last_obs, config, device):
    """f along every active receiver's rollout plus diagnostics. donors: receiver x donor bool."""
    steps, n = rollout.values.shape
    valid, last_valid = warm_mask(rollout.dones, config.window)
    best = torch.full((steps, n), -np.inf)
    best_timeout = torch.full((steps, n), -np.inf)
    best_last = torch.full((n,), -np.inf)
    rows = {t: o for t, o in timeout_obs.items() if t < steps}
    for j in range(n):
        recv = np.flatnonzero(donors[:, j])
        if not len(recv):
            continue
        idx = torch.as_tensor(recv, dtype=torch.long)
        v, v_timeout, v_last = donor_values(agents[j], rollout.obs[:, idx], rollout.dones,
                                            {t: o[idx] for t, o in rows.items()}, last_obs[idx], device)
        best[:, idx] = torch.maximum(best[:, idx], v)
        best_timeout[:, idx] = torch.maximum(best_timeout[:, idx], v_timeout)
        best_last[idx] = torch.maximum(best_last[idx], v_last)
    active = donors.any(1)
    f_values, f_last = rollout.values.clone(), rollout.last_values.clone()
    timeout_values = (rollout.timeout_values if rollout.timeout_values is not None
                      else torch.zeros_like(rollout.values))
    f_timeout = timeout_values.clone()
    is_timeout = torch.zeros(steps)
    if rows:
        is_timeout[list(rows)] = 1.0
    b = torch.zeros(steps, n)
    for i in np.flatnonzero(active):
        b[:, i] = uplift(rollout.values[:, i], best[:, i], config.cap) * valid
        f_values[:, i] += b[:, i]
        b_timeout = uplift(timeout_values[:, i], best_timeout[:, i], config.cap) * valid * is_timeout
        f_timeout[:, i] += b_timeout
        if last_valid:
            f_last[i] += uplift(rollout.last_values[i], best_last[i], config.cap)
    raw = torch.clamp(best - rollout.values, min=0.0) * valid[:, None]
    return f_values, f_last, f_timeout, active, b, raw


@contextmanager
def install(cfg, run_dir=None, config=None):
    """Scoped adapter; the trainer source is unchanged. Wraps make_env (contact log), collect_rollout (last
    observation) and update_ippo (actor advantages injected at its single compute_gae call)."""
    config = config or MaxGAEConfig()
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
        donors = np.zeros((n, n), dtype=bool) if held else select_donors(config.donors, team_ids, contact, rng)
        call = lambda: originals["update_ippo"](agents, optimizers, rollout, run_cfg, graph_estimator, device,
                                                 rng_ippo, peer_allocation=peer_allocation,
                                                 peer_probe_sequences=peer_probe_sequences)
        same = team_ids[:, None] == team_ids[None, :]
        extra = dict(max_gae_held=float(held), max_gae_contact_arcs=float(np.sum(contact & ~np.eye(n, dtype=bool))),
                     max_gae_donor_arcs=float(donors.sum()), max_gae_same_team_arcs=float(np.sum(donors & same)),
                     max_gae_cross_team_arcs=float(np.sum(donors & ~same)),
                     max_gae_receivers=float(donors.any(1).sum()),
                     max_gae_arc_changes=float(np.sum(donors ^ state["previous"]))
                     if state["previous"] is not None else 0.0)
        state["previous"] = donors
        # Each critic on its own rollout against its own value targets (pre-update weights; pure computation).
        _, own_returns = originals_gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values,
                                       run_cfg.gamma, run_cfg.gae_lambda, rollout.timeout_values)
        ev = explained_variance(rollout.values, own_returns)
        extra.update({f"max_gae_critic_ev_agent_{k}": float(v) for k, v in enumerate(ev)})
        row = dict(update=state["update"], held=held, contact=contact.tolist(), donors=donors.tolist(),
                   critic_ev=[float(v) for v in ev])
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
