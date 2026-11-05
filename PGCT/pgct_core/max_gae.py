"""Max-aggregated advantage from approved teammates' critics (D2; D91-D99), environment independent.

Extracted without algorithmic change from RWARE ``latent_team/max_gae.py`` (commit 122615f). The RWARE
module keeps its trainer adapter (contact wrapper, monkeypatched update) and imports these functions.

For receiver i with donor set G_i (fixed for one update), each donor critic j runs recurrently over i's own
rollout critic inputs. With b = clip(max_j V_j(h_i) - V_i(h_i), 0, cap) and f = V_i + b, A_max is
GAE(lambda_f) of r + gamma f(s') - f(s); the actor advantage is A_self + eta (A_max - A_self). Value targets
stay the receiver's own GAE returns. 'shrink' (D98) scales the difference per receiver by
s = min(1, std(A_self) / std(A_max - A_self)). Donor hidden states start cold at the rollout boundary; for
the first ``window - 1`` steps (before any reset) the actor advantage stays A_self.

Requirements on the networks: ``initial_hidden(batch, device) -> (actor_h, critic_h)`` and
``_critic_step(obs, critic_h) -> (value, next_h)`` (pgct_core.networks.RecurrentActorCritic and the RWARE
trainer's network both provide them). Donor and receiver critic inputs must share one space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

DONOR_MODES = ("oracle", "all", "allk", "gate", "oracle_mature")
SCALES = ("raw", "shrink", "relative")


@dataclass(frozen=True)
class MaxGAEConfig:
    donors: str = "oracle"
    eta: float = .5                      # 1.0 with scale 'raw' = D1
    scale: str = "raw"                   # raw | shrink (eta_eff <= eta) | relative (record only; amplifies small b)
    gae_lambda: Optional[float] = None   # None: the trainer's lambda
    cap: Optional[float] = None          # upper bound on b; None = no cap
    window: int = 8                      # donor warm-up steps after a cold start at the rollout boundary
    warmup_updates: int = 0              # first updates without donors (update <= warmup)
    seed: int = 0                        # 'allk' sampling stream
    min_critic_ev: Optional[float] = None  # optional donor-only EV floor; no research threshold chosen yet

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
        if self.min_critic_ev is not None and not np.isfinite(self.min_critic_ev):
            raise ValueError("min_critic_ev must be finite")


def compute_gae(rewards, dones, values, last_values, gamma, gae_lambda, timeout_values=None):
    """GAE identical to the RWARE trainer's ``compute_gae`` (shared done vector, timeout bootstrap)."""
    rollout_steps, n_agents = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((n_agents,), dtype=torch.float32)
    for step in reversed(range(rollout_steps)):
        if step == rollout_steps - 1:
            next_value = last_values
        else:
            next_value = values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        if timeout_values is not None:
            delta = delta + gamma * timeout_values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def select_donors(mode, team_ids, contact, rng, *, approved=None, mature=None, critic_mature=None):
    """receiver x donor; met agents only, never oneself. Only 'oracle'/'allk'/'oracle_mature' read team_ids."""
    team_ids = np.asarray(team_ids)
    met = np.asarray(contact, dtype=bool).copy()
    np.fill_diagonal(met, False)
    if mode not in DONOR_MODES:
        raise ValueError(f"Unknown donor mode: {mode}")
    if mode == "gate":  # no ground-truth labels participate in this decision
        if approved is None or np.shape(approved) != met.shape:
            raise ValueError("gate donors require an approved matrix matching contact")
        donors = met & np.asarray(approved, dtype=bool)
    elif mode == "oracle_mature":
        if mature is None or np.shape(mature) != (len(met),):
            raise ValueError("oracle_mature donors require per-agent reward maturity")
        mature = np.asarray(mature, dtype=bool)
        donors = met & (team_ids[:, None] == team_ids[None, :]) & mature[:, None] & mature[None, :]
    elif mode == "all":
        donors = met
    else:
        oracle = met & (team_ids[:, None] == team_ids[None, :])
        if mode == "oracle":
            donors = oracle
        else:
            donors = np.zeros_like(met)
            for i in range(len(team_ids)):
                k = int(oracle[i].sum())
                if k:
                    donors[i, rng.choice(np.flatnonzero(met[i]), size=k, replace=False)] = True
    if critic_mature is not None:
        if np.shape(critic_mature) != (len(met),):
            raise ValueError("critic maturity must have one flag per donor")
        donors &= np.asarray(critic_mature, dtype=bool)[None, :]
    return donors


def shuffled_gate_donors(contact, approved, rng):
    """Random-gate control without labels: each receiver keeps the number of donors the PGCT gate would
    give it this update (contact & approved), drawn uniformly among the agents it met."""
    met = np.asarray(contact, dtype=bool).copy()
    np.fill_diagonal(met, False)
    gate = met & np.asarray(approved, dtype=bool)
    donors = np.zeros_like(met)
    for i in range(len(met)):
        k = int(gate[i].sum())
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
    gae = gae or compute_gae
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


def advantage_diagnostics(actor, adv_self, adv_max, b, raw, active, valid, cap):
    """The max_gae_* statistics logged by both the RWARE adapter and the generic trainer."""
    cols = np.flatnonzero(active)
    scale = adv_self[:, cols].std(0, unbiased=False).clamp_min(1e-8)
    shift = ((actor[:, cols] - adv_self[:, cols]).abs().mean(0) / scale)
    delta = (adv_max[:, cols] - adv_self[:, cols]) * valid[:, None]
    eta_eff = ((actor[:, cols] - adv_self[:, cols]).std(0, unbiased=False)
               / delta.std(0, unbiased=False).clamp_min(1e-8))
    capped = (raw[:, cols] > b[:, cols] + 1e-9).float().mean() if cap is not None else torch.tensor(0.)
    extra = dict(max_gae_select_frac=float((b[:, cols] > 0).float().mean()),
                 max_gae_mean_b=float(b[:, cols].mean()), max_gae_max_b=float(b[:, cols].max()),
                 max_gae_capped_frac=float(capped), max_gae_adv_shift=float(shift.mean()),
                 max_gae_eta_eff=float(eta_eff.mean()))
    row = dict(select_frac=(b > 0).float().mean(0).tolist(), mean_b=b.mean(0).tolist(),
               adv_shift={int(c): float(s) for c, s in zip(cols, shift)},
               eta_eff={int(c): float(e) for c, e in zip(cols, eta_eff)})
    return extra, row
