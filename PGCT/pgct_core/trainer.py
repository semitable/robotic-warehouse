"""Generic recurrent IPPO trainer with the PGCT-D2 hooks, for environments behind an EnvAdapter.

RWARE keeps its own (frozen) trainer and plugs the same pgct_core components into it. New environments
(Overcooked) use this loop. The method only changes the donor set of Max-GAE; environment, PPO settings,
budget, seeds and evaluation are identical across methods.

  ippo          no critic access
  oracle        Max-GAE donors = met agents of the same TRUE team          (reads labels: Oracle baseline)
  all           Max-GAE donors = every met agent
  pgct_d2       Max-GAE donors = met agents approved by the reward-probe gate (no labels)
  pgct_id_only  the gate runs and is logged, but no critic access (identification without transfer)
  random_gate   the gate runs; each receiver gets as many donors as pgct_d2 would, drawn at random among
                the agents it met (shuffled-gate control, no labels)

Per update: (1) the gate snapshots each agent's sendable state at the rollout start (contact_snapshot),
(2) a rollout of individual rewards is collected while the gate's predictors see only their own public
transitions, (3) the gate compares rollout-contact pairs and refits, (4) donors are selected, (5) Max-GAE
(if any donor) replaces the actor advantages, critics keep own returns, (6) PPO.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .max_gae import (MaxGAEConfig, advantage_diagnostics, blended_advantages, compute_gae, explained_variance,
                      max_targets, select_donors, shuffled_gate_donors, warm_mask)
from .metrics import same_team_matrix
from .networks import RecurrentActorCritic
from .ppo import PPOConfig, Rollout, ppo_update
from .reward_gate import RewardGateConfig, RewardProbeGate, audit

METHODS = {
    "ippo": dict(gate=False, donors=None),
    "oracle": dict(gate=False, donors="oracle"),
    "all": dict(gate=False, donors="all"),
    "pgct_d2": dict(gate=True, donors="gate"),
    "pgct_id_only": dict(gate=True, donors=None),
    "random_gate": dict(gate=True, donors="shuffled"),
}
LABEL_METHODS = {"oracle"}  # the only training methods allowed to read ground-truth teams


@dataclass
class TrainerConfig:
    method: str = "pgct_d2"
    seed: int = 1
    total_timesteps: int = 5_000_000
    device: str = "cpu"
    warmup_updates: int = 100     # Max-GAE donors held and gate comparisons inactive for update <= warmup
    eval_interval: int = 50       # updates between evaluations (0: final only)
    eval_episodes: int = 10
    eval_seed: int = 5_000_001
    eval_deterministic: bool = True
    save_interval: int = 250      # updates between checkpoints (0: final only)
    log_interval: int = 10
    torch_threads: int = 0        # 0 keeps torch's default
    audit: bool = True            # label-based audit metrics in the logs (evaluation side only)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if self.total_timesteps <= 0 or self.warmup_updates < 0:
            raise ValueError("total_timesteps must be positive and warmup_updates nonnegative")


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_hash(agents) -> str:
    h = hashlib.sha256()
    for agent in agents:
        for key, value in agent.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except Exception:  # psutil is optional
        return None


class JsonlWriter:
    def __init__(self, path: Path):
        self.stream = Path(path).open("a", encoding="utf-8", buffering=1)

    def write(self, row):
        self.stream.write(json.dumps(row, default=_json_default) + "\n")

    def close(self):
        self.stream.close()


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


class Runner:
    """Rollout state between updates."""
    def __init__(self, adapter, agents, cfg: PPOConfig, device):
        self.adapter, self.agents, self.cfg, self.device = adapter, agents, cfg, device
        n = adapter.n_agents
        self.actor_h = torch.zeros((n, cfg.recurrent_hidden_dim), device=device)
        self.critic_h = torch.zeros_like(self.actor_h)
        self.obs = None
        self.episode_return = np.zeros(n)
        self.episode_events: Dict[str, np.ndarray] = {}
        self.episode_length = 0

    def reset(self, seed=None):
        self.obs = np.asarray(self.adapter.reset(seed=seed), dtype=np.float32)
        self.actor_h.zero_()
        self.critic_h.zero_()
        self.episode_return[:] = 0
        self.episode_events = {}
        self.episode_length = 0

    def collect(self, gate: Optional[RewardProbeGate]):
        adapter, agents, cfg, device = self.adapter, self.agents, self.cfg, self.device
        n, steps, dim, actions_n = adapter.n_agents, cfg.rollout_steps, adapter.obs_dim, adapter.n_actions
        H = cfg.recurrent_hidden_dim
        obs_buf = torch.zeros((steps, n, dim))
        actions_buf = torch.zeros((steps, n), dtype=torch.long)
        masks_buf = torch.ones((steps, n, actions_n))
        logp_buf = torch.zeros((steps, n))
        rewards_buf = torch.zeros((steps, n))
        timeout_buf = torch.zeros((steps, n))
        dones_buf = torch.zeros((steps,))
        values_buf = torch.zeros((steps, n))
        actor_h_buf = torch.zeros((steps, n, H))
        critic_h_buf = torch.zeros((steps, n, H))
        contact = np.zeros((n, n), dtype=bool)
        timeout_obs: Dict[int, torch.Tensor] = {}
        episodes: List[dict] = []
        for step in range(steps):
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=device)
            obs_buf[step] = obs_t.cpu()
            actor_h_buf[step] = self.actor_h.cpu()
            critic_h_buf[step] = self.critic_h.cpu()
            actions = []
            for i, net in enumerate(agents):
                action, logp, value, ah, ch = net.act(obs_t[i], self.actor_h[i], self.critic_h[i])
                actions.append(int(action.item()))
                actions_buf[step, i] = actions[-1]
                logp_buf[step, i] = float(logp.item())
                values_buf[step, i] = float(value.item())
                self.actor_h[i] = ah.squeeze(0)
                self.critic_h[i] = ch.squeeze(0)
            next_obs, rewards, terminated, truncated, info = adapter.step(np.asarray(actions))
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            if gate is not None and gate.active:
                gate.observe(list(self.obs), np.asarray(actions), list(next_obs), rewards)
            contact |= np.asarray(adapter.contact_matrix(), dtype=bool)
            terminal = bool(terminated or truncated)
            if truncated and not terminated:  # bootstrap the final observation with its pre-reset hidden state
                t_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                timeout_obs[step] = t_obs.cpu()
                with torch.no_grad():
                    for i, net in enumerate(agents):
                        v, _ = net._critic_step(t_obs[i].unsqueeze(0), self.critic_h[i].unsqueeze(0))
                        timeout_buf[step, i] = float(v.item())
            rewards_buf[step] = torch.as_tensor(rewards)
            dones_buf[step] = float(terminal)
            self.episode_return += rewards
            self.episode_length += 1
            for key, value in (info.get("events") or {}).items():
                self.episode_events[key] = self.episode_events.get(key, 0) + np.asarray(value, dtype=np.float64)
            if terminal:
                episodes.append(dict(returns=self.episode_return.tolist(), length=self.episode_length,
                                     events={k: v.tolist() for k, v in self.episode_events.items()}))
                self.reset()
                next_obs = self.obs
            self.obs = next_obs
        last_obs = torch.as_tensor(self.obs, dtype=torch.float32)
        last_values = torch.zeros((n,))
        with torch.no_grad():
            for i, net in enumerate(agents):
                v, _ = net._critic_step(last_obs[i].unsqueeze(0).to(device), self.critic_h[i].unsqueeze(0))
                last_values[i] = float(v.item())
        rollout = Rollout(obs=obs_buf, actions=actions_buf, action_masks=masks_buf, old_log_probs=logp_buf,
                          rewards=rewards_buf, dones=dones_buf, values=values_buf, actor_h=actor_h_buf,
                          critic_h=critic_h_buf, last_values=last_values, timeout_values=timeout_buf)
        return rollout, contact, timeout_obs, last_obs, episodes


def evaluate(make_adapter: Callable, agents, ppo_cfg: PPOConfig, cfg: TrainerConfig, device, team_ids=None):
    """Fixed-seed episodes with a separate environment; never enters training buffers or the gate."""
    adapter = make_adapter()
    rows = []
    for e in range(cfg.eval_episodes):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.eval_seed + e)
            obs = np.asarray(adapter.reset(seed=cfg.eval_seed + e), dtype=np.float32)
            n = adapter.n_agents
            ah = torch.zeros((n, ppo_cfg.recurrent_hidden_dim), device=device)
            ch = torch.zeros_like(ah)
            total, events, done = np.zeros(n), {}, False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = []
                for i, net in enumerate(agents):
                    a, _, _, ah_i, ch_i = net.act(obs_t[i], ah[i], ch[i], deterministic=cfg.eval_deterministic)
                    actions.append(int(a.item()))
                    ah[i], ch[i] = ah_i.squeeze(0), ch_i.squeeze(0)
                obs, rewards, terminated, truncated, info = adapter.step(np.asarray(actions))
                obs = np.asarray(obs, dtype=np.float32)
                total += np.asarray(rewards, dtype=np.float64)
                for key, value in (info.get("events") or {}).items():
                    events[key] = events.get(key, 0) + np.asarray(value, dtype=np.float64)
                done = bool(terminated or truncated)
        rows.append(dict(returns=total.tolist(), events={k: v.tolist() for k, v in events.items()}))
    summary = dict(eval_return_mean=float(np.mean([np.mean(r["returns"]) for r in rows])),
                   eval_return_per_agent=np.mean([r["returns"] for r in rows], axis=0).tolist())
    keys = sorted({k for r in rows for k in r["events"]})
    for key in keys:
        per_agent = np.mean([r["events"].get(key, np.zeros(adapter.n_agents)) for r in rows], axis=0)
        summary[f"eval_{key}_per_agent"] = per_agent.tolist()
        summary[f"eval_{key}_total"] = float(per_agent.sum())
    if team_ids is not None and hasattr(adapter, "team_summary"):
        summary.update(adapter.team_summary(rows, team_ids))  # evaluation-only label use
    return summary


def run_training(make_adapter: Callable, out_dir, cfg: TrainerConfig, ppo_cfg: Optional[PPOConfig] = None,
                 gate_cfg: Optional[RewardGateConfig] = None, max_cfg: Optional[MaxGAEConfig] = None,
                 extra_config: Optional[dict] = None) -> Dict[str, object]:
    ppo_cfg = ppo_cfg or PPOConfig()
    spec = METHODS[cfg.method]
    out_dir = Path(out_dir)
    if (out_dir / "final.pt").exists():
        raise FileExistsError(f"{out_dir} already holds a finished run; use a new output folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    device = torch.device(cfg.device)
    set_global_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)                  # PPO minibatch order (as the RWARE trainer)
    donor_rng = np.random.default_rng(cfg.seed * 1000 + 17)  # random_gate draws (as RWARE Max-GAE plans)
    adapter = make_adapter()
    n = adapter.n_agents
    agents = [RecurrentActorCritic(adapter.obs_dim, adapter.n_actions, ppo_cfg.mlp_hidden_dim,
                                   ppo_cfg.recurrent_hidden_dim).to(device) for _ in range(n)]
    optimizers = [torch.optim.Adam(a.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5) for a in agents]
    gate = None
    if spec["gate"]:
        gate_cfg = replace(gate_cfg or RewardGateConfig(), contact_snapshot=True)  # D103 for every gate method
        gate = RewardProbeGate(n, gate_cfg, adapter.projection)
    max_cfg = max_cfg or MaxGAEConfig(donors="gate", scale="shrink")
    lam_f = ppo_cfg.gae_lambda if max_cfg.gae_lambda is None else max_cfg.gae_lambda
    critic_params = sum(p.numel() for p in agents[0].critic_parameters())
    view_bytes = adapter.projection.view_size * np.dtype(getattr(adapter.projection, "view_dtype", np.uint8)).itemsize
    (out_dir / "config.json").write_text(json.dumps(dict(
        trainer=asdict(cfg), ppo=asdict(ppo_cfg), gate=asdict(gate.config) if gate else None,
        max_gae=asdict(max_cfg), lambda_f=lam_f, method_spec=spec, n_agents=n, obs_dim=adapter.obs_dim,
        n_actions=adapter.n_actions, probe_view_size=adapter.projection.view_size,
        critic_parameters=critic_params, **(extra_config or {})), indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "initial_weights_sha256.txt").write_text(model_hash(agents) + "\n", encoding="utf-8")
    metrics_log, gate_log, eval_log = (JsonlWriter(out_dir / f) for f in ("metrics.jsonl", "gate.jsonl", "eval.jsonl"))
    team_ids = None

    def labels():  # evaluation / Oracle only
        nonlocal team_ids
        if team_ids is None:
            team_ids = np.asarray(adapter.oracle_team_ids())
        return team_ids

    runner = Runner(adapter, agents, ppo_cfg, device)
    runner.reset(seed=cfg.seed)
    total_updates = math.ceil(cfg.total_timesteps / ppo_cfg.rollout_steps)
    previous_donors = None
    start = time.perf_counter()
    try:
        for update in range(1, total_updates + 1):
            t0 = time.perf_counter()
            if gate is not None:
                gate.snapshot()
                gate.active = True
            try:
                rollout, contact, timeout_obs, last_obs, episodes = runner.collect(gate)
            finally:
                if gate is not None:
                    gate.active = False
            t1 = time.perf_counter()
            held = update <= cfg.warmup_updates
            approved = mature = None
            if gate is not None:
                gate(update, contact, active=not held)
                mature, approved = gate.maturity(), gate.approved()
            t2 = time.perf_counter()
            _, own_returns = compute_gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values,
                                         ppo_cfg.gamma, ppo_cfg.gae_lambda, rollout.timeout_values)
            ev = explained_variance(rollout.values, own_returns)
            critic_mature = (np.ones(n, dtype=bool) if max_cfg.min_critic_ev is None else
                             np.isfinite(ev.numpy()) & (ev.numpy() >= max_cfg.min_critic_ev))
            rule = spec["donors"]
            if held or rule is None:
                donors = np.zeros((n, n), dtype=bool)
            elif rule == "shuffled":
                donors = shuffled_gate_donors(contact, approved, donor_rng) & critic_mature[None, :]
            else:
                ids = labels() if rule == "oracle" else np.zeros(n, dtype=np.int64)
                donors = select_donors(rule, ids, contact, donor_rng, approved=approved, mature=mature,
                                       critic_mature=critic_mature)
            extra = dict(max_gae_held=float(held),
                         max_gae_contact_arcs=float(np.sum(contact & ~np.eye(n, dtype=bool))),
                         max_gae_donor_arcs=float(donors.sum()), max_gae_receivers=float(donors.any(1).sum()),
                         max_gae_arc_changes=float(np.sum(donors ^ previous_donors)) if previous_donors is not None
                         else 0.0)
            previous_donors = donors
            extra.update({f"max_gae_critic_ev_agent_{k}": float(v) for k, v in enumerate(ev)})
            t3 = time.perf_counter()
            if donors.any():
                f_values, f_last, f_timeout, active, b, raw = max_targets(agents, rollout, donors, timeout_obs,
                                                                          last_obs, max_cfg, device)
                valid, _ = warm_mask(rollout.dones, max_cfg.window)
                actor, returns, adv_self, adv_max = blended_advantages(
                    rollout, ppo_cfg.gamma, ppo_cfg.gae_lambda, lam_f, max_cfg.eta, f_values, f_last, f_timeout,
                    active, scale=max_cfg.scale, valid=valid)
                t4 = time.perf_counter()
                ppo_metrics = ppo_update(agents, optimizers, rollout, ppo_cfg, device, rng, advantages=actor,
                                         returns=returns)
                diag, diag_row = advantage_diagnostics(actor, adv_self, adv_max, b, raw, active, valid, max_cfg.cap)
                extra.update(diag)
            else:
                t4 = time.perf_counter()
                ppo_metrics = ppo_update(agents, optimizers, rollout, ppo_cfg, device, rng)
                diag_row = {}
            t5 = time.perf_counter()
            env_steps = update * ppo_cfg.rollout_steps
            # Communication cost (approximate): probe views both ways + responses; critic weights per donor arc.
            pairs = (gate.last_log.get("ac_pairs", []) if gate is not None and gate.last_probe_update == update
                     else [])
            exchanged = [p for p in pairs if p.get("probes", 0) > 0]
            probe_bytes = sum(p["probes"] * (2 * view_bytes + 1) + 2 * p["probes"] * 4 for p in exchanged)
            row = dict(update=update, env_steps=env_steps, method=cfg.method, **ppo_metrics, **extra,
                       cost_probe_exchanges=float(len(exchanged)), cost_probe_bytes=float(probe_bytes),
                       cost_critic_bytes=float(donors.sum() * critic_params * 4),
                       time_rollout_s=t1 - t0, time_gate_s=t2 - t1, time_max_gae_s=t4 - t3,
                       time_ppo_s=t5 - t4, sps=env_steps / max(time.perf_counter() - start, 1e-9),
                       rss_mb=_rss_mb())
            if gate is not None:
                row.update(gate.last_metrics)
                row.update(max_gae_reward_mature_agents=float(mature.sum()), max_gae_approved_arcs=float(approved.sum()))
            if episodes:
                row.update(episodes=len(episodes),
                           episode_return_mean=float(np.mean([np.mean(e["returns"]) for e in episodes])),
                           episode_return_per_agent=np.mean([e["returns"] for e in episodes], axis=0).tolist())
                for key in sorted({k for e in episodes for k in e["events"]}):
                    row[f"episode_{key}_per_agent"] = np.mean(
                        [e["events"].get(key, np.zeros(n)) for e in episodes], axis=0).tolist()
            if cfg.audit:  # evaluation-side accounting with ground-truth teams; never fed back
                same = same_team_matrix(labels())
                row.update(max_gae_same_team_arcs=float(np.sum(donors & same)),
                           max_gae_cross_team_arcs=float(np.sum(donors & ~same & ~np.eye(n, dtype=bool))))
                if gate is not None:
                    row.update(max_gae_approved_same_team_arcs=float(np.sum(approved & same)),
                               max_gae_approved_cross_team_arcs=float(np.sum(approved & ~same & ~np.eye(n, dtype=bool))),
                               max_gae_same_team_approval_coverage=float(np.sum(approved & same) / max(1, same.sum())))
                    if pairs:
                        row.update(audit(gate, labels()))
            metrics_log.write(row)
            gate_log.write(dict(update=update, env_steps=env_steps, held=held, contact=contact, donors=donors,
                                matrix_convention="receiver_donor", critic_ev=[float(v) for v in ev],
                                **(dict(approved=approved, reward_mature=mature, rounds=gate.rounds, ema=gate.ema,
                                        pairs=pairs) if gate is not None else {}), **diag_row))
            if cfg.log_interval and update % cfg.log_interval == 0:
                print(f"update {update}/{total_updates} steps {env_steps} return "
                      f"{row.get('episode_return_mean', float('nan')):.3f} donors {int(donors.sum())} "
                      f"approved {int(approved.sum()) if approved is not None else '-'} sps {row['sps']:.0f}",
                      flush=True)
            if cfg.eval_interval and update % cfg.eval_interval == 0:
                eval_log.write(dict(update=update, env_steps=env_steps,
                                    **evaluate(make_adapter, agents, ppo_cfg, cfg, device,
                                               labels() if cfg.audit else None)))
            if cfg.save_interval and update % cfg.save_interval == 0:
                save_checkpoint(out_dir / f"checkpoint_{update:05d}.pt", agents, optimizers, gate, cfg, update,
                                env_steps, rng, donor_rng)
        final_eval = evaluate(make_adapter, agents, ppo_cfg, cfg, device, labels() if cfg.audit else None)
        eval_log.write(dict(update=total_updates, env_steps=total_updates * ppo_cfg.rollout_steps, final=True,
                            **final_eval))
        save_checkpoint(out_dir / "final.pt", agents, optimizers, gate, cfg, total_updates,
                        total_updates * ppo_cfg.rollout_steps, rng, donor_rng)
    finally:
        for log in (metrics_log, gate_log, eval_log):
            log.close()
    result = dict(status="complete", updates=total_updates, env_steps=total_updates * ppo_cfg.rollout_steps,
                  seconds=time.perf_counter() - start, final_eval=final_eval)
    (out_dir / "status.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


def save_checkpoint(path, agents, optimizers, gate, cfg, update, env_steps, rng, donor_rng):
    torch.save(dict(format="pgct_core.trainer/1", config=asdict(cfg), update=update, env_steps=env_steps,
                    agents=[a.state_dict() for a in agents], optimizers=[o.state_dict() for o in optimizers],
                    reward_gate=gate.state_dict() if gate is not None else None,
                    rng=dict(numpy=rng.bit_generator.state, donor=donor_rng.bit_generator.state,
                             torch=torch.get_rng_state())), path)


def load_agents(path, obs_dim, n_actions, ppo_cfg: Optional[PPOConfig] = None, device="cpu"):
    """Rebuild the actor-critics of a saved checkpoint (evaluation, analysis)."""
    ppo_cfg = ppo_cfg or PPOConfig()
    payload = torch.load(path, map_location=device, weights_only=False)
    agents = []
    for state in payload["agents"]:
        net = RecurrentActorCritic(obs_dim, n_actions, ppo_cfg.mlp_hidden_dim, ppo_cfg.recurrent_hidden_dim)
        net.load_state_dict(state)
        agents.append(net.to(device))
    return agents, payload
